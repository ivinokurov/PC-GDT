"""Procedural target-domain render I_real (stand-in for real UAV photos /
photorealistic UE5 output) + automatic labels A = (boxes, classes, masks).
Used as x0 for diffusion training and as the reference for metrics.
NOT photorealistic -- the pipeline structure is identical when real images
are plugged in via dataset.SceneDataset.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from .geometry import Scene, shadow_points_on_face
from .conditions import (ConditionMaps, IDX_GROUND, IDX_ROAD, IDX_FLAT, IDX_SLOPE,
                         IDX_RIDGE, IDX_CANOPY, IDX_SHADOW, world_to_pixel, world_grids)

_ALBEDO = {IDX_GROUND: (0.36, 0.43, 0.30), IDX_ROAD: (0.28, 0.28, 0.30),
           IDX_FLAT: (0.45, 0.44, 0.42), IDX_SLOPE: (0.58, 0.36, 0.28),
           IDX_RIDGE: (0.30, 0.22, 0.20), IDX_CANOPY: (0.20, 0.40, 0.18),
           IDX_SHADOW: (0.58, 0.36, 0.28)}


def _value_noise(H, W, rng, cell=8):
    g = torch.from_numpy(rng.random((1, 1, H // cell + 2, W // cell + 2))).float()
    n = F.interpolate(g, size=(H, W), mode="bilinear", align_corners=False)
    return n[0, 0].numpy()


def render_scene(scene: Scene, maps: ConditionMaps, extent=60.0, rng=None) -> np.ndarray:
    """Returns (3, H, W) float32 in [-1, 1].  Lambert shading with the exact
    sun vector, physical shadow darkening, foliage/roof textures, haze, noise."""
    rng = rng or np.random.default_rng(0)
    H, W = maps.height, maps.width
    alb = np.zeros((H, W, 3), np.float32)
    for idx, col in _ALBEDO.items():
        alb[maps.semantic == idx] = col
    tex = _value_noise(H, W, rng)
    stripes = 0.5 + 0.5 * np.sin(np.arange(H)[:, None] * 1.4)
    roof = (maps.semantic == IDX_SLOPE) | (maps.semantic == IDX_FLAT)
    alb[roof] *= (0.85 + 0.3 * stripes * tex[..., None])[roof]
    veg = maps.semantic == IDX_CANOPY
    alb[veg] *= (0.7 + 0.6 * tex[..., None])[veg]
    alb[~roof & ~veg] *= (0.85 + 0.3 * tex[..., None])[~roof & ~veg]

    d = scene.sun_vector
    lit = np.clip(-(maps.normals @ d), 0, 1)
    direct = 0.75 * lit
    direct[maps.shadow_mask == 1] *= 0.15
    img = alb * (0.32 + direct[..., None])
    img = img * (1 - 0.08) + 0.08 * np.array([0.75, 0.80, 0.88])   # haze
    img += rng.normal(0, 0.02, img.shape)
    img = np.clip(img, 0, 1)
    return (img.astype(np.float32) * 2 - 1).transpose(2, 0, 1)


def scene_labels(scene: Scene, maps: ConditionMaps, extent=60.0,
                 n_samples=1024, seed=0) -> dict:
    """Automatic annotation A inherited from the physical render:
    per-canopy boxes + per-canopy shadow boxes (class by roof kind),
    union masks M_shadow / M_tree, world canopy centres (for PGPF pairing)."""
    rng = np.random.default_rng(seed)
    H, W = maps.height, maps.width
    d = scene.sun_vector
    boxes, centres = [], []
    for ci, cp in enumerate(scene.canopies):
        centres.append(cp.centre[:2].tolist())
        py, px = world_to_pixel([cp.centre[0] - cp.radius, cp.centre[0] + cp.radius],
                                [cp.centre[1] + cp.radius, cp.centre[1] - cp.radius],
                                extent, H, W)
        if len(py):
            boxes.append({"cls": "canopy", "canopy_id": ci,
                          "bbox": [int(px.min()), int(py.min()),
                                   int(px.max()), int(py.max())]})
        hits, kinds = [], []
        for face in scene.faces:
            h = shadow_points_on_face(cp, face, d, n_samples, scene.canopies, rng)
            if len(h):
                hits.append(h); kinds.append(face.kind)
        if hits:
            h = np.vstack(hits)
            py, px = world_to_pixel(h[:, 0], h[:, 1], extent, H, W)
            if len(py):
                cls = "shadow_pitched" if "pitched" in kinds else "shadow_flat"
                boxes.append({"cls": cls, "canopy_id": ci,
                              "bbox": [int(px.min()), int(py.min()),
                                       int(px.max()), int(py.max())],
                              "world_centroid": h.mean(0).tolist()})
    return {"boxes": boxes, "canopy_centres_world": centres,
            "M_shadow": maps.shadow_mask.copy(),
            "M_tree": (maps.semantic == IDX_CANOPY).astype(np.uint8)}