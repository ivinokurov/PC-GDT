"""Six condition maps C1..C6 (UE5/Colosseum export stand-in) + encodings.

9-channel concatenation (normalised class index) vs 16-channel one-hot,
exactly as in the paper.  C6 is the ASPECT (steepest-descent direction,
cos/sin), matching the paper's "slope orientation".
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
from .geometry import Scene, illuminated_faces, shadow_points_on_face

CLASS_NAMES = ["background", "ground", "road", "flat_roof",
               "pitched_slope", "ridge", "canopy", "shadow"]
N_CLASSES = len(CLASS_NAMES)
IDX_BACKGROUND, IDX_GROUND, IDX_ROAD, IDX_FLAT = range(4)
IDX_SLOPE, IDX_RIDGE, IDX_CANOPY, IDX_SHADOW = 4, 5, 6, 7


@dataclass
class ConditionMaps:
    semantic: np.ndarray        # (H, W) uint8
    depth: np.ndarray           # (H, W) [0, 1]
    normals: np.ndarray         # (H, W, 3) [-1, 1]
    shadow_mask: np.ndarray     # (H, W) {0,1}
    ridge_mask: np.ndarray      # (H, W) {0,1}
    slope_orient: np.ndarray    # (H, W, 2) cos/sin of steepest descent
    sun_azimuth: float
    sun_elevation: float

    @property
    def height(self): return self.semantic.shape[0]
    @property
    def width(self): return self.semantic.shape[1]

    def concatenated_normalised(self) -> np.ndarray:
        sem = self.semantic.astype(np.float32) / (N_CLASSES - 1)
        return np.concatenate([sem[..., None], self.depth[..., None], self.normals,
                               self.shadow_mask[..., None].astype(np.float32),
                               self.ridge_mask[..., None].astype(np.float32),
                               self.slope_orient], axis=-1).astype(np.float32)

    def concatenated_onehot(self) -> np.ndarray:
        oh = np.eye(N_CLASSES, dtype=np.float32)[self.semantic]
        return np.concatenate([oh, self.depth[..., None], self.normals,
                               self.shadow_mask[..., None].astype(np.float32),
                               self.ridge_mask[..., None].astype(np.float32),
                               self.slope_orient], axis=-1).astype(np.float32)


def world_grids(H, W, extent):
    yy, xx = np.mgrid[:H, :W]
    xw = xx / (W - 1) * 2 * extent - extent
    yw = extent - yy / (H - 1) * 2 * extent
    return xw.astype(np.float64), yw.astype(np.float64)


def world_to_pixel(x, y, extent, H, W):
    px = np.round((np.asarray(x) + extent) / (2 * extent) * (W - 1)).astype(np.int64)
    py = np.round((extent - np.asarray(y)) / (2 * extent) * (H - 1)).astype(np.int64)
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    return py[ok], px[ok]


def pixel_to_world(px, py, extent, H, W):
    x = np.asarray(px, float) / (W - 1) * 2 * extent - extent
    y = extent - np.asarray(py, float) / (H - 1) * 2 * extent
    return x, y


def _fill_polygon(mask, poly_xy, extent, value=1):
    H, W = mask.shape
    ys, xs = poly_xy[:, 1], poly_xy[:, 0]
    to_row = lambda y: (extent - y) / (2 * extent) * (H - 1)
    ymin = max(int(np.floor(to_row(ys.max()))), 0)
    ymax = min(int(np.ceil(to_row(ys.min()))), H - 1)
    n = len(poly_xy)
    for r in range(ymin, ymax + 1):
        y_w = extent - r / (H - 1) * 2 * extent          # pixel-centre convention
        xints = []
        for i in range(n):
            (x0, y0), (x1, y1) = poly_xy[i], poly_xy[(i + 1) % n]
            if (y0 <= y_w < y1) or (y1 <= y_w < y0):
                t = (y_w - y0) / (y1 - y0 + 1e-12)
                xints.append(x0 + t * (x1 - x0))
        xints.sort()
        for j in range(0, len(xints) - 1, 2):
            c0 = int((xints[j] + extent) / (2 * extent) * (W - 1))
            c1 = int((xints[j + 1] + extent) / (2 * extent) * (W - 1))
            mask[r, max(c0, 0):min(c1, W - 1) + 1] = value


def _draw_line(mask, p0, p1, extent, width=2):
    H, W = mask.shape
    dxy = np.asarray(p1)[:2] - np.asarray(p0)[:2]
    steps = int(np.hypot(*dxy) / (2 * extent / W)) + 2
    for t in np.linspace(0, 1, steps):
        p = np.asarray(p0) + t * (np.asarray(p1) - np.asarray(p0))
        py, px = world_to_pixel([p[0]], [p[1]], extent, H, W)
        for r0, c0 in zip(py, px):
            r1, r2 = max(r0 - width // 2, 0), min(r0 + width // 2, H - 1)
            c1, c2 = max(c0 - width // 2, 0), min(c0 + width // 2, W - 1)
            mask[r1:r2 + 1, c1:c2 + 1] = 1


def _disk_pad(mask, k, op):
    pad = k // 2
    sp = np.pad(mask, pad, mode="constant")
    acc = np.zeros_like(sp) if op == "max" else np.ones_like(sp)
    for dy in range(k):
        for dx in range(k):
            if (dy - pad) ** 2 + (dx - pad) ** 2 <= pad ** 2:
                sh = np.roll(np.roll(sp, dy, 0), dx, 1)
                acc = np.maximum(acc, sh) if op == "max" else np.minimum(acc, sh)
    return acc[pad:pad + mask.shape[0], pad:pad + mask.shape[1]]


def rasterise_scene(scene: Scene, H=128, W=128, extent=60.0,
                    n_shadow_samples=2048, seed=0) -> ConditionMaps:
    """Deterministic raster of all six maps from the geometric twin.
    Depth/normals are filled densely over each face; the shadow mask uses the
    exact ray-plane projection WITH sun-ray occlusion, then a small closing
    (fills sampling holes, does not inflate), restricted to roof surfaces."""
    rng = np.random.default_rng(seed)
    xw, yw = world_grids(H, W, extent)
    sem = np.full((H, W), IDX_GROUND, dtype=np.uint8)
    depth = np.full((H, W), 0.15, dtype=np.float32)
    normals = np.zeros((H, W, 3), dtype=np.float32); normals[..., 2] = 1.0
    shadow = np.zeros((H, W), dtype=np.uint8)
    ridge = np.zeros((H, W), dtype=np.uint8)
    orient = np.zeros((H, W, 2), dtype=np.float32)
    roof_region = np.zeros((H, W), dtype=np.uint8)
    canopy_region = np.zeros((H, W), dtype=np.uint8)

    sem[H // 2 - 4:H // 2 + 4, :] = IDX_ROAD

    for face in scene.faces:
        m = np.zeros((H, W), dtype=np.uint8)
        _fill_polygon(m, face.vertices[:, :2], extent, 1)
        sel = m == 1
        roof_region |= m
        sem[sel] = IDX_SLOPE if face.kind == "pitched" else IDX_FLAT
        n = face.normal()
        normals[sel] = n
        if abs(n[2]) > 1e-6:
            z = (face.plane_offset() - n[0] * xw - n[1] * yw) / n[2]
        else:
            z = np.full_like(xw, face.vertices[:, 2].mean())
        depth[sel] = np.clip(z[sel] / 10.0, 0, 1).astype(np.float32)
        g = n[:2]; ng = float(np.linalg.norm(g))
        orient[sel] = (-g / ng) if ng > 1e-6 else 0.0   # aspect = steepest descent

    for r0, r1 in scene.ridge_segments:
        _draw_line(ridge, r0, r1, extent, width=2)
    sem[ridge == 1] = IDX_RIDGE

    for cp in scene.canopies:
        rr = ((xw - cp.centre[0]) ** 2 + (yw - cp.centre[1]) ** 2) / cp.radius ** 2
        foot = rr < 1.0
        canopy_region |= foot.astype(np.uint8)
        sem[foot] = IDX_CANOPY
        depth[foot] = np.clip(cp.centre[2] / 10.0, 0, 1)
        normals[foot] = np.stack([(xw[foot] - cp.centre[0]) / cp.radius,
                                  (yw[foot] - cp.centre[1]) / cp.radius,
                                  np.sqrt(np.clip(1 - rr[foot], 0, 1))], axis=-1)

    d = scene.sun_vector
    for cp in scene.canopies:
        for face in scene.faces:
            hits = shadow_points_on_face(cp, face, d, n_shadow_samples,
                                         scene.canopies, rng)
            if len(hits):
                py, px = world_to_pixel(hits[:, 0], hits[:, 1], extent, H, W)
                shadow[py, px] = 1
    shadow = _disk_pad(_disk_pad(shadow, 3, "max"), 3, "min")   # closing
    shadow &= roof_region & (1 - canopy_region)
    sem[(shadow == 1) & (sem != IDX_RIDGE)] = IDX_SHADOW

    return ConditionMaps(sem, depth, normals, shadow, ridge, orient,
                         scene.sun_azimuth, scene.sun_elevation)


def perturb_conditions(maps: ConditionMaps, rng,
                       xi=(0.04, 0.18, 0.14, 0.06, 0.09, 0.15)) -> ConditionMaps:
    """Stand-in for the pseudo-condition noise of Appendix C (per-channel
    magnitudes from Table 1).  Used at TRAINING time only; inference gets
    clean UE5 maps."""
    sem = maps.semantic.copy()
    flip = rng.random(sem.shape) < xi[0]
    sem[flip] = rng.integers(0, N_CLASSES, size=flip.sum())
    depth = np.clip(maps.depth + rng.normal(0, xi[1], maps.depth.shape), 0, 1)
    normals = maps.normals + rng.normal(0, xi[2], maps.normals.shape)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-9
    sh = maps.shadow_mask.copy(); f = rng.random(sh.shape) < xi[3]; sh[f] ^= 1
    rg = maps.ridge_mask.copy(); f = rng.random(rg.shape) < xi[4]; rg[f] ^= 1
    ang = rng.normal(0, xi[5], maps.slope_orient.shape[:2])
    c, s = np.cos(ang), np.sin(ang)
    o = maps.slope_orient
    orient = np.stack([c * o[..., 0] - s * o[..., 1],
                       s * o[..., 0] + c * o[..., 1]], axis=-1)
    return replace(maps, semantic=sem.astype(np.uint8), depth=depth.astype(np.float32),
                   normals=normals.astype(np.float32), shadow_mask=sh,
                   ridge_mask=rg, slope_orient=orient.astype(np.float32))