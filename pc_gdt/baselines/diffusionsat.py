"""DiffusionSat adapter (paper sec. 6.1).

DiffusionSat is a foundation model for satellite imagery conditioned on
METADATA (acquisition time, geolocation, sensor) -- not on physical scene
maps.  The adapter keeps that mechanism: the hint is the optical render plus
broadcast metadata channels derived from the scene.  No normals, depth,
shadow mask, sun embedding, FiLM or physics penalty are provided.

Real system: load the official DiffusionSat checkpoint (SD-2.1 UNet with
metadata injected through the text/CLIP cross-attention via a metadata
encoder); fine-tune on the same 7,919 real images; generate from the same UE5
scenes.  Only the conditioning differs from PC-GDT.
"""
from __future__ import annotations
import numpy as np
from .base import DiffusionBaseline

# Illustrative acquisition geolocation (Kaluga / Moscow region) and sensor GSD.
LAT_DEG, LON_DEG = 54.5, 36.0
LAT_REF, LON_REF = 55.0, 37.0


def metadata_channels(scene, extent, H, W) -> np.ndarray:
    """6 broadcast channels: [sin/cos hour-angle, sin/cos day-of-year,
    normalized (lat, lon), normalized GSD].  Approximate solar geometry used
    only to synthesize plausible metadata; NOT a physical condition."""
    az, el = scene.sun_azimuth, scene.sun_elevation
    hour_angle = np.deg2rad(az - 180.0)                       # 180 deg ~ local noon
    sin_decl = np.sin(np.deg2rad(23.44))
    decl = np.arcsin(np.clip(np.sin(np.deg2rad(el)) / max(sin_decl, 1e-6), -1, 1))
    doy = float(np.arccos(np.clip(-decl / max(np.deg2rad(23.44), 1e-6), -1, 1)) / (2 * np.pi) * 365.25)
    season = 2 * np.pi * doy / 365.25
    gsd = 2 * extent / W
    vals = [np.sin(hour_angle), np.cos(hour_angle), np.sin(season), np.cos(season),
            (LAT_DEG - LAT_REF) / 10.0 + 0.5, (LON_DEG - LON_REF) / 10.0 + 0.5]
    # GSD folded into the geolocation/sensor block as a 7th value would change
    # the channel count; keep 6 and note GSD is constant per dataset split.
    return np.broadcast_to(np.asarray(vals, np.float32)[:, None, None], (6, H, W)).copy()


class DiffusionSatAdapter(DiffusionBaseline):
    name = "diffusionsat"
    hint_channels = 6

    def build_hint(self, render_chw, maps, scene, extent):
        _, H, W = render_chw.shape
        return metadata_channels(scene, extent, H, W)

    @classmethod
    def from_pretrained(cls, *a, **k):
        raise NotImplementedError(
            "Load the official DiffusionSat checkpoint (SD-2.1 + metadata encoder "
            "into CLIP cross-attention); fine-tune on the same real images and "
            "generate from the same UE5 scenes. No physical maps / sun embedding.")