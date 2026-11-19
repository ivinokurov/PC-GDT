"""DiffCR adapter (paper sec. 6.1).

DiffCR is a fast conditional diffusion model for cloud removal, conditioned on
a CLOUD MASK and an optical (cloudy) image.  Following the paper, the physical
shadow mask C4 is fed as the "cloud mask" (an extra advantage for DiffCR), and
the UE5 render is the optical image.  No normals, depth, sun embedding, FiLM
or physics penalty are provided.

Real system: load the official DiffCR checkpoint; fine-tune on the same 7,919
real images with (cloudy image = UE5 render, cloud mask = C4, target = real);
generate from the same UE5 scenes.  Only the conditioning differs from PC-GDT.
"""
from __future__ import annotations
import numpy as np
from .base import DiffusionBaseline


class DiffCRAdapter(DiffusionBaseline):
    name = "diffcr"
    hint_channels = 1                      # cloud mask only (C4)

    def build_hint(self, render_chw, maps, scene, extent):
        return maps.shadow_mask.astype(np.float32)[None]   # C4 as the cloud mask

    @classmethod
    def from_pretrained(cls, *a, **k):
        raise NotImplementedError(
            "Load the official DiffCR checkpoint; condition on (optical image, "
            "cloud mask=C4); fine-tune on the same real images and generate from "
            "the same UE5 scenes. No physical maps / sun embedding.")