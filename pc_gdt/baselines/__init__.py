"""Generative baselines for remote sensing (paper sec. 6.1): DiffusionSat, DiffCR.

Both are fine-tuned on the same real images as PC-ControlNet and keep their
native conditioning; neither receives physical scene maps, the sun embedding,
FiLM or the physics penalty.  The same PGPF and COCO export are applied to all.
"""
from .diffusionsat import DiffusionSatAdapter, metadata_channels
from .diffcr import DiffCRAdapter

__all__ = ["DiffusionSatAdapter", "DiffCRAdapter", "metadata_channels"]