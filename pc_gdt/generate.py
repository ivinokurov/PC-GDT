"""Generation + Real-ESRGAN x2 upscaling (bicubic fallback) + label transfer
(coordinates x factor, masks nearest -- no geometric warp, as in the paper)."""
from __future__ import annotations
import json, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .model import sample
from .render import scene_labels


def upscale_image(img_hwc01: np.ndarray, factor=2) -> np.ndarray:
    """Real-ESRGAN if installed, else bicubic (documented fallback).
    Real-ESRGAN is a single-forward GAN (no diffusion steps)."""
    try:
        from realesrgan import RealESRGANer           # optional heavy dep
        from basicsr.archs.rrdbnet_arch import RRDBNet
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=2)
        up = RealESRGANer(scale=2, model_path="weights/RealESRGAN_x2plus.pth", model=net)
        out, _ = up.enhance((img_hwc01 * 255).astype(np.uint8))
        return out.astype(np.float32) / 255
    except Exception:
        t = torch.from_numpy(img_hwc01.transpose(2, 0, 1))[None]
        t = F.interpolate(t, scale_factor=factor, mode="bicubic", align_corners=False)
        return t[0].clamp(0, 1).numpy().transpose(1, 2, 0)


def upscale_labels_real_esrgan(image_512, label_coords, factor=2):
    """Paper's label transfer: multiply coordinates by the factor."""
    return np.asarray(label_coords) * factor


def transfer_labels(labels: dict, factor: int) -> dict:
    out = {"boxes": [], "canopy_centres_world": labels["canopy_centres_world"]}
    for b in labels["boxes"]:
        bb = (np.asarray(b["bbox"]) * factor).tolist()
        out["boxes"].append({**b, "bbox": bb})
    for k in ("M_shadow", "M_tree"):
        m = torch.from_numpy(labels[k])[None, None].float()
        m = F.interpolate(m, scale_factor=factor, mode="nearest")
        out[k] = m[0, 0].numpy().astype(np.uint8)
    return out


def generate_dataset(model, ds, indices, cfg, log=print):
    os.makedirs(cfg.out, exist_ok=True)
    manifest = []
    for idx in indices:
        item = ds[idx]
        cond, emb = item["cond"][None].to(cfg.device), item["sun"][None].to(cfg.device)
        img = sample(model, cond, emb, shape=(1, 3, ds.H, ds.W),
                     guidance_w=cfg.w, steps=cfg.gen_steps,
                     T_total=cfg.T_total, seed=cfg.seed + int(idx))
        img01 = ((img[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)
        up = upscale_image(img01, cfg.upscale)
        sc, maps = ds.scenes[idx], ds.maps[idx]
        labels = transfer_labels(scene_labels(sc, maps, ds.extent, seed=cfg.seed + idx),
                                 cfg.upscale)
        name = f"scene_{idx:04d}"
        Image.fromarray((up * 255).astype(np.uint8)).save(os.path.join(cfg.out, name + ".png"))
        np.savez_compressed(os.path.join(cfg.out, name + "_maps.npz"),
                            semantic=maps.semantic, shadow=maps.shadow_mask,
                            ridge=maps.ridge_mask)
        Image.fromarray(labels["M_shadow"] * 255).save(os.path.join(cfg.out, name + "_mshadow.png"))
        Image.fromarray(labels["M_tree"] * 255).save(os.path.join(cfg.out, name + "_mtree.png"))
        manifest.append({"id": idx, "file": name + ".png", "maps": name + "_maps.npz",
                         "mshadow": name + "_mshadow.png", "mtree": name + "_mtree.png",
                         "size": [up.shape[1], up.shape[0]], "base_size": [ds.W, ds.H],
                         "factor": cfg.upscale, "extent": ds.extent,
                         "scene": {"az": sc.sun_azimuth, "el": sc.sun_elevation,
                                   "canopies": [{"centre": c.centre.tolist(), "radius": c.radius}
                                                for c in sc.canopies]},
                         "labels": {"boxes": labels["boxes"],
                                    "canopy_centres_world": labels["canopy_centres_world"]}})
        log(f"[gen] {name}: {up.shape[1]}x{up.shape[0]}, {len(labels['boxes'])} boxes")
    with open(os.path.join(cfg.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest