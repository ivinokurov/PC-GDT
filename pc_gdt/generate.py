"""Generation, super-resolution and label transfer (paper, sec. 3).

The trained PC-ControlNet receives clean UE5 condition maps; the 512x512
output is upscaled x2 to 1024x1024 by Real-ESRGAN (RealESRGAN_x2plus, a
single-forward GAN; a documented bicubic fallback keeps the pipeline runnable
without the checkpoint).  The annotation A inherited from the physical render
is transferred by multiplying box coordinates by the upscale factor and
resizing masks with nearest neighbour -- no geometric warp, which is why the
shadow IoU changes only within noise (0.87 -> 0.86) and the azimuth error
4.2 deg -> 4.4 deg after super-resolution.

`write_sample` is the single persistence routine shared by PC-GDT and the
generative baselines (DiffusionSat / DiffCR), so every method emits the same
manifest format and the identical PGPF (filtering.filter_manifest) runs on all
outputs.
"""
from __future__ import annotations
import json, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .model import sample
from .render import scene_labels


def upscale_image(img_hwc01: np.ndarray, factor: int = 2) -> np.ndarray:
    """Real-ESRGAN x2plus if installed, else bicubic (documented fallback).
    Real-ESRGAN is a GAN with a single forward pass (no diffusion steps)."""
    try:
        from realesrgan import RealESRGANer            # optional heavy dependency
        from basicsr.archs.rrdbnet_arch import RRDBNet
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                      num_block=6, num_grow_ch=32, scale=2)
        up = RealESRGANer(scale=2, model_path="weights/RealESRGAN_x2plus.pth", model=net)
        out, _ = up.enhance((img_hwc01 * 255).astype(np.uint8))
        return out.astype(np.float32) / 255
    except Exception:
        t = torch.from_numpy(img_hwc01.transpose(2, 0, 1))[None]
        t = F.interpolate(t, scale_factor=factor, mode="bicubic", align_corners=False)
        return t[0].clamp(0, 1).numpy().transpose(1, 2, 0)


def upscale_labels_real_esrgan(image_512, label_coords, factor: int = 2):
    """Paper's label transfer: multiply coordinates by the upscale factor
    (no geometric distortion).  Kept for backward compatibility."""
    return np.asarray(label_coords) * factor


def transfer_labels(labels: dict, factor: int) -> dict:
    """Transfer the inherited annotation A through the x2 super-resolution:
    box coordinates x factor, masks resized by nearest neighbour."""
    out = {"boxes": [], "canopy_centres_world": labels["canopy_centres_world"]}
    for b in labels["boxes"]:
        out["boxes"].append({**b, "bbox": (np.asarray(b["bbox"]) * factor).tolist()})
    for k in ("M_shadow", "M_tree"):
        m = torch.from_numpy(labels[k])[None, None].float()
        m = F.interpolate(m, scale_factor=factor, mode="nearest")
        out[k] = m[0, 0].numpy().astype(np.uint8)
    return out


def write_sample(out_dir, idx, img01, sc, maps, extent, base_size, factor, seed):
    """Persist one generated sample in the canonical format shared by PC-GDT
    and the baselines, so the identical PGPF runs on every method's output.

    img01 : (H, W, 3) float array in [0, 1], already super-resolved.
    Returns the manifest entry (image, condition maps, masks, scene, labels).
    """
    name = f"scene_{idx:04d}"
    Hs, Ws = img01.shape[:2]
    Image.fromarray((np.clip(img01, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(out_dir, name + ".png"))
    np.savez_compressed(os.path.join(out_dir, name + "_maps.npz"),
                        semantic=maps.semantic, shadow=maps.shadow_mask,
                        ridge=maps.ridge_mask)
    labels = transfer_labels(scene_labels(sc, maps, extent, seed=seed), factor)
    Image.fromarray(labels["M_shadow"] * 255).save(os.path.join(out_dir, name + "_mshadow.png"))
    Image.fromarray(labels["M_tree"] * 255).save(os.path.join(out_dir, name + "_mtree.png"))
    return {"id": int(idx), "file": name + ".png", "maps": name + "_maps.npz",
            "mshadow": name + "_mshadow.png", "mtree": name + "_mtree.png",
            "size": [int(Ws), int(Hs)], "base_size": [int(base_size[0]), int(base_size[1])],
            "factor": int(factor), "extent": float(extent),
            "scene": {"az": float(sc.sun_azimuth), "el": float(sc.sun_elevation),
                      "canopies": [{"centre": c.centre.tolist(), "radius": float(c.radius)}
                                   for c in sc.canopies]},
            "labels": {"boxes": labels["boxes"],
                       "canopy_centres_world": labels["canopy_centres_world"]}}


def generate_dataset(model, ds, indices, cfg, log=print):
    """Sample PC-GDT over a scene-disjoint split, upscale x2, transfer the
    inherited labels and write the manifest consumed by PGPF and COCO export."""
    os.makedirs(cfg.out, exist_ok=True)
    manifest = []
    for idx in indices:
        item = ds[idx]
        cond = item["cond"][None].to(cfg.device)
        emb = item["sun"][None].to(cfg.device)
        img = sample(model, cond, emb, shape=(1, 3, ds.H, ds.W),
                     guidance_w=cfg.w, steps=cfg.gen_steps,
                     T_total=cfg.T_total, seed=cfg.seed + int(idx))
        img01 = ((img[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)
        up = upscale_image(img01, cfg.upscale)
        entry = write_sample(cfg.out, int(idx), up, ds.scenes[idx], ds.maps[idx],
                             ds.extent, (ds.W, ds.H), cfg.upscale, cfg.seed + int(idx))
        manifest.append(entry)
        log(f"[gen] scene_{idx:04d}: {up.shape[1]}x{up.shape[0]}, "
            f"{len(entry['labels']['boxes'])} boxes")
    with open(os.path.join(cfg.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest