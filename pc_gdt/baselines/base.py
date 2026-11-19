"""Shared baseline protocol (paper sec. 6.1, "Условия сравнения").

Identical for every method: the same UE5 scenes, the same real fine-tuning
set, the same generation budget, the same DDPM schedule / sampler (w, T) as
PC-GDT, and the SAME post-generative physical filter (filtering.filter_manifest)
and COCO export.  A baseline never receives normals / depth / the physical
shadow map as a condition, the sun embedding, FiLM, or the lambda*R penalty --
those are PC-GDT's contribution.  Only the native conditioning differs:
metadata (DiffusionSat) or a cloud mask (DiffCR).

CPU stand-in: a small image-space conditional denoiser with the same interface
as the real models; real checkpoints plug in via `from_pretrained`.
"""
from __future__ import annotations
import itertools, json, math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from ..model import ResBlock, cosine_lr_with_warmup
from ..render import scene_labels


def _betas(T, b0=1e-4, b1=0.02):
    return torch.linspace(b0, b1, T)


def _sin_t(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(a), torch.sin(a)], -1)


class ConditionalDenoiser(nn.Module):
    """eps_theta(x_t, t, hint). hint = [render || native_condition].
    Image-space stand-in for the baseline UNet; no FiLM, no sun embedding."""
    def __init__(self, hint_channels: int, base_ch: int = 32):
        super().__init__()
        self.base_ch = base_ch
        self.time_mlp = nn.Sequential(nn.Linear(base_ch, base_ch * 2), nn.SiLU(),
                                      nn.Linear(base_ch * 2, base_ch))
        self.hint_enc = nn.Sequential(nn.Conv2d(hint_channels, base_ch, 3, padding=1), nn.SiLU(),
                                      nn.Conv2d(base_ch, base_ch, 3, padding=1))
        self.in_conv = nn.Conv2d(3 + base_ch + base_ch, base_ch, 3, padding=1)
        self.b1, self.b2 = ResBlock(base_ch, base_ch), ResBlock(base_ch, base_ch)
        self.out = nn.Conv2d(base_ch, 3, 3, padding=1)

    def forward(self, x_t, t, hint):
        B = x_t.shape[0]
        temb = self.time_mlp(_sin_t(t, self.base_ch))[..., None, None]
        temb = temb.expand(B, -1, *x_t.shape[2:])
        h = self.in_conv(torch.cat([x_t, temb, self.hint_enc(hint)], 1))
        return self.out(self.b2(self.b1(h)))


@torch.no_grad()
def sample_image(net, hint, shape, guidance_w=5.0, steps=30, T_total=200, seed=0, clip_x0=4.0):
    """Strided DDPM ancestral sampling with CFG on a zeroed hint -- the same
    schedule and sampler as PC-GDT (model.sample), in image space."""
    dev = next(net.parameters()).device
    g = torch.Generator(device=dev).manual_seed(seed)
    steps = min(steps, T_total)
    ab = torch.cumprod(1.0 - _betas(T_total).to(dev), 0)
    x = torch.randn(shape, generator=g, device=dev)
    null = torch.zeros_like(hint)
    ts = torch.linspace(T_total - 1, 0, steps).long().unique_consecutive()
    for i, ti in enumerate(ts):
        t = int(ti); tp = int(ts[i + 1]) if i + 1 < len(ts) else -1
        ab_t = ab[t]; ab_p = ab[tp] if tp >= 0 else torch.tensor(1.0, device=dev)
        tt = torch.full((shape[0],), t, device=dev)
        ec = net(x, tt, hint); eu = net(x, tt, null)
        eps = eu + guidance_w * (ec - eu)
        x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        if clip_x0:
            x0 = x0.clamp(-clip_x0, clip_x0)
        if tp >= 0:
            var = (1 - ab_p) / (1 - ab_t) * (1 - ab_t / ab_p)
            x = ab_p.sqrt() * x0 + (1 - ab_p - var).clamp(min=0).sqrt() * eps \
                + var.sqrt() * torch.randn(x.shape, generator=g, device=dev)
        else:
            x = x0
    return x.clamp(-1, 1)


def write_sample(out_dir, idx, img01, sc, maps, extent, base_size, factor, seed):
    """Persist one generated sample in the SAME format as PC-GDT
    (generate.generate_dataset), so the identical PGPF runs on every method."""
    name = f"scene_{idx:04d}"
    Hs, Ws = img01.shape[:2]
    Image.fromarray((img01 * 255).astype(np.uint8)).save(os.path.join(out_dir, name + ".png"))
    np.savez_compressed(os.path.join(out_dir, name + "_maps.npz"),
                        semantic=maps.semantic, shadow=maps.shadow_mask, ridge=maps.ridge_mask)
    labels = scene_labels(sc, maps, extent, seed=seed)
    ms = torch.from_numpy(labels["M_shadow"])[None, None].float()
    mt = torch.from_numpy(labels["M_tree"])[None, None].float()
    ms = F.interpolate(ms, scale_factor=factor, mode="nearest")[0, 0].numpy().astype(np.uint8)
    mt = F.interpolate(mt, scale_factor=factor, mode="nearest")[0, 0].numpy().astype(np.uint8)
    Image.fromarray(ms * 255).save(os.path.join(out_dir, name + "_mshadow.png"))
    Image.fromarray(mt * 255).save(os.path.join(out_dir, name + "_mtree.png"))
    boxes = [{**b, "bbox": (np.asarray(b["bbox"]) * factor).tolist()} for b in labels["boxes"]]
    return {"id": idx, "file": name + ".png", "maps": name + "_maps.npz",
            "mshadow": name + "_mshadow.png", "mtree": name + "_mtree.png",
            "size": [Ws, Hs], "base_size": list(base_size), "factor": factor, "extent": extent,
            "scene": {"az": sc.sun_azimuth, "el": sc.sun_elevation,
                      "canopies": [{"centre": c.centre.tolist(), "radius": c.radius}
                                   for c in sc.canopies]},
            "labels": {"boxes": boxes, "canopy_centres_world": labels["canopy_centres_world"]}}


class DiffusionBaseline:
    """Common fine-tuning / generation / persistence protocol for baselines."""
    name = "baseline"
    hint_channels = 0                      # native condition channels (subclass)

    def __init__(self, base_ch=32, T_total=200, device="cpu"):
        self.base_ch, self.T_total, self.device = base_ch, T_total, device
        self.net = ConditionalDenoiser(3 + self.hint_channels, base_ch).to(device)

    # ---- native conditioning (subclass) ----
    def build_hint(self, render_chw, maps, scene, extent) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, *a, **k):
        raise NotImplementedError(
            f"Load the official {cls.name} checkpoint here and keep its native "
            "conditioning; see the subclass docstring for the wiring.")

    def _hint_tensor(self, ds, i):
        render = ds[i]["image"].numpy()
        hint = self.build_hint(render, ds.maps[i], ds.scenes[i], ds.extent)
        return torch.from_numpy(np.concatenate([render, hint], 0))

    # ---- unified fine-tuning (same real set, same optimizer as PC-GDT) ----
    def finetune(self, ds_real, steps=250, batch=4, lr=1e-4, warmup=20, log=print):
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
        loader = itertools.cycle(DataLoader(ds_real, batch_size=min(batch, len(ds_real)), shuffle=True))
        ab = torch.cumprod(1.0 - _betas(self.T_total).to(self.device), 0)
        hist = []
        for step in range(steps):
            for g in opt.param_groups:
                g["lr"] = cosine_lr_with_warmup(step, steps, warmup, lr)
            b = next(loader)
            x0 = b["image"].to(self.device); B = x0.shape[0]
            hint = torch.stack([self._hint_tensor(ds_real, int(i)) for i in b["idx"]]).to(self.device)
            t = torch.randint(0, self.T_total, (B,), device=self.device)
            a = ab[t].view(B, 1, 1, 1)
            noise = torch.randn_like(x0)
            xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
            loss = F.mse_loss(self.net(xt, t, hint), noise)   # L_diff only, no physics penalty
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0); opt.step()
            if (step + 1) % max(steps // 5, 1) == 0:
                log(f"[{self.name}] finetune {step+1}/{steps} L={loss.item():.4f}")
                hist.append({"step": step + 1, "loss": loss.item()})
        return hist

    # ---- unified generation (same budget, sampler, persistence as PC-GDT) ----
    def generate(self, ds_scenes, out_dir, count, w=5.0, steps=30, seed=0, upscale=2, log=print):
        os.makedirs(out_dir, exist_ok=True)
        manifest = []
        n = len(ds_scenes)
        for k in range(count):
            i = k % n
            hint = torch.from_numpy(np.concatenate(
                [ds_scenes[i]["image"].numpy(),
                 self.build_hint(ds_scenes[i]["image"].numpy(), ds_scenes.maps[i],
                                 ds_scenes.scenes[i], ds_scenes.extent)], 0))[None].to(self.device)
            img = sample_image(self.net, hint, (1, 3, ds_scenes.H, ds_scenes.W),
                               guidance_w=w, steps=steps, T_total=self.T_total, seed=seed + k)
            img01 = ((img[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)
            if upscale != 1:
                t = torch.from_numpy(img01.transpose(2, 0, 1))[None]
                t = F.interpolate(t, scale_factor=upscale, mode="bicubic", align_corners=False)
                img01 = t[0].clamp(0, 1).numpy().transpose(1, 2, 0)
            entry = write_sample(out_dir, k, img01, ds_scenes.scenes[i], ds_scenes.maps[i],
                                 ds_scenes.extent, (ds_scenes.W, ds_scenes.H), upscale, seed + k)
            manifest.append(entry)
            if (k + 1) % max(count // 5, 1) == 0:
                log(f"[{self.name}] generated {k+1}/{count}")
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest