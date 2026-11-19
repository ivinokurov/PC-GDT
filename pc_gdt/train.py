"""Training: (0) pretrain the tiny VAE stand-in and freeze it (analogue of
SD's pretrained frozen KL-VAE), (1) train control branch + FiLM with
AdamW + cosine warmup and the penalised objective L_diff + lambda*R."""
from __future__ import annotations
import itertools, json, os
import torch
from torch.utils.data import DataLoader
from .model import PCControlNet, pc_gdt_loss, cosine_lr_with_warmup


def pretrain_vae(model, ds, steps=120, bs=4, lr=1e-3, device="cpu", log=print):
    for p in list(model.enc.parameters()) + list(model.dec.parameters()):
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr)
    loader = itertools.cycle(DataLoader(ds, batch_size=min(bs, len(ds)), shuffle=True))
    for step in range(steps):
        x = next(loader)["image"].to(device)
        rec = model.dec(model.enc(x))
        loss = torch.nn.functional.mse_loss(rec, x)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % max(steps // 4, 1) == 0:
            log(f"[vae] step {step+1}/{steps} recon={loss.item():.4f}")
    for p in list(model.enc.parameters()) + list(model.dec.parameters()):
        p.requires_grad_(False)
    with torch.no_grad():                       # latent scaling, as in SD
        x = torch.stack([ds[i]["image"] for i in range(min(8, len(ds)))]).to(device)
        std = model.enc(x).std().clamp(min=1e-3)
        model.latent_scale.fill_((1.0 / std).item())
    log(f"[vae] frozen; latent_scale={model.latent_scale.item():.3f}")


def train_pc_controlnet(cfg, model=None, ds_train=None, ds_val=None, log=print):
    device = cfg.device
    model = model or PCControlNet(base_ch=cfg.base_ch, cond_channels=9,
                                  use_film=cfg.use_film, use_controlnet=cfg.use_controlnet)
    model.T_total = cfg.T_total
    model.to(device)
    if cfg.vae_steps > 0:
        pretrain_vae(model, ds_train, cfg.vae_steps, cfg.batch, device=device, log=log)
    stats = model.trainability(freeze_backbone=True)
    log(f"[train] trainable {stats['trainable']}/{stats['total']} params")
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.lr,
                            betas=(0.9, 0.999), weight_decay=0.01)
    loader = itertools.cycle(DataLoader(ds_train, batch_size=min(cfg.batch, len(ds_train)),
                                        shuffle=True))
    history = []
    for step in range(cfg.steps):
        lr = cosine_lr_with_warmup(step, cfg.steps, cfg.warmup, cfg.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        b = next(loader)
        loss, parts = pc_gdt_loss(model, b["image"].to(device), b["cond"].to(device),
                                  b["sun"].to(device), None, lam=cfg.lam, T=cfg.T_total)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        opt.step()
        if (step + 1) % max(cfg.steps // 10, 1) == 0:
            log(f"[train] step {step+1}/{cfg.steps} L={loss.item():.4f} "
                f"diff={parts['L_diff']:.4f} pen={parts['L_penalty']:.4f} lr={lr:.2e}")
            history.append({"step": step + 1, "loss": loss.item(), **parts})
    os.makedirs(cfg.out, exist_ok=True)
    ckpt = os.path.join(cfg.out, "pc_controlnet.pt")
    torch.save({"state_dict": model.state_dict(), "config": vars(cfg)}, ckpt)
    with open(os.path.join(cfg.out, "train_log.json"), "w") as f:
        json.dump(history, f, indent=2)
    log(f"[train] saved {ckpt}")
    return model