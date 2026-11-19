"""PC-ControlNet: SD-2.1-style latent diffusion + ControlNet branch + FiLM.

CPU-runnable stand-in with the EXACT interface and mathematics of the paper:
  * diffusion in latent space (eq. (5)): z0 = E(x0)*scale, zt = sqrt(ab)z0 + ...
  * frozen backbone/VAE; trainable = hint encoder + control blocks + zero-conv
    + fusion adapter + FiLM (paper's recipe);
  * FiLM per-channel gamma_l, beta_l in R^{C_l}, zero-init (identity start);
  * loss L_diff + lambda * R, penalty on the one-step x0 estimate, weighted
    by ab_t (SNR) because x0_hat is meaningless at high noise;
  * CFG on zeroed conditions (stand-in for the empty text prompt);
  * correct strided DDPM ancestral sampling.
`from_pretrained_sd21` documents the swap to real diffusers checkpoints.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .violations import ViolationMeasures, compute_violations


def sun_embedding_torch(az_deg, el_deg):
    az, el = torch.deg2rad(az_deg.float()), torch.deg2rad(el_deg.float())
    return torch.stack([torch.cos(az), torch.sin(az), torch.cos(el), torch.sin(el)], -1)


class FiLMSunModulator(nn.Module):
    """(gamma_l, beta_l) = MLP_l(s), gamma_l, beta_l in R^{C_l}."""
    def __init__(self, channels: list[int], hidden: int = 32):
        super().__init__()
        self.channels = list(channels)
        self.mlp = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.heads = nn.ModuleList([nn.Linear(hidden, 2 * c) for c in channels])
        for h in self.heads:
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)

    def params(self, e):
        h = self.mlp(e)
        out = []
        for head, c in zip(self.heads, self.channels):
            gb = head(h)
            out.append((1.0 + gb[:, :c], gb[:, c:]))
        return out

    @staticmethod
    def apply(h, gamma, beta):
        if gamma is None:
            return h
        return gamma[..., None, None] * h + beta[..., None, None]


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        return self.conv2(F.silu(self.conv1(x))) + self.skip(x)


class PCControlNet(nn.Module):
    def __init__(self, img_channels=3, latent_ch=4, base_ch=32,
                 cond_channels=9, use_film=True, use_controlnet=True):
        super().__init__()
        assert base_ch % 2 == 0, "base_ch must be even (sinusoidal time embedding)"
        self.img_channels, self.latent_ch, self.base_ch = img_channels, latent_ch, base_ch
        self.use_film, self.use_controlnet = use_film, use_controlnet
        self.T_total = 1000
        self.register_buffer("latent_scale", torch.tensor(1.0))

        self.time_mlp = nn.Sequential(nn.Linear(base_ch, base_ch * 2), nn.SiLU(),
                                      nn.Linear(base_ch * 2, base_ch))
        self.enc = nn.Sequential(nn.Conv2d(img_channels, latent_ch, 3, 2, 1), nn.SiLU(),
                                 nn.Conv2d(latent_ch, latent_ch, 3, 2, 1))
        self.dec = nn.Sequential(nn.ConvTranspose2d(latent_ch, max(latent_ch // 2, 2), 4, 2, 1),
                                 nn.SiLU(),
                                 nn.ConvTranspose2d(max(latent_ch // 2, 2), img_channels, 4, 2, 1),
                                 nn.Tanh())
        self.main_in = nn.Conv2d(latent_ch + base_ch, base_ch, 3, padding=1)
        self.main_b1, self.main_b2 = ResBlock(base_ch, base_ch), ResBlock(base_ch, base_ch)
        self.main_out = nn.Conv2d(base_ch, latent_ch, 3, padding=1)

        self.cond_in = nn.Sequential(nn.Conv2d(cond_channels, base_ch, 3, 2, 1), nn.SiLU(),
                                     nn.Conv2d(base_ch, base_ch, 3, 2, 1))
        self.ctl_b1, self.ctl_b2 = ResBlock(base_ch, base_ch), ResBlock(base_ch, base_ch)
        self.ctl_zero = nn.Conv2d(base_ch, base_ch, 1)
        nn.init.zeros_(self.ctl_zero.weight); nn.init.zeros_(self.ctl_zero.bias)
        self.film = FiLMSunModulator([base_ch] * 4) if use_film else None
        self.fusion = nn.Conv2d(base_ch * 2, base_ch, 1)
        for p in list(self.enc.parameters()) + list(self.dec.parameters()):
            p.requires_grad_(False)

    # ---- VAE analogues (frozen; pretrained by train.pretrain_vae) ----
    def encode(self, x):
        with torch.no_grad():
            return self.enc(x) * self.latent_scale

    def decode(self, z, grad: bool = False):
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            return self.dec(z / self.latent_scale)

    # ---- diffusion ----
    @staticmethod
    def betas(T, b0=1e-4, b1=0.02):
        return torch.linspace(b0, b1, T)

    def q_sample(self, z0, t_idx, noise):
        ab = torch.cumprod(1.0 - self.betas(self.T_total).to(z0.device), 0)[t_idx]
        ab = ab.view(-1, 1, 1, 1)
        return ab.sqrt() * z0 + (1 - ab).sqrt() * noise, ab

    @staticmethod
    def _sinusoidal_t(t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        a = t.float()[:, None] * freqs[None]
        return torch.cat([torch.cos(a), torch.sin(a)], -1)

    def forward(self, z_t, t, cond, sun_emb=None):
        """eps_theta(z_t, t, C, s) in LATENT space."""
        B = z_t.shape[0]
        temb = self.time_mlp(self._sinusoidal_t(t, self.base_ch))[..., None, None]
        h = self.main_in(torch.cat([z_t, temb.expand(B, -1, *z_t.shape[2:])], 1))
        h = self.main_b2(self.main_b1(h))
        fp = (self.film.params(sun_emb)
              if (self.use_film and self.film is not None and sun_emb is not None)
              else [(None, None)] * 4)
        if self.use_controlnet:
            c = FiLMSunModulator.apply(self.cond_in(cond), *fp[0])
            g = FiLMSunModulator.apply(self.ctl_b1(c), *fp[1])
            g2 = FiLMSunModulator.apply(self.ctl_b2(g), *fp[2])
            add = self.ctl_zero(g2)
            if add.shape[-2:] != h.shape[-2:]:
                add = F.interpolate(add, size=h.shape[-2:], mode="bilinear", align_corners=False)
            h = h + add
            feat = torch.cat([h, g2], 1)
        else:
            feat = torch.cat([h, torch.zeros_like(h)], 1)
        h = FiLMSunModulator.apply(self.fusion(feat), *fp[3])
        return self.main_out(h)

    # ---- training recipe ----
    def trainability(self, freeze_backbone=True):
        for m in (self.time_mlp, self.main_in, self.main_b1, self.main_b2, self.main_out):
            for p in m.parameters():
                p.requires_grad_(not freeze_backbone)
        for m in (self.enc, self.dec):
            for p in m.parameters():
                p.requires_grad_(False)
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"trainable": tr, "total": sum(p.numel() for p in self.parameters())}

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    @classmethod
    def from_pretrained_sd21(cls, *args, **kwargs):
        """Hook for the real system (diffusers):
        unet = UNet2DConditionModel.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder='unet')  # frozen
        vae  = AutoencoderKL... (frozen, scaling_factor=0.18215 -> self.latent_scale)
        controlnet = ControlNetModel...; replace conditioning_embedding conv_in
            with in_channels=9 (or 16 for one-hot);
        inject FiLMSunModulator into controlnet down-blocks (per-channel heads);
        train controlnet + FiLM only; loss / sampling identical to this module.
        """
        raise NotImplementedError("Load real SD 2.1 / ControlNet checkpoints here; "
                                  "see docstring for the wiring.")


def pc_gdt_loss(model, x0, cond, sun_emb, violation=None, lam=2.0, T=1000,
                generator=None, penalty=True, thr=None):
    """L = E||eps - eps_theta||^2 + lambda * R  (paper eq. (5) + penalty).
    If `violation` is None and penalty=True, R is computed differentiably on
    the one-step x0 estimate and weighted by ab_t (SNR)."""
    model.T_total = T
    z0 = model.encode(x0)
    B = z0.shape[0]
    t = torch.randint(0, T, (B,), generator=generator, device=z0.device)
    noise = torch.randn(z0.shape, generator=generator, device=z0.device)
    zt, ab = model.q_sample(z0, t, noise)
    eps_hat = model(zt, t, cond, sun_emb)
    l_diff = F.mse_loss(eps_hat, noise, reduction="none").flatten(1).mean(1)   # (B,)
    l_pen = torch.zeros(B, device=z0.device)
    if lam > 0 and penalty:
        if violation is None:
            z0_hat = (zt - (1 - ab).sqrt() * eps_hat) / ab.sqrt()
            x0_hat = model.decode(z0_hat, grad=True)
            violation = compute_violations(x0_hat, cond, sun_emb, thr)
        l_pen = lam * violation.total().to(z0.device) * ab.view(B).detach()
    return l_diff.mean() + l_pen.mean(), {"L_diff": l_diff.mean().item(),
                                          "L_penalty": l_pen.mean().item()}


def cosine_lr_with_warmup(step, total_steps=150_000, warmup=1000, base_lr=1e-4):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    prog = (step - warmup) / max(total_steps - warmup, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


@torch.no_grad()
def sample(model, cond, sun_emb, shape, guidance_w=5.0, steps=30,
           T_total=None, seed=0, clip_x0=4.0):
    """Strided DDPM ancestral sampling with CFG (uncond = zeroed conditions,
    the stand-in for the empty text prompt).  `shape` is the IMAGE shape
    (B,3,H,W); the loop runs in latent space and decodes once at the end."""
    dev = next(model.parameters()).device
    g = torch.Generator(device=dev).manual_seed(seed)
    T = T_total or model.T_total
    steps = min(steps, T)
    ab = torch.cumprod(1.0 - model.betas(T).to(dev), 0)
    B, C, H, W = shape
    z = torch.randn((B, model.latent_ch, H // 4, W // 4), generator=g, device=dev)
    null = torch.zeros_like(cond)
    ts = torch.linspace(T - 1, 0, steps).long().unique_consecutive()
    for i, ti in enumerate(ts):
        t = int(ti)
        tp = int(ts[i + 1]) if i + 1 < len(ts) else -1
        ab_t = ab[t]
        ab_p = ab[tp] if tp >= 0 else torch.tensor(1.0, device=dev)
        tt = torch.full((B,), t, device=dev)
        ec = model(z, tt, cond, sun_emb)
        eu = model(z, tt, null, sun_emb)
        eps = eu + guidance_w * (ec - eu)
        z0 = (z - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        if clip_x0:
            z0 = z0.clamp(-clip_x0, clip_x0)
        if tp >= 0:
            var = (1 - ab_p) / (1 - ab_t) * (1 - ab_t / ab_p)
            z = (ab_p.sqrt() * z0
                 + (1 - ab_p - var).clamp(min=0).sqrt() * eps
                 + var.sqrt() * torch.randn(z.shape, generator=g, device=dev))
        else:
            z = z0
    return model.decode(z)