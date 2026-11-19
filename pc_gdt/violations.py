"""Differentiable physical-violation measures R = R_az + R_iou + R_ridge
(penalty form of the paper).  Computed on the one-step denoised estimate
x0_hat; the frozen DE-ViT segmenter is replaced by a smooth luminance proxy
so that gradients reach the control branch.  Hinges use the SAME thresholds
as PGPF (Definition 1).  Azimuth uses the chord surrogate
beta ~= sqrt(2(1-cos beta)) (smooth, exact to O(beta^3)).
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from .pgpf import PGPFThresholds


@dataclass
class ViolationMeasures:
    azimuth_dev_deg: torch.Tensor       # (B,) hinge: max(0, dev - delta_az)
    iou_deficit: torch.Tensor           # (B,) max(0, eta - softIoU)
    ridge_grad_shortfall: torch.Tensor  # (B,) max(0, tau - ridge gradient)

    def total(self) -> torch.Tensor:
        return self.azimuth_dev_deg / 180.0 + self.iou_deficit + self.ridge_grad_shortfall


def _sobel(lum):
    kx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      device=lum.device).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(lum, kx, padding=1)
    gy = F.conv2d(lum, kx.transpose(-1, -2), padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)


def compute_violations(x_hat: torch.Tensor, cond: torch.Tensor,
                       sun_emb: torch.Tensor,
                       thr: PGPFThresholds | None = None) -> ViolationMeasures:
    """x_hat: (B,3,H,W) in [-1,1]; cond: 9-channel tensor; sun_emb: (B,4)."""
    thr = thr or PGPFThresholds()
    B, _, H, W = x_hat.shape
    dev = x_hat.device
    lum = ((x_hat + 1) / 2).mean(1, keepdim=True)                 # (B,1,H,W)
    sem = (cond[:, 0:1] * 7).round()
    canopy = (sem == 6).float()
    roof = ((sem == 3) | (sem == 4) | (sem == 5)).float()
    c4, ridge = cond[:, 5:6], cond[:, 6:7]

    p = torch.sigmoid((0.45 - lum) / 0.05) * roof                 # soft shadow
    inter = (p * c4).sum((1, 2, 3))
    union = (p + c4 - p * c4).sum((1, 2, 3)) + 1e-6
    iou_def = torch.clamp(thr.eta_iou - inter / union, min=0)

    rows = torch.arange(H, device=dev).float().view(1, 1, H, 1)
    cols = torch.arange(W, device=dev).float().view(1, 1, 1, W)
    ps, cs = p.sum((1, 2, 3)) + 1e-6, canopy.sum((1, 2, 3)) + 1e-6
    sr = (p * rows).sum((1, 2, 3)) / ps; sc = (p * cols).sum((1, 2, 3)) / ps
    cr = (canopy * rows).sum((1, 2, 3)) / cs; cc = (canopy * cols).sum((1, 2, 3)) / cs
    dir_e, dir_n = sc - cc, -(sr - cr)                            # east, north
    az = torch.atan2(sun_emb[:, 1], sun_emb[:, 0])
    anti_e, anti_n = -torch.sin(az), -torch.cos(az)
    cosb = (dir_e * anti_e + dir_n * anti_n) / (
        torch.sqrt(dir_e ** 2 + dir_n ** 2) + 1e-6)
    beta = torch.sqrt(torch.clamp(2 * (1 - cosb), min=0))         # rad, surrogate
    az_hinge = torch.clamp(torch.rad2deg(beta) - thr.delta_az_deg, min=0)

    mag = _sobel(lum) / (lum.amax((1, 2, 3), keepdim=True)
                         - lum.amin((1, 2, 3), keepdim=True) + 1e-6)
    sel = ridge * F.max_pool2d(c4, 5, stride=1, padding=2)
    sel_sum = sel.sum((1, 2, 3))
    g = (mag * sel).sum((1, 2, 3)) / (sel_sum + 1e-6)
    ridge_short = torch.where(sel_sum > 0,
                              torch.clamp(thr.tau_ridge - g, min=0),
                              torch.zeros(B, device=dev))
    return ViolationMeasures(az_hinge, iou_def, ridge_short)