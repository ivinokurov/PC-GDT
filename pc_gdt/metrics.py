"""Geometry metrics + distribution metrics (FID/KID) + bootstrap CIs.

FID/KID use InceptionV3 pool3 features if torchvision weights are available
(use_inception=True); otherwise a documented lightweight feature stand-in
whose numbers are NOT comparable to the paper's FID/KID.
"""
from __future__ import annotations
import warnings
import numpy as np


def geometry_summary(measurements):
    az = [m.az_dev_deg for m in measurements]
    iou = [m.shadow_iou for m in measurements]
    return {"az_mean_deg": float(np.mean(az)), "az_std_deg": float(np.std(az)),
            "iou_mean": float(np.mean(iou))}


def _features(imgs01, use_inception=False):
    """imgs01: (N,H,W,3) in [0,1]."""
    if use_inception:
        import torch, torchvision
        net = torchvision.models.inception_v3(
            weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True).eval()
        net.fc = torch.nn.Identity()
        import torch.nn.functional as F
        t = torch.from_numpy(imgs01.transpose(0, 3, 1, 2)).float()
        t = F.interpolate(t, size=(299, 299), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        with torch.no_grad():
            return net((t - mean) / std).numpy()
    warnings.warn("FID/KID feature stand-in (no Inception weights): "
                  "values are NOT comparable to the paper.")
    feats = []
    for im in imgs01:
        small = im[::4, ::4]
        f = [small.mean((0, 1)), small.std((0, 1))]
        for c in range(3):
            f.append(np.histogram(small[..., c], bins=8, range=(0, 1))[0] / small[..., c].size)
        gy, gx = np.gradient(small.mean(-1))
        f.append(np.array([np.abs(gx).mean(), np.abs(gy).mean()]))
        feats.append(np.concatenate(f))
    return np.stack(feats)


def _fid(mu1, s1, mu2, s2):
    diff = mu1 - mu2
    a = np.linalg.eigvalsh(s1)
    s1h = (np.linalg.eigh(s1)[1] * np.sqrt(np.clip(a, 0, None))) @ np.linalg.eigh(s1)[1].T
    tr = np.sqrt(np.clip(np.linalg.eigvalsh(s1h @ s2 @ s1h), 0, None)).sum()
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * tr)


def _kid(x, y, subsets=20, seed=0):
    rng = np.random.default_rng(seed)
    n = min(len(x), len(y), 100)
    d = x.shape[1]

    def k(a, b):
        return ((a @ b.T) / d + 1) ** 3

    vals = []
    for _ in range(subsets):
        a = x[rng.choice(len(x), n, replace=False)]
        b = y[rng.choice(len(y), n, replace=False)]
        ka, kb, kab = k(a, a), k(b, b), k(a, b)
        np.fill_diagonal(ka, 0); np.fill_diagonal(kb, 0)
        vals.append(ka.sum() / (n * (n - 1)) + kb.sum() / (n * (n - 1)) - 2 * kab.mean())
    return float(np.mean(vals))


def fid_kid(real01, fake01, use_inception=False):
    fr, ff = _features(np.asarray(real01), use_inception), _features(np.asarray(fake01), use_inception)
    return {"FID": _fid(fr.mean(0), np.cov(fr.T), ff.mean(0), np.cov(ff.T)),
            "KID": _kid(fr, ff), "inception": use_inception}


def bootstrap_ci(values, n_boot=1000, seed=0, alpha=0.05):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def paired_bootstrap_diff(a, b, n_boot=1000, seed=0):
    """Paired-by-scene CI of mean(b)-mean(a) (paper's protocol for B-A, D-A)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))