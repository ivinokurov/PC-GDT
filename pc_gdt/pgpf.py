"""PGPF: three sequential physical checks + Bayesian reliability analysis.

Check 1: |az(crown->shadow) - antisolar| <= 15 deg
Check 2: IoU(pred shadow, C4) >= 0.65
Check 3: mean normalised luminance gradient on ridge inside shadow >= 0.15
Azimuths are geographic (deg CW from North); centroids must be in WORLD
(east, north) coordinates -- use conditions.pixel_to_world for image masks.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def iou_mask(pred, truth) -> float:
    pred, truth = pred.astype(bool), truth.astype(bool)
    inter = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    return float(inter / union) if union else 1.0


def crown_shadow_azimuth_deg(crown_centre_xy, shadow_centroid_xy) -> float:
    d = np.asarray(shadow_centroid_xy, float) - np.asarray(crown_centre_xy, float)
    if abs(d[0]) < 1e-12 and abs(d[1]) < 1e-12:
        return float("nan")
    return float(np.degrees(np.arctan2(d[0], d[1])) % 360.0)


def _dilate(mask, k):
    """Zero-padded dilation (no torus wrap)."""
    mask = mask.astype(bool)
    pad = k
    sp = np.pad(mask, pad, mode="constant")
    out = np.zeros_like(sp)
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            out |= np.roll(np.roll(sp, dy, 0), dx, 1)
    return out[pad:pad + mask.shape[0], pad:pad + mask.shape[1]]


def normalized_ridge_gradient(lum, ridge_mask, shadow_mask) -> float:
    """Sobel magnitude normalised by the image luminance range, averaged on
    ridge pixels inside (or adjacent to) the shadow.  If the scene has no
    ridge pixels at all, returns NaN -> check 3 is vacuously passed."""
    lum = np.asarray(lum, float)
    rng_ = lum.max() - lum.min()
    ridge = ridge_mask.astype(bool)
    if not ridge.any():
        return float("nan")
    if rng_ < 1e-9:
        return 0.0
    gx = np.zeros_like(lum); gy = np.zeros_like(lum)
    gx[:, 1:-1] = (lum[:, 2:] - lum[:, :-2]) / 2
    gy[1:-1, :] = (lum[2:, :] - lum[:-2, :]) / 2
    mag = np.hypot(gx, gy) / rng_
    sel = ridge & _dilate(shadow_mask, 2)
    if not sel.any():
        return float("nan")          # shadow does not touch the ridge: vacuous pass
    return float(mag[sel].mean())


@dataclass
class PGPFThresholds:
    delta_az_deg: float = 15.0
    eta_iou: float = 0.65
    tau_ridge: float = 0.15


@dataclass
class SampleMeasurements:
    az_dev_deg: float
    shadow_iou: float
    ridge_grad: float


@dataclass
class PGPFResult:
    passed_check1: np.ndarray
    passed_check12: np.ndarray
    passed_all: np.ndarray
    rejection_stage: np.ndarray

    def pass_rates(self):
        return {"check1": float(self.passed_check1.mean()),
                "check1+2": float(self.passed_check12.mean()),
                "all_three": float(self.passed_all.mean()),
                "n": int(len(self.passed_check1))}


def run_pgpf(measurements, thr: PGPFThresholds = PGPFThresholds()) -> PGPFResult:
    if isinstance(measurements, list):
        m = np.asarray([[x.az_dev_deg, x.shadow_iou, x.ridge_grad]
                        for x in measurements], float)
    else:
        m = np.asarray(measurements, float)
    c1 = m[:, 0] <= thr.delta_az_deg
    c2 = c1 & (m[:, 1] >= thr.eta_iou)
    ridge_ok = np.isnan(m[:, 2]) | (m[:, 2] >= thr.tau_ridge)   # NaN = vacuous pass
    c3 = c2 & ridge_ok
    stage = np.where(c3, 0, np.where(c2, 3, np.where(c1, 2, 1)))
    return PGPFResult(c1, c2, c3, stage)


# ---------------- reliability ----------------
def posterior_defect_rate(q, pi, f) -> float:
    num = q * (1 - pi)
    den = num + (1 - q) * (1 - f)
    return num / den if den > 0 else 0.0


def posterior_interval(q_rng, pi_rng, f) -> tuple[float, float]:
    vals = [posterior_defect_rate(q, pi, f)
            for q in q_rng for pi in pi_rng]
    return min(vals), max(vals)


def expected_acceptance(q, pi, f) -> float:
    return q * (1 - pi) + (1 - q) * (1 - f)


def solve_total_false_rejection(observed_pass, q, pi) -> tuple[float, float]:
    """F_min/F_max at the CENTRAL q, over pi_full in [pi, 1]:
    acceptance = (1-q)(1-F) + q(1-pi_full).
    For accept=0.862, q=0.10, pi=0.85 -> [0.0422, 0.0589] (paper's interval)."""
    q_c = float(np.mean(q)) if isinstance(q, (tuple, list)) else float(q)
    pi_lo = min(pi) if isinstance(pi, (tuple, list)) else float(pi)
    f_min = max(0.0, 1.0 - observed_pass / (1 - q_c))                    # pi_full = 1
    f_max = max(0.0, 1.0 - (observed_pass - q_c * (1 - pi_lo)) / (1 - q_c))
    return min(f_min, f_max), max(f_min, f_max)


def total_variation_bound(f_total_hi, q_post_hi) -> float:
    return q_post_hi + f_total_hi


def estimate_reliability(expert_valid, rejected, n_boot=1000, seed=0):
    """q, pi, f from an expert-verified subsample (the 200-image protocol):
    expert_valid[i] = image physically correct; rejected[i] = filter decision.
    Returns point estimates + percentile bootstrap 95% CIs."""
    v = np.asarray(expert_valid, bool); r = np.asarray(rejected, bool)
    rng = np.random.default_rng(seed)

    def est(idx):
        vv, rr = v[idx], r[idx]
        q = float((~vv).mean())
        pi = float(rr[~vv].mean()) if (~vv).any() else 0.0
        f = float(rr[vv].mean()) if vv.any() else 0.0
        return q, pi, f

    point = est(np.arange(len(v)))
    boots = np.array([est(rng.integers(0, len(v), len(v))) for _ in range(n_boot)])
    ci = {k: (float(np.percentile(boots[:, i], 2.5)),
              float(np.percentile(boots[:, i], 97.5)))
          for i, k in enumerate(("q", "pi", "f"))}
    return dict(zip(("q", "pi", "f"), point)), ci


if __name__ == "__main__":
    print("q_post point:", round(posterior_defect_rate(.10, .85, .02), 4))
    print("q_post interval:", [round(x, 4) for x in
                               posterior_interval((.06, .15), (.75, .90), .02)])
    print("acceptance (IoU only):", round(expected_acceptance(.10, .85, .02), 4))
    lo, hi = solve_total_false_rejection(.862, .10, .85)
    print("F_sigma:", round(lo, 4), round(hi, 4),
          "| TV <=", round(total_variation_bound(hi, .0174), 4))