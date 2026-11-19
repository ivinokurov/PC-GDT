"""PGPF over generated images (paper, sec. 5 and Definition 1).

Measurements of the shadow mask and of the "crown -> shadow" direction are
performed by the FROZEN DE-ViT detector together with a shadow segmenter.
Pass `detector=` to use them; the Otsu luminance segmenter with canopy centres
inherited from the physical render is only a lightweight fallback that keeps
the pipeline runnable without the detector checkpoint.

Check 3 follows Definition 1 literally: the normalised luminance gradient is
averaged over ridge pixels INSIDE the predicted shadow (normalisation by the
image luminance range).  Widening the selection to the shadow neighbourhood
is an explicit opt-in (`strict_ridge=False`).

Geometry is reported at the native generation resolution AND after the x2
super-resolution, mirroring Table `quality` (IoU 0.87 -> 0.86, azimuth
4.2 deg -> 4.4 deg).  The PGPF decision uses the native-resolution values.

Reliability protocol (sec. 5, sec. 5.1): q, pi, f are estimated on 200
expert-verified GENERATED images, disjoint from the 150 images used to
calibrate tau_ridge; the false-rejection audit uses 200 REJECTED images with
at least 50 per check, labelled blindly by two experts.
"""
from __future__ import annotations
import json, os
from typing import Callable, Protocol
import numpy as np
from PIL import Image
from .conditions import pixel_to_world, IDX_CANOPY, IDX_FLAT, IDX_SLOPE, IDX_RIDGE
from .pgpf import (SampleMeasurements, PGPFThresholds, run_pgpf, iou_mask,
                   crown_shadow_azimuth_deg, normalized_ridge_gradient,
                   posterior_defect_rate, posterior_interval, expected_acceptance,
                   solve_total_false_rejection, total_variation_bound)
from .geometry import antisolar_azimuth, angular_difference_deg
from .metrics import bootstrap_ci


class ReliabilityProtocol:
    """Subsample sizes and prior estimates reported in sec. 5 / sec. 5.1."""
    N_RELIABILITY = 200      # generated images, expert-verified -> q, pi, f
    N_TAU_CALIB = 150        # images for tau_ridge, disjoint from the above
    N_AUDIT = 200            # rejected images audited blindly by two experts
    N_AUDIT_PER_STAGE = 50   # minimum per check
    Q_CI = (0.06, 0.15)      # 95% CI of the prior defect probability
    PI_CI = (0.75, 0.90)     # 95% CI of the IoU-check power
    PRIORS = (0.10, 0.85, 0.02)   # q, pi, f (point estimates)


class FrozenDetector(Protocol):
    """Interface of the frozen DE-ViT + shadow segmenter used by PGPF.

    Input : img01_hwc -- float32 (H, W, 3) in [0, 1], super-resolved image.
    Output: (canopy_centres_world, shadow_mask), where the centres are (x, y)
            in world (east, north) metres and the mask is a bool array of the
            image size.  Returning None for the centres falls back to the
            annotation inherited from the physical render.
    """
    def __call__(self, img01_hwc: np.ndarray) -> tuple[list | None, np.ndarray]: ...


def _otsu(vals):
    hist, edges = np.histogram(vals, bins=64, range=(0, 1))
    p = hist / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p); w1 = 1 - w0
    m0 = np.cumsum(p * centers) / np.maximum(w0, 1e-9)
    m1 = (np.sum(p * centers) - np.cumsum(p * centers)) / np.maximum(w1, 1e-9)
    var = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.argmax(var))])


def _resize_bool(mask, size_wh):
    im = Image.fromarray(mask.astype(np.uint8) * 255).resize(size_wh, Image.NEAREST)
    return np.asarray(im) > 127


def _resize_lum(lum, size_wh):
    im = Image.fromarray((lum * 255).astype(np.uint8)).resize(size_wh, Image.BILINEAR)
    return np.asarray(im).astype(np.float32) / 255


def _segment_fallback(lum, sem):
    """Otsu stand-in for the frozen shadow segmenter (roof pixels only)."""
    roof = np.isin(sem, [IDX_FLAT, IDX_SLOPE, IDX_RIDGE])
    tau = _otsu(lum[roof]) if roof.any() else 0.4
    return (lum < tau) & roof & (sem != IDX_CANOPY)


def _azimuth_stats(pred, boxes, centres, extent, size_wh, anti):
    """Deviations of the measured "crown -> shadow" azimuth from the
    anti-solar azimuth (theta_sun + pi) mod 2pi, one value per pair."""
    W, H = size_wh
    devs = []
    for b in boxes:
        if not b["cls"].startswith("shadow") or b.get("canopy_id", -1) >= len(centres):
            continue
        x0, y0, x1, y1 = b["bbox"]
        r0, c0 = max(y0 - 2, 0), max(x0 - 2, 0)
        sub = pred[r0:y1 + 3, c0:x1 + 3]
        if sub.sum() < 3:
            continue
        rr, cc = np.nonzero(sub)
        wx, wy = pixel_to_world(cc.mean() + c0, rr.mean() + r0, extent, H, W)
        az = crown_shadow_azimuth_deg(centres[b["canopy_id"]], (wx, wy))
        if not np.isnan(az):
            devs.append(abs(float(angular_difference_deg(az, anti))))
    return devs


def measure_image(img01_hwc, maps_npz, scene_meta, labels, extent, base_size,
                  detector: Callable | None = None, strict_ridge: bool = True):
    """Measurements of one generated image for the three PGPF checks.

    Returns (SampleMeasurements, predicted shadow mask at base resolution).
    The decision values are computed at the native generation resolution;
    `az_dev_sr_deg` and `shadow_iou_sr` are the same quantities measured after
    the x2 super-resolution (reported, not used by the filter).
    """
    Hb, Wb = int(base_size[1]), int(base_size[0])
    Hs, Ws = img01_hwc.shape[:2]
    sem, c4, ridge = maps_npz["semantic"], maps_npz["shadow"], maps_npz["ridge"]
    lum_sr = img01_hwc.mean(-1)
    lum_b = _resize_lum(lum_sr, (Wb, Hb))

    centres = labels["canopy_centres_world"]
    if detector is not None:                       # paper's measurement path
        det_centres, det_mask = detector(img01_hwc)
        if det_centres:
            centres = det_centres
        pred_sr = np.asarray(det_mask, dtype=bool)
        pred_b = _resize_bool(pred_sr, (Wb, Hb))
    else:                                          # documented fallback
        pred_b = _segment_fallback(lum_b, sem)
        pred_sr = _segment_fallback(lum_sr, _resize_bool(sem, (Ws, Hs)))

    anti = float(antisolar_azimuth(scene_meta["az"]))
    boxes_b = [{**b, "bbox": [int(v * Wb / Ws) for v in b["bbox"]]}
               for b in labels["boxes"]]
    devs_b = _azimuth_stats(pred_b, boxes_b, centres, extent, (Wb, Hb), anti)
    devs_sr = _azimuth_stats(pred_sr, labels["boxes"], centres, extent, (Ws, Hs), anti)

    if strict_ridge:                               # Definition 1, item 3
        sel = ridge.astype(bool) & pred_b
        grad = (normalized_ridge_gradient(lum_b, sel.astype(np.uint8), pred_b)
                if sel.any() else float("nan"))    # no ridge in shadow: vacuous pass
    else:
        grad = normalized_ridge_gradient(lum_b, ridge, pred_b)

    m = SampleMeasurements(max(devs_b) if devs_b else 180.0,
                           iou_mask(pred_b, c4), grad)
    m.az_dev_mean_deg = float(np.mean(devs_b)) if devs_b else 180.0
    m.az_dev_sr_deg = max(devs_sr) if devs_sr else 180.0
    m.shadow_iou_sr = iou_mask(pred_sr, _resize_bool(c4, (Ws, Hs)))
    return m, pred_b


def filter_manifest(manifest, out_dir, thr: PGPFThresholds = PGPFThresholds(),
                    priors=ReliabilityProtocol.PRIORS, detector: Callable | None = None,
                    strict_ridge: bool = True, audit: dict | None = None, log=print):
    """Run the three sequential checks and write pgpf_report.json / accepted.json.

    `audit` optionally carries the independent expert audit of REJECTED images
    (sec. 5.1): {"expert_valid": [...], "stage": [...], "expert2_valid": [...]}.
    """
    measurements, entries = [], []
    for e in manifest:
        img = np.asarray(Image.open(os.path.join(out_dir, e["file"]))).astype(np.float32) / 255
        maps = np.load(os.path.join(out_dir, e["maps"]))
        m, _ = measure_image(img, maps, e["scene"], e["labels"], e["extent"],
                             e["base_size"], detector=detector, strict_ridge=strict_ridge)
        measurements.append(m); entries.append(e)

    res = run_pgpf(measurements, thr)
    accepted = [e for e, ok in zip(entries, res.passed_all) if ok]
    q, pi, f = priors
    rates = res.pass_rates()
    accept = max(rates["all_three"], 1e-9)
    qp = posterior_defect_rate(q, pi, f)
    qlo, qhi = posterior_interval(ReliabilityProtocol.Q_CI, ReliabilityProtocol.PI_CI, f)
    qp_full_hi = q * (1 - pi) / accept          # sec. 5: q_post <= q(1-pi)/0.862
    flo, fhi = solve_total_false_rejection(accept, q, pi)

    report = {
        "thresholds": {"delta_az_deg": thr.delta_az_deg, "eta_iou": thr.eta_iou,
                       "tau_ridge": thr.tau_ridge},
        "measurement_backend": "frozen_de-vit" if detector is not None else "otsu_fallback",
        "ridge_selection": "inside_shadow" if strict_ridge else "shadow_neighbourhood",
        "pass_rates": rates,
        "rejection_stages": {str(s): int((res.rejection_stage == s).sum()) for s in (1, 2, 3)},
        "measurements": [{"az_dev": m.az_dev_deg, "az_dev_mean": m.az_dev_mean_deg,
                          "az_dev_sr": m.az_dev_sr_deg, "iou": m.shadow_iou,
                          "iou_sr": m.shadow_iou_sr, "ridge": m.ridge_grad}
                         for m in measurements],
        "geometry": {"iou_mean": float(np.mean([m.shadow_iou for m in measurements])),
                     "iou_sr_mean": float(np.mean([m.shadow_iou_sr for m in measurements])),
                     "az_mean_deg": float(np.mean([m.az_dev_mean_deg for m in measurements])),
                     "az_sr_mean_deg": float(np.mean([m.az_dev_sr_deg for m in measurements]))},
        "reliability": {
            "protocol": {"n_reliability": ReliabilityProtocol.N_RELIABILITY,
                         "n_tau_calibration": ReliabilityProtocol.N_TAU_CALIB,
                         "disjoint_subsamples": True,
                         "n_audit": ReliabilityProtocol.N_AUDIT,
                         "n_audit_per_stage": ReliabilityProtocol.N_AUDIT_PER_STAGE},
            "q": q, "pi": pi, "f": f,
            "q_post_point": qp, "q_post_interval": [qlo, qhi],
            "q_post_full_upper": qp_full_hi,
            "acceptance_model_iou_only": expected_acceptance(q, pi, f),
            "tv_bound_iou_only": total_variation_bound(f, qp),
            "f_sigma_interval": [flo, fhi],
            "tv_bound": total_variation_bound(fhi, qp_full_hi)},
    }
    if audit:
        report["audit"] = audit_false_rejections(audit["expert_valid"], audit["stage"],
                                                 audit.get("expert2_valid"))
    with open(os.path.join(out_dir, "pgpf_report.json"), "w") as fp:
        json.dump(report, fp, indent=2, default=float)
    with open(os.path.join(out_dir, "accepted.json"), "w") as fp:
        json.dump(accepted, fp, indent=2)
    log(f"[pgpf] {rates['n']} images: check1 {rates['check1']:.1%}, "
        f"+2 {rates['check1+2']:.1%}, all {rates['all_three']:.1%}; "
        f"q_post<={qp_full_hi:.3f}, TV<={report['reliability']['tv_bound']:.3f}")
    return accepted, report


def audit_false_rejections(expert_valid, stage, expert2_valid=None, n_boot=1000, seed=0):
    """Independent blind audit of rejected images (sec. 5.1).

    expert_valid[i] -- expert verdict "physically correct" (i.e. a FALSE
    rejection by the filter); stage[i] in {1, 2, 3} -- the failing check.
    Returns the false-rejection rate with a 95% bootstrap CI, the per-check
    breakdown and the protocol-compliance flags.
    """
    v = np.asarray(expert_valid, bool)
    s = np.asarray(stage, int)
    per_stage = {str(k): {"n": int((s == k).sum()), "false": int(v[s == k].sum())}
                 for k in (1, 2, 3)}
    out = {"n": int(len(v)), "false_rejections": int(v.sum()),
           "false_rejection_rate": float(v.mean()) if len(v) else 0.0,
           "ci95": list(bootstrap_ci(v.astype(float), n_boot=n_boot, seed=seed)) if len(v) else None,
           "per_stage": per_stage,
           "protocol_ok": bool(len(v) >= ReliabilityProtocol.N_AUDIT and
                               all((s == k).sum() >= ReliabilityProtocol.N_AUDIT_PER_STAGE
                                   for k in (1, 2, 3)))}
    if expert2_valid is not None:
        out["cohen_kappa"] = cohen_kappa(v, expert2_valid)
    return out


def cohen_kappa(a, b) -> float:
    """Inter-expert agreement (sec. 5.1 reports kappa = 0.81)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if abs(1 - pe) > 1e-12 else 1.0