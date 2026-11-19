"""PGPF over generated images.  Measurements use a luminance/Otsu shadow
segmenter + condition-derived canopy centres (stand-in for the frozen
DE-ViT + shadow segmenter; swap in the real detector via `measure_fn`).
Azimuths are computed in WORLD coordinates (pixel y is south!)."""
from __future__ import annotations
import json, os
import numpy as np
from PIL import Image
from .conditions import pixel_to_world, IDX_CANOPY, IDX_FLAT, IDX_SLOPE, IDX_RIDGE
from .pgpf import (SampleMeasurements, PGPFThresholds, run_pgpf, iou_mask,
                   crown_shadow_azimuth_deg, normalized_ridge_gradient,
                   posterior_defect_rate, posterior_interval, expected_acceptance,
                   solve_total_false_rejection, total_variation_bound)
from .geometry import antisolar_azimuth, angular_difference_deg


def _otsu(vals):
    hist, edges = np.histogram(vals, bins=64, range=(0, 1))
    p = hist / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p); w1 = 1 - w0
    m0 = np.cumsum(p * centers) / np.maximum(w0, 1e-9)
    m1 = (np.sum(p * centers) - np.cumsum(p * centers)) / np.maximum(w1, 1e-9)
    var = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.argmax(var))])


def measure_image(img01_hwc, maps_npz, scene_meta, labels, extent, base_size):
    lum = img01_hwc.mean(-1)
    Hb, Wb = base_size[1], base_size[0]
    lum_b = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).resize((Wb, Hb))) / 255
    sem, c4, ridge = maps_npz["semantic"], maps_npz["shadow"], maps_npz["ridge"]
    roof = np.isin(sem, [IDX_FLAT, IDX_SLOPE, IDX_RIDGE])
    canopy = sem == IDX_CANOPY
    tau = _otsu(lum_b[roof]) if roof.any() else 0.4
    pred = (lum_b < tau) & roof & ~canopy
    iou = iou_mask(pred, c4)
    anti = float(antisolar_azimuth(scene_meta["az"]))
    devs = []
    centres = labels["canopy_centres_world"]
    for b in labels["boxes"]:
        if not b["cls"].startswith("shadow"):
            continue
        x0, y0, x1, y1 = [int(v / (img01_hwc.shape[1] / Wb)) for v in b["bbox"]]
        sub = pred[max(y0 - 2, 0):y1 + 3, max(x0 - 2, 0):x1 + 3]
        if sub.sum() < 3:
            continue
        rr, cc = np.nonzero(sub)
        px, py = cc.mean() + max(x0 - 2, 0), rr.mean() + max(y0 - 2, 0)
        wx, wy = pixel_to_world(px, py, extent, Hb, Wb)
        az = crown_shadow_azimuth_deg(centres[b["canopy_id"]], (wx, wy))
        if not np.isnan(az):
            devs.append(abs(float(angular_difference_deg(az, anti))))
    az_dev = float(np.mean(devs)) if devs else 180.0
    grad = normalized_ridge_gradient(lum_b, ridge, pred)
    return SampleMeasurements(az_dev, iou, grad), pred


def filter_manifest(manifest, out_dir, thr: PGPFThresholds = PGPFThresholds(),
                    priors=(0.10, 0.85, 0.02), log=print):
    measurements, entries = [], []
    for e in manifest:
        img = np.asarray(Image.open(os.path.join(out_dir, e["file"]))).astype(np.float32) / 255
        maps = np.load(os.path.join(out_dir, e["maps"]))
        m, _ = measure_image(img, maps, e["scene"], e["labels"], e["extent"], e["base_size"])
        measurements.append(m); entries.append(e)
    res = run_pgpf(measurements, thr)
    accepted = [e for e, ok in zip(entries, res.passed_all) if ok]
    q, pi, f = priors
    qp = posterior_defect_rate(q, pi, f)
    qlo, qhi = posterior_interval((0.06, 0.15), (0.75, 0.90), f)
    rates = res.pass_rates()
    flo, fhi = solve_total_false_rejection(rates["all_three"], q, pi)
    report = {"pass_rates": rates,
              "rejection_stages": {str(s): int((res.rejection_stage == s).sum())
                                   for s in (1, 2, 3)},
              "measurements": [{"az_dev": m.az_dev_deg, "iou": m.shadow_iou,
                                "ridge": m.ridge_grad} for m in measurements],
              "reliability": {"q": q, "pi": pi, "f": f,
                              "q_post_point": qp, "q_post_interval": [qlo, qhi],
                              "acceptance_model_iou_only": expected_acceptance(q, pi, f),
                              "f_sigma_interval": [flo, fhi],
                              "tv_bound": total_variation_bound(fhi, q * (1 - pi) / max(rates["all_three"], 1e-9))}}
    with open(os.path.join(out_dir, "pgpf_report.json"), "w") as fp:
        json.dump(report, fp, indent=2)
    with open(os.path.join(out_dir, "accepted.json"), "w") as fp:
        json.dump(accepted, fp, indent=2)
    log(f"[pgpf] {rates['n']} images: check1 {rates['check1']:.1%}, "
        f"+2 {rates['check1+2']:.1%}, all {rates['all_three']:.1%}; "
        f"q_post~{qp:.3f}, TV<={report['reliability']['tv_bound']:.3f}")
    return accepted, report