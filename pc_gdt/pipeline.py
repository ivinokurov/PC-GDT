"""End-to-end CLI.  Quick start:  python -m pc_gdt.pipeline demo --out runs/demo"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from PIL import Image
from .config import demo_config, paper_config
from .dataset import SceneDataset, make_splits
from .train import train_pc_controlnet
from .generate import generate_dataset
from .filtering import filter_manifest
from .pgpf import PGPFThresholds
from .metrics import geometry_summary, fid_kid, bootstrap_ci
from .export import export_coco


def run_demo(cfg, log=print):
    os.makedirs(cfg.out, exist_ok=True)
    tr, va, te = make_splits(cfg.n_train, cfg.n_val, cfg.n_test, cfg.seed)
    ds_tr = SceneDataset(tr, cfg.H, cfg.W, cfg.extent, cfg.n_shadow_samples,
                         cfg.pseudo_noise, cfg.seed)
    ds_va = SceneDataset(va, cfg.H, cfg.W, cfg.extent, cfg.n_shadow_samples, False, cfg.seed + 500)
    ds_te = SceneDataset(te, cfg.H, cfg.W, cfg.extent, cfg.n_shadow_samples, False, cfg.seed + 900)
    model = train_pc_controlnet(cfg, ds_train=ds_tr, ds_val=ds_va, log=log)

    gen_dir = os.path.join(cfg.out, "generated")
    manifest = generate_dataset(model, ds_te, list(range(len(ds_te))), cfg, log=log)
    thr = PGPFThresholds(cfg.delta_az, cfg.eta_iou, cfg.tau_ridge)
    accepted, pgpf_report = filter_manifest(manifest, gen_dir, thr, log=log)

    meas = [type("M", (), m)() for m in pgpf_report["measurements"]]
    geo = geometry_summary(meas)
    real = np.stack([((ds_va[i]["image"].numpy().transpose(1, 2, 0) + 1) / 2)
                     for i in range(len(ds_va))])
    fake = np.stack([np.asarray(Image.open(os.path.join(gen_dir, e["file"]))
                                .resize((cfg.W, cfg.H))).astype(np.float32) / 255
                     for e in manifest])
    dist = fid_kid(real, fake, use_inception=False)
    ious = [m.shadow_iou for m in meas]
    report = {"config": vars(cfg), "geometry": geo,
              "iou_ci95": bootstrap_ci(ious) if ious else None,
              "distribution": dist, "pgpf": pgpf_report,
              "n_generated": len(manifest), "n_accepted": len(accepted)}
    coco = export_coco(accepted, gen_dir,
                       splits={"train": [e["id"] for e in accepted[:max(len(accepted) - 2, 1)]],
                               "val": [e["id"] for e in accepted[max(len(accepted) - 2, 1):]]})
    report["coco_files"] = coco
    with open(os.path.join(cfg.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    log(f"[done] report: {os.path.join(cfg.out, 'report.json')}")
    log(f"       IoU={geo['iou_mean']:.3f}, az={geo['az_mean_deg']:.1f} deg, "
        f"accepted {len(accepted)}/{len(manifest)}, FID*={dist['FID']:.2f}, KID*={dist['KID']:.4f}"
        f"  (*feature stand-in, not paper-comparable)")
    return report


def main():
    ap = argparse.ArgumentParser(prog="pc_gdt.pipeline")
    ap.add_argument("cmd", choices=["demo", "paper"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--pseudo-noise", action="store_true")
    a = ap.parse_args()
    cfg = demo_config() if a.cmd == "demo" else paper_config()
    for k in ("out", "device", "seed", "steps"):
        if getattr(a, k) is not None:
            setattr(cfg, k, getattr(a, k))
    if a.pseudo_noise:
        cfg.pseudo_noise = True
    if a.cmd == "paper":
        raise SystemExit("Paper scale requires GPU + real SD2.1/ControlNet/VAE checkpoints "
                         "and real imagery; see model.from_pretrained_sd21 and README.")
    run_demo(cfg)


if __name__ == "__main__":
    main()