"""Run the generative-baseline comparison (paper sec. 6.1).

For each method: fine-tune on the same real set -> generate the same budget
from the same UE5 scenes -> apply the SAME PGPF (filtering.filter_manifest) ->
export COCO for DE-ViT configuration B -> optional volume equalization
(subsample N accepted images per method, paper uses 8,000).

Detector mAP is measured in the external DE-ViT repository on the exported
COCO files; this script produces the datasets and the generation/PGPF report.

Usage:
    python -m pc_gdt.baselines.run_baselines --methods diffusionsat diffcr \
        --out runs/baselines --gen-count 18 --steps 250 --equalize 6
    python -m pc_gdt.baselines.run_baselines --include-pcgdt   # add PC-GDT reference
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from ..config import demo_config
from ..dataset import SceneDataset, make_splits
from ..filtering import filter_manifest
from ..pgpf import PGPFThresholds
from ..export import export_coco
from .diffusionsat import DiffusionSatAdapter
from .diffcr import DiffCRAdapter

ADAPTERS = {"diffusionsat": DiffusionSatAdapter, "diffcr": DiffCRAdapter}


def run_method(adapter, ds_real, ds_scenes, out_dir, cfg, gen_count, log=print):
    method_dir = os.path.join(out_dir, adapter.name)
    os.makedirs(method_dir, exist_ok=True)
    hist = adapter.finetune(ds_real, steps=cfg.steps, batch=cfg.batch, lr=cfg.lr,
                            warmup=cfg.warmup, log=log)
    gen_dir = os.path.join(method_dir, "generated")
    manifest = adapter.generate(ds_scenes, gen_dir, count=gen_count, w=cfg.w,
                                steps=cfg.gen_steps, seed=cfg.seed, upscale=cfg.upscale, log=log)
    thr = PGPFThresholds(cfg.delta_az, cfg.eta_iou, cfg.tau_ridge)
    accepted, report = filter_manifest(manifest, gen_dir, thr, log=log)   # SAME PGPF
    meas = [type("M", (), m)() for m in report["measurements"]]
    accepted_meas = [m for m, ok in zip(meas, report["pass_rates"].get("_mask", [])) ] if False else meas
    summary = {"method": adapter.name, "finetune_log": hist,
               "n_generated": len(manifest), "n_accepted": len(accepted),
               "pass_rates": report["pass_rates"],
               "geometry_accepted": {
                   "iou_mean": float(np.mean([m.shadow_iou for m in meas])) if meas else None,
                   "az_mean_deg": float(np.mean([m.az_dev_mean_deg for m in meas])) if meas else None},
               "tv_bound": report["reliability"]["tv_bound"]}
    export_coco(accepted, gen_dir, splits={"train": [e["id"] for e in accepted]})
    with open(os.path.join(method_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary, accepted, gen_dir


def equalize_volumes(accepted_by_method, out_dir, n, seed=0, log=print):
    """Subsample n accepted images per method (paper: 8,000) and re-export COCO."""
    rng = np.random.default_rng(seed)
    eq = {}
    for name, (accepted, gen_dir) in accepted_by_method.items():
        if len(accepted) < n:
            log(f"[equalize] {name}: only {len(accepted)} accepted < {n}; skipped")
            continue
        idx = rng.choice(len(accepted), n, replace=False)
        sub = [accepted[int(i)] for i in idx]
        eq_dir = os.path.join(out_dir, name, f"equalized_{n}")
        os.makedirs(eq_dir, exist_ok=True)
        export_coco(sub, gen_dir, splits={"train": [e["id"] for e in sub]})
        with open(os.path.join(eq_dir, "subset.json"), "w") as f:
            json.dump(sub, f, indent=2)
        eq[name] = n
        log(f"[equalize] {name}: {n} images -> {eq_dir}")
    return eq


def main():
    ap = argparse.ArgumentParser(prog="pc_gdt.baselines.run_baselines")
    ap.add_argument("--methods", nargs="+", default=["diffusionsat", "diffcr"],
                    choices=list(ADAPTERS))
    ap.add_argument("--out", default="runs/baselines")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--gen-count", type=int, default=18)
    ap.add_argument("--equalize", type=int, default=None)
    ap.add_argument("--include-pcgdt", action="store_true")
    a = ap.parse_args()
    cfg = demo_config(out=a.out, device=a.device, seed=a.seed, steps=a.steps)
    os.makedirs(cfg.out, exist_ok=True)
    tr, va, te = make_splits(cfg.n_train, cfg.n_val, cfg.n_test, cfg.seed)
    ds_real = SceneDataset(tr, cfg.H, cfg.W, cfg.extent, cfg.n_shadow_samples, False, cfg.seed)
    ds_scenes = SceneDataset(te, cfg.H, cfg.W, cfg.extent, cfg.n_shadow_samples, False, cfg.seed + 900)

    summaries, accepted_by_method = {}, {}
    if a.include_pcgdt:
        from ..train import train_pc_controlnet
        from ..generate import generate_dataset
        model = train_pc_controlnet(cfg, ds_train=ds_real)
        gen_dir = os.path.join(cfg.out, "pcgdt", "generated")
        manifest = generate_dataset(model, ds_scenes, list(range(len(ds_scenes))), cfg)
        thr = PGPFThresholds(cfg.delta_az, cfg.eta_iou, cfg.tau_ridge)
        accepted, report = filter_manifest(manifest, gen_dir, thr)
        summaries["pcgdt"] = {"method": "pcgdt", "n_generated": len(manifest),
                              "n_accepted": len(accepted), "pass_rates": report["pass_rates"]}
        export_coco(accepted, gen_dir, splits={"train": [e["id"] for e in accepted]})
        accepted_by_method["pcgdt"] = (accepted, gen_dir)

    for name in a.methods:
        adapter = ADAPTERS[name](base_ch=cfg.base_ch, T_total=cfg.T_total, device=cfg.device)
        s, accepted, gen_dir = run_method(adapter, ds_real, ds_scenes, cfg.out, cfg, a.gen_count)
        summaries[name] = s
        accepted_by_method[name] = (accepted, gen_dir)

    if a.equalize:
        summaries["equalized"] = equalize_volumes(accepted_by_method, cfg.out, a.equalize, cfg.seed)
    with open(os.path.join(cfg.out, "baselines_report.json"), "w") as f:
        json.dump(summaries, f, indent=2, default=float)
    print(f"[done] {os.path.join(cfg.out, 'baselines_report.json')}")


if __name__ == "__main__":
    main()