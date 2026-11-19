# PC-GDT: Physics-Conditioned Generative Digital Twin

[![Release](https://img.shields.io/badge/release-v1.0.0-blue)](#quick-start)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](#license)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#quick-start)

Synthesis of photorealistic aerial images of low-rise residential areas with **physically consistent tree-crown shadows** on pitched roofs, plus automatic annotation for training the [DE-ViT](https://github.com/ivinokurov/DE-ViT) shadow detector.

Pipeline: deterministic UE5 (+Colosseum) render → six condition maps + sun embedding → latent diffusion (SD 2.1 + ControlNet + FiLM) → Real-ESRGAN ×2 → **PGPF** physical filtering → image + inherited labels $(\hat{I}, \mathcal{A})$.

This repository is a **CPU-runnable reference implementation**: exact interfaces and mathematics of the paper, with lightweight stand-ins for UE5, SD 2.1, DE-ViT and Real-ESRGAN. Production checkpoints plug in through documented hooks. Demo metrics are **not** comparable to the paper's tables.

## Quick start

```bash
pip install -r requirements.txt                  # numpy, torch, pillow
python -m pc_gdt.pipeline demo --out runs/demo   # end-to-end, ~1–3 min on CPU
python -m pc_gdt.tests.test_all                  # 10 regression tests
```

Optional extras: `torchvision` (Inception FID/KID), `diffusers` (real SD 2.1), `realesrgan basicsr` (×2 SR), `lpips`, `pytest`.

## Project structure

```
pc_gdt/
├── __init__.py     # Public API: geometry, conditions, model, PGPF re-exports
├── geometry.py     # Roof faces, canopies, sun vector, ray-plane shadow projection
├── conditions.py   # Six maps C1..C6, 9/16-channel encodings, pseudo-condition noise
├── violations.py   # Differentiable penalties R_az + R_iou + R_ridge
├── model.py        # PC-ControlNet: latent diffusion + ControlNet + FiLM, CFG sampling
├── pgpf.py         # Three-stage physical filter + Bayesian reliability
├── render.py       # Procedural target render (UE5/photo stand-in) + labels A
├── dataset.py      # Scene-disjoint datasets and splits
├── train.py        # VAE pretraining, AdamW + cosine warmup, checkpoints
├── generate.py     # Sampling, ×2 upscaling, label transfer, shared write_sample()
├── filtering.py    # Measurements from images, PGPF run, reliability report
├── metrics.py      # IoU / azimuth, FID / KID, bootstrap CIs
├── export.py       # COCO export for DE-ViT
├── config.py       # demo / paper presets, DE-ViT hyperparameters
├── pipeline.py     # CLI: python -m pc_gdt.pipeline demo|paper
├── baselines/      # Generative baselines for remote sensing (paper sec. 6.1)
│   ├── __init__.py
│   ├── base.py             # Shared protocol: denoiser, DDPM+CFG sampler, fine-tuning
│   ├── diffusionsat.py     # DiffusionSat: metadata-only conditioning
│   ├── diffcr.py           # DiffCR: C4 fed as the cloud mask + optical render
│   └── run_baselines.py    # CLI: finetune -> generate -> PGPF -> COCO -> equalization
└── tests/
    ├── __init__.py
    ├── test_all.py         # Regression tests: geometry, conditions, model, PGPF, overfit
    └── test_baselines.py   # Baseline hints, no physical-condition leak, PGPF end-to-end

.gitignore          # Excludes runs/, weights/, caches and virtualenv
README.md           # Documentation
requirements.txt    # numpy, torch, pillow (optional extras commented)
```

Baselines share the PC-GDT protocol (same scenes, fine-tuning set, budget, sampler and
PGPF) and reuse `generate.write_sample`, so every method emits one manifest format:

```bash
python -m pc_gdt.baselines.run_baselines --methods diffusionsat diffcr --equalize 6
```

Outputs in `runs/demo/`: `pc_controlnet.pt`, `train_log.json`, `report.json`, and `generated/` with images, condition maps, masks, `manifest.json`, `pgpf_report.json`, `accepted.json`, `coco_*.json`.

## Usage

```bash
python -m pc_gdt.pipeline demo --out runs/exp1 --steps 2000 --seed 0 --pseudo-noise
python -m pc_gdt.pipeline paper --device cuda   # guarded: needs GPU + real checkpoints
```

Flags: `--out`, `--device`, `--seed`, `--steps`, `--pseudo-noise` (noisy conditions at training, clean UE5 maps at inference).

Stage-by-stage API:

```python
from pc_gdt.config import demo_config
from pc_gdt.dataset import SceneDataset, make_splits
from pc_gdt.train import train_pc_controlnet
from pc_gdt.generate import generate_dataset
from pc_gdt.filtering import filter_manifest
from pc_gdt.pgpf import PGPFThresholds
from pc_gdt.export import export_coco

cfg = demo_config(out="runs/exp1", steps=2000, lam=2.0)
tr, va, te = make_splits(cfg.n_train, cfg.n_val, cfg.n_test, seed=cfg.seed)
ds_tr = SceneDataset(tr, cfg.H, cfg.W, cfg.extent, seed=cfg.seed)
ds_te = SceneDataset(te, cfg.H, cfg.W, cfg.extent, seed=cfg.seed + 900)

model = train_pc_controlnet(cfg, ds_train=ds_tr)
manifest = generate_dataset(model, ds_te, list(range(len(ds_te))), cfg)
accepted, report = filter_manifest(manifest, "runs/exp1/generated", PGPFThresholds())
export_coco(accepted, "runs/exp1/generated", splits={"train": [e["id"] for e in accepted]})
```

## Conditions and filtering

**Condition tensor (9 channels):** semantics (1, normalized class index; 16-ch one-hot available) + depth (1) + normals (3) + physical shadow mask (1) + ridge mask (1) + slope aspect (2). Global sun embedding $s=(\cos\theta,\sin\theta,\cos\alpha,\sin\alpha)$ injected via per-channel FiLM.

Azimuths in code are **geographic** (deg CW from North); the paper uses $\theta = \pi/2 - \theta_{geo}$. Convert pixel masks with `conditions.pixel_to_world` before measuring azimuths.

**PGPF — an image is accepted iff all three checks pass:**

| # | Check | Threshold |
|---|---|---|
| 1 | Crown→shadow azimuth vs anti-solar azimuth | ≤ 15° |
| 2 | IoU(predicted shadow, $C_4$) | ≥ 0.65 |
| 3 | Normalized luminance gradient on ridge inside shadow | ≥ 0.15 |

Training objective: $\mathcal{L}_{diff} + \lambda R$ with $\lambda = 2$; $R$ uses the same thresholds as PGPF, evaluated on the one-step $\hat{x}_0$ estimate with SNR weighting. Only the control branch + FiLM are trainable; UNet/VAE/text encoder are frozen.

Sampling: CFG $w \in [3,7]$ (default 5), $T \in [20,50]$ steps (default 30), varied seed; ×2 upscaling with label transfer (coordinates ×2, masks nearest — no geometric warp).

## Reported results (production system, A100)

| Metric | Value |
|---|---|
| FID / KID ↓ | 12.4 / 0.008 |
| Shadow IoU vs $C_4$ ↑ | 0.87 (0.86 after SR) |
| Azimuth error ↓ | 4.2° ± 2.1° |
| PGPF pass rate ↑ | 86.2% (17,234 of 20,000) |
| Residual defects $q_{post}$ | ≈ 1.7% |
| mAP@0.5, pitched-roof shadows | 0.612 → **0.651** (+3.9 pp) |
| mAP@0.5 with 25% manual labels | 0.643 (non-inferior, margin 2 pp) |

Data: 2,847 UAV images + 5,072 OAM-TCD = 7,919 (7,400 train / 519 val); test 65 geographically disjoint scenes (2,156 crowns, 1,691 shadows). Manual labeling reduced by **75%**.

## Stand-ins vs production

| Component | Demo | Production | Hook |
|---|---|---|---|
| Conditions | numpy rasteriser | UE5 + Colosseum | `conditions.rasterise_scene` |
| Target images | procedural render | real UAV / OAM-TCD | `dataset.SceneDataset` |
| UNet / VAE | tiny conv stand-in | SD 2.1 checkpoints | `PCControlNet.from_pretrained_sd21` |
| PGPF measurement | Otsu segmentation | frozen DE-ViT | `filtering.measure_image` |
| Super-resolution | bicubic | Real-ESRGAN x2plus | `generate.upscale_image` |
| FID / KID | feature stand-in | InceptionV3 | `metrics.fid_kid(use_inception=True)` |

## Example

<p align="center">
  <img src="images/compare_ue5.jpg" width="32%" alt="UE5 geometric render">
  <img src="images/compare_pcgdt.jpg" width="32%" alt="PC-GDT synthesis after PGPF">
  <img src="images/compare_uav.jpg" width="32%" alt="Real UAV image of the target domain">
</p>
<p align="center">
  <em>(a) UE5 geometric render</em> &nbsp;&nbsp;&nbsp; <em>(b) PC-GDT synthesis (after PGPF)</em> &nbsp;&nbsp;&nbsp; <em>(c) Real UAV image</em>
</p>

Domain comparison on the same scene (Fig. 3 of the paper): **(a)** deterministic UE5 render — the source of the condition maps C₁–C₆ and the inherited annotation A; **(b)** PC-GDT synthesis after post-generative physical filtering (PGPF); **(c)** real UAV image of the target domain. Solar azimuth and elevation are identical across all panels.

## Hardware

- **Demo:** any CPU, ~4 GB RAM, 128×128, ≈1–3 min.
- **Paper scale:** 1× A100 80 GB, FP16, 512×512 → 1024×1024; 150k steps AdamW, lr 1e-4 cosine (warmup 1000), batch 8; total ≈106 h (72 train + 18 generate + 3 SR + 2 PGPF + 11 DE-ViT); inference ≈4.1 s/image.

## Limitations

Manual UE5 scene modelling; hip/multi-pitch roofs unsupported; no multi-temporal generation; pseudo-condition noise $\max_k \xi_k \approx 0.18$; PGPF uses the frozen target detector, so selection bias is bounded (±0.4 pp) but not eliminated.

## License

Code: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). OAM-TCD imagery: CC BY 4.0 / Apache 2.0 per tile. Third-party checkpoints retain their original licenses.
