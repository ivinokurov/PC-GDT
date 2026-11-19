"""Pipeline configurations. `demo` is CPU-runnable; `paper` lists the real
recipe (needs GPU + real SD 2.1 / ControlNet / VAE checkpoints and real data)."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Config:
    out: str = "runs/demo"
    device: str = "cpu"
    seed: int = 0
    H: int = 128; W: int = 128; extent: float = 60.0
    n_train: int = 12; n_val: int = 4; n_test: int = 6
    n_shadow_samples: int = 2048
    pseudo_noise: bool = False
    base_ch: int = 32
    T_total: int = 200
    vae_steps: int = 120
    steps: int = 250
    warmup: int = 20
    lr: float = 1e-4
    batch: int = 4
    lam: float = 2.0
    use_film: bool = True
    use_controlnet: bool = True
    w: float = 5.0
    gen_steps: int = 30
    upscale: int = 2
    delta_az: float = 15.0
    eta_iou: float = 0.65
    tau_ridge: float = 0.15


def demo_config(**kw) -> Config:
    return Config(**kw)


def paper_config(**kw) -> Config:
    """Paper recipe: 512x512, T_total=1000, 150k steps, warmup 1000, batch 8,
    7400 train / 519 val real images, 20k generated, w in [3,7], T in [20,50].
    Requires GPU and real checkpoints (model.from_pretrained_sd21)."""
    c = Config(H=512, W=512, T_total=1000, steps=150_000, warmup=1000,
               batch=8, n_train=7400, n_val=519, device="cuda", vae_steps=0)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# DE-ViT hyperparameters (paper, sec. 4) -- consumed by the external detector repo.
DEVIT = dict(layers=12, embed_dim=768, heads=12, d_q=256,
             lambda_def=0.1, lambda_cross=0.5, lambda_roof=0.2, lambda_connect=0.3,
             epochs=100, input=1024)