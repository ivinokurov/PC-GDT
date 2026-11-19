"""Baseline-adapter tests (pytest-compatible; python -m pc_gdt.tests.test_baselines)."""
from __future__ import annotations
import os, tempfile
import numpy as np
import torch
from ..dataset import SceneDataset
from ..geometry import random_scenes
from ..baselines.diffusionsat import DiffusionSatAdapter, metadata_channels
from ..baselines.diffcr import DiffCRAdapter
from ..filtering import filter_manifest


def _ds(n=2, seed=5):
    return SceneDataset(random_scenes(n, seed), 64, 64, seed=seed)


def test_hint_channels_and_no_physical_leak():
    ds = _ds(1)
    render = ds[0]["image"].numpy()
    sat = DiffusionSatAdapter(base_ch=16, T_total=50)
    cr = DiffCRAdapter(base_ch=16, T_total=50)
    h_sat = sat.build_hint(render, ds.maps[0], ds.scenes[0], ds.extent)
    h_cr = cr.build_hint(render, ds.maps[0], ds.scenes[0], ds.extent)
    assert h_sat.shape == (6, 64, 64) and h_cr.shape == (1, 64, 64)
    assert np.allclose(h_cr[0], ds.maps[0].shadow_mask)          # DiffCR: C4 as cloud mask
    # DiffusionSat metadata must not encode the physical shadow mask / normals
    assert not np.allclose(h_sat[0], ds.maps[0].shadow_mask)


def test_finetune_loss_decreases():
    torch.manual_seed(0)
    ds = _ds(2)
    cr = DiffCRAdapter(base_ch=16, T_total=50)
    x0 = ds[0]["image"][None]
    hint = torch.from_numpy(np.concatenate(
        [x0[0].numpy(), cr.build_hint(x0[0].numpy(), ds.maps[0], ds.scenes[0], ds.extent)], 0))[None]
    ab = torch.cumprod(1.0 - torch.linspace(1e-4, 0.02, 50), 0)
    opt = torch.optim.AdamW(cr.net.parameters(), lr=1e-3)

    def loss():
        t = torch.randint(0, 50, (1,)); a = ab[t].view(1, 1, 1, 1)
        noise = torch.randn_like(x0); xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
        return torch.nn.functional.mse_loss(cr.net(xt, t, hint), noise)

    l0 = loss().item()
    for _ in range(60):
        l = loss(); opt.zero_grad(); l.backward(); opt.step()
    assert loss().item() < l0


def test_generate_and_shared_pgpf():
    torch.manual_seed(0)
    ds = _ds(2)
    sat = DiffusionSatAdapter(base_ch=16, T_total=50)
    with tempfile.TemporaryDirectory() as out:
        manifest = sat.generate(ds, out, count=2, steps=5, seed=0, upscale=2)
        assert len(manifest) == 2 and os.path.exists(os.path.join(out, "manifest.json"))
        accepted, report = filter_manifest(manifest, out, log=lambda *a: None)
        assert "pass_rates" in report and len(accepted) <= 2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok:", fn.__name__)
    print(f"{len(fns)} tests passed")