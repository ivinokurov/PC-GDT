"""Regression tests (pytest-compatible; also: python -m pc_gdt.tests.test_all)."""
from __future__ import annotations
import numpy as np
import torch
from ..geometry import (sun_direction, sun_embedding, angular_difference_deg,
                        make_gabled_house, random_scenes, project_point_on_face,
                        shadow_points_on_face, Canopy, Scene, illuminated_faces)
from ..conditions import rasterise_scene, IDX_SLOPE
from ..model import PCControlNet, pc_gdt_loss, sample, FiLMSunModulator
from ..pgpf import (posterior_defect_rate, posterior_interval, expected_acceptance,
                    solve_total_false_rejection, run_pgpf, SampleMeasurements, _dilate)
from ..violations import compute_violations


def test_sun():
    d = sun_direction(123.0, 30.0)
    assert abs(np.linalg.norm(d) - 1) < 1e-9 and d[2] < 0
    e1 = sun_embedding(10.0, 30.0); e2 = sun_embedding(370.0, 30.0)
    assert np.allclose(e1, e2, atol=1e-9)
    assert abs(angular_difference_deg(350, 10) + 20) < 1e-9


def test_contains_xy_both_orientations():
    faces, _ = make_gabled_house(0, 0)
    for f in faces:                       # left face is CW, right CCW
        v = f.vertices[:, :2]
        edge_mid = (v[0] + v[1]) / 2
        assert f.contains_xy(edge_mid)[0], "boundary point must be inside"
        assert f.contains_xy(v.mean(0))[0]


def test_projection_and_occlusion():
    faces, ridge = make_gabled_house(0, 0)
    sc = Scene(faces, [], 0.0, 45.0, [ridge])
    d = sc.sun_vector
    fi = illuminated_faces(sc)[0]
    cp = Canopy(np.array([0.0, 0.0, 8.0]), 2.0)
    pts = shadow_points_on_face(cp, faces[fi], d, 512)
    blocker = Canopy(cp.centre - d * 3.0, 5.0)      # between sun and canopy
    pts2 = shadow_points_on_face(cp, faces[fi], d, 512, [blocker])
    assert len(pts) > 0 and len(pts2) == 0, "sun-ray occlusion must block"


def test_conditions_dense_and_channels():
    sc = random_scenes(1, 3)[0]
    m = rasterise_scene(sc, 96, 96, seed=1)
    assert m.concatenated_normalised().shape[-1] == 9
    assert m.concatenated_onehot().shape[-1] == 16
    roof = m.semantic == IDX_SLOPE
    assert roof.any() and (m.depth[roof] > 0.2).mean() > 0.9, "depth must fill faces"
    g = m.normals[roof][:, :2]
    o = m.slope_orient[roof]
    assert (np.sum(g * o, axis=1) <= 1e-5).mean() > 0.9, "aspect = steepest descent"


def test_film_identity_and_channels():
    f = FiLMSunModulator([8, 8])
    e = torch.randn(2, 4)
    (g, b), _ = f.params(e)
    assert g.shape == (2, 8) and torch.allclose(g, torch.ones_like(g))
    assert torch.allclose(b, torch.zeros_like(b))


def test_model_shapes_freeze_and_grad():
    torch.manual_seed(0)
    m = PCControlNet(base_ch=16)
    m.T_total = 50
    m.trainability()
    x = torch.rand(2, 3, 64, 64) * 2 - 1
    cond = torch.rand(2, 9, 64, 64)
    emb = torch.randn(2, 4)
    loss, _ = pc_gdt_loss(m, x, cond, emb, lam=2.0, T=50)
    loss.backward()
    assert m.ctl_zero.weight.grad is not None, "penalty/diff grad must reach control branch"
    assert m.main_b1.conv1.weight.grad is None, "backbone must be frozen"
    out = sample(m, cond, emb, shape=(2, 3, 64, 64), steps=5, T_total=50, seed=1)
    assert out.shape == (2, 3, 64, 64) and torch.isfinite(out).all()
    assert out.abs().max() <= 1.0 + 1e-5


def test_violations_zero_on_perfect_and_finite():
    cond = torch.zeros(1, 9, 32, 32)
    cond[:, 0] = 4 / 7                       # slope everywhere
    cond[:, 5, 8:16, 8:16] = 1               # physical shadow
    cond[:, 6, 12, 8:16] = 1                 # ridge
    x = torch.ones(1, 3, 32, 32) * 0.6
    x[:, :, 8:16, 8:16] = -0.6               # dark shadow exactly on C4
    emb = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    v = compute_violations(x, cond, emb)
    assert torch.isfinite(v.total()).all() and v.iou_deficit.item() < 0.05


def test_pgpf_numbers():
    assert abs(posterior_defect_rate(.10, .85, .02) - 0.016722) < 1e-4
    lo, hi = posterior_interval((.06, .15), (.75, .90), .02)
    assert abs(lo - 0.00647) < 1e-3 and abs(hi - 0.04308) < 1e-3
    assert abs(expected_acceptance(.10, .85, .02) - 0.897) < 1e-9
    flo, fhi = solve_total_false_rejection(.862, .10, .85)
    assert abs(flo - 0.0422) < 1e-3 and abs(fhi - 0.0589) < 1e-3


def test_pgpf_stages_and_dilate():
    ms = [SampleMeasurements(27, .71, .22), SampleMeasurements(9, .41, .19),
          SampleMeasurements(6, .68, .08), SampleMeasurements(5, .9, .3)]
    r = run_pgpf(ms)
    assert list(r.rejection_stage) == [1, 2, 3, 0]
    m = np.zeros((5, 5), bool); m[0, 0] = True
    assert not _dilate(m, 1)[-1, -1], "dilation must not wrap"


def test_loss_decreases_tiny_overfit():
    from ..dataset import SceneDataset
    from ..geometry import random_scenes
    from ..train import pretrain_vae
    torch.manual_seed(0)
    ds = SceneDataset(random_scenes(2, 7), 64, 64, seed=7)
    m = PCControlNet(base_ch=16); m.T_total = 50
    pretrain_vae(m, ds, steps=30, log=lambda *a: None)
    m.trainability()
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-3)
    b = ds[0]
    x, c, e = b["image"][None], b["cond"][None], b["sun"][None]
    l0, _ = pc_gdt_loss(m, x, c, e, lam=0.0, T=50)
    for _ in range(60):
        loss, _ = pc_gdt_loss(m, x, c, e, lam=0.0, T=50)
        opt.zero_grad(); loss.backward(); opt.step()
    l1, _ = pc_gdt_loss(m, x, c, e, lam=0.0, T=50)
    assert l1.item() < l0.item()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok:", fn.__name__)
    print(f"{len(fns)} tests passed")