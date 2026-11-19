"""Scene-disjoint datasets: (target image, 9-channel C, sun embedding)."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from .geometry import random_scenes
from .conditions import rasterise_scene, perturb_conditions
from .render import render_scene
from .geometry import sun_embedding


def make_splits(n_train=12, n_val=4, n_test=6, seed=0):
    """Disjoint scene blocks (paper: test scenes never used in UE5 builds)."""
    sc = random_scenes(n_train + n_val + n_test, seed)
    return (sc[:n_train], sc[n_train:n_train + n_val], sc[n_train + n_val:])


class SceneDataset(Dataset):
    def __init__(self, scenes, H=128, W=128, extent=60.0,
                 n_shadow_samples=2048, pseudo_noise=False, seed=0):
        self.scenes, self.H, self.W, self.extent = scenes, H, W, extent
        self.maps, self.items = [], []
        rng = np.random.default_rng(seed)
        for i, sc in enumerate(scenes):
            maps = rasterise_scene(sc, H, W, extent, n_shadow_samples, seed=seed + i)
            clean = maps
            if pseudo_noise:
                maps = perturb_conditions(maps, rng)
            img = render_scene(sc, clean, extent, np.random.default_rng(seed + 1000 + i))
            cond = maps.concatenated_normalised().transpose(2, 0, 1)
            emb = sun_embedding(sc.sun_azimuth, sc.sun_elevation)[0]
            self.maps.append(clean)
            self.items.append((torch.from_numpy(img), torch.from_numpy(cond),
                               torch.from_numpy(emb.astype(np.float32)), i))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img, cond, emb, i = self.items[idx]
        return {"image": img, "cond": cond, "sun": emb, "idx": torch.tensor(i)}