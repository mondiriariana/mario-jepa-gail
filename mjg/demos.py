"""In-memory human demonstration set built from scripts/build_dataset.py outputs, with
the train / held-out split, latent normalisation and batch samplers for BC, JEPA and the
discriminator."""
from __future__ import annotations

import os

import numpy as np
import torch

from . import data as D
from .actions import N_ACTIONS


def split_plays(rows, heldout_every=6, val_offset=2):
    """Split by whole sessions of each subject (in session order):
    held-out test = sessions heldout_every-1, 2*heldout_every-1, ... (never used to choose
    anything); validation = sessions val_offset, val_offset+heldout_every, ... (checkpoint
    selection); train = the rest."""
    train, val, held = [], [], []
    for sub in sorted({r["sub"] for r in rows}):
        sess = sorted({r["ses"] for r in rows if r["sub"] == sub})
        held_ses = set(sess[heldout_every - 1::heldout_every])
        val_ses = set(sess[val_offset::heldout_every]) - held_ses
        for r in rows:
            if r["sub"] == sub:
                (held if r["ses"] in held_ses else val if r["ses"] in val_ses else train).append(r)
    return train, val, held


class DemoSet:
    def __init__(self, feat_dir, rows, horizon=4, stats=None, device="cuda"):
        self.rows, self.horizon, self.device = rows, horizon, device
        zs, acts, ctrl, play, step, xs = [], [], [], [], [], []
        for k, r in enumerate(rows):
            z = np.load(os.path.join(feat_dir, r["play_id"] + ".z.npy"))
            m = np.load(os.path.join(feat_dir, r["play_id"] + ".meta.npz"))
            n = len(z)
            zs.append(z)
            acts.append(m["actions"][:n])
            ctrl.append(m["control"][:n])
            xs.append(m["x"][:n])
            play.append(np.full(n, k))
            step.append(np.arange(n))
        self.z = torch.from_numpy(np.concatenate(zs))  # float16 [N,16,1024] on CPU
        self.a = torch.from_numpy(np.concatenate(acts)).long()
        self.control = torch.from_numpy(np.concatenate(ctrl)).bool()
        self.play = torch.from_numpy(np.concatenate(play)).long()
        self.step = torch.from_numpy(np.concatenate(step)).long()
        self.x = torch.from_numpy(np.concatenate(xs)).long()
        prev = torch.roll(self.a, 1)
        prev[self.step == 0] = N_ACTIONS
        self.prev = prev
        N, K = len(self.a), horizon
        same = torch.zeros(N, dtype=torch.bool)
        same[:-K] = self.play[K:] == self.play[:-K]
        self.jepa_idx = torch.nonzero(same).squeeze(1)
        self.bc_idx = torch.nonzero(self.control).squeeze(1)
        if stats is None:
            sel = torch.randperm(N, generator=torch.Generator().manual_seed(0))[:20000]
            zz = self.z[sel].float()
            stats = {"mu": zz.mean((0, 1)), "sd": zz.std((0, 1)).clamp_min(1e-3)}
        self.stats = stats
        self.mu = stats["mu"].to(device)
        self.sd = stats["sd"].to(device)

    def __len__(self):
        return len(self.a)

    def norm(self, z):
        return (z.to(self.device, non_blocking=True).float() - self.mu) / self.sd

    def bc_batch(self, n, gen=None):
        i = self.bc_idx[torch.randint(len(self.bc_idx), (n,), generator=gen)]
        return self.norm(self.z[i]), self.prev[i].to(self.device), self.a[i].to(self.device)

    def jepa_batch(self, n, gen=None, idx=None):
        i = self.jepa_idx[torch.randint(len(self.jepa_idx), (n,), generator=gen)] if idx is None else idx
        K = self.horizon
        acts = torch.stack([self.a[i + k] for k in range(K)], 1)
        return self.norm(self.z[i]), acts.to(self.device), self.norm(self.z[i + K])

    def disc_batch(self, n, gen=None):
        return self.bc_batch(n, gen)

    def level_freq(self):
        levels = [r["level"] for r in self.rows]
        u = sorted(set(levels))
        steps = {l: sum(r["n_control"] for r in self.rows if r["level"] == l) for l in u}
        return u, np.array([steps[l] for l in u], float)
