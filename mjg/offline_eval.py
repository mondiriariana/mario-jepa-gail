"""Offline metrics on held-out human plays: action agreement and latent prediction.

Everything here is computed from data; nothing is estimated or extrapolated.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .actions import N_ACTIONS


@torch.no_grad()
def action_metrics(agent, demo, batch=2048):
    agent.eval()
    idx = demo.bc_idx
    preds, nll, probs_true = [], [], []
    for s in range(0, len(idx), batch):
        i = idx[s:s + batch]
        logits, _, _ = agent(demo.norm(demo.z[i]), demo.prev[i].to(demo.device))
        y = demo.a[i].to(demo.device)
        preds.append(logits.argmax(-1).cpu())
        nll.append(F.cross_entropy(logits, y, reduction="none").cpu())
    agent.train()
    p = torch.cat(preds).numpy()
    y = demo.a[idx].numpy()
    prev = demo.prev[idx].numpy()
    return _summary(y, p, torch.cat(nll).numpy(), prev)


def _summary(y, p, nll, prev):
    cm = np.zeros((N_ACTIONS, N_ACTIONS), int)
    np.add.at(cm, (y, p), 1)
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
    f1 = 2 * recall * prec / np.maximum(recall + prec, 1e-9)
    present = cm.sum(1) > 0
    majority = np.bincount(y, minlength=N_ACTIONS).argmax()
    repeat_ok = prev < N_ACTIONS
    return {
        "n": int(len(y)),
        "accuracy": float((y == p).mean()),
        "nll": float(nll.mean()),
        "balanced_accuracy": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "baseline_majority_accuracy": float((y == majority).mean()),
        "baseline_repeat_prev_accuracy": float((y[repeat_ok] == prev[repeat_ok]).mean()),
        "accuracy_when_action_changes": float((y == p)[repeat_ok & (y != prev)].mean()),
        "pred_action_hist": np.bincount(p, minlength=N_ACTIONS).tolist(),
        "true_action_hist": np.bincount(y, minlength=N_ACTIONS).tolist(),
        "confusion": cm.tolist(),
    }


@torch.no_grad()
def jepa_metrics(agent, demo, n=20000, batch=1024, seed=0):
    """Held-out latent prediction K steps ahead, against simple controls:
    copy (z_hat = z_t), mean (z_hat = training mean = 0 in normalised units),
    shuffled actions (true state, actions from another random sample), all-NOOP actions."""
    if agent.predictor is None:
        return None
    agent.eval()
    g = torch.Generator().manual_seed(seed)
    idx = demo.jepa_idx[torch.randperm(len(demo.jepa_idx), generator=g)[:n]]
    acc = {k: [] for k in ["l1", "mse", "cos", "copy_mse", "copy_l1", "mean_mse", "shuf_mse", "noop_mse",
                           "mse_on_diff", "shuf_mse_on_diff"]}
    var = []
    for s in range(0, len(idx), batch):
        i = idx[s:s + batch]
        z, acts, zt = demo.jepa_batch(0, idx=i)
        H, _ = agent.adapter(z)
        zh, _ = agent.predictor(H, acts, z)
        perm = torch.randperm(len(i), device=acts.device)
        zs, _ = agent.predictor(H, acts[perm], z)
        zn, _ = agent.predictor(H, torch.zeros_like(acts), z)
        acc["l1"].append((zh - zt).abs().mean((1, 2)))
        acc["mse"].append(((zh - zt) ** 2).mean((1, 2)))
        acc["cos"].append(F.cosine_similarity(zh.flatten(1), zt.flatten(1), dim=-1))
        acc["copy_mse"].append(((z - zt) ** 2).mean((1, 2)))
        acc["copy_l1"].append((z - zt).abs().mean((1, 2)))
        acc["mean_mse"].append((zt ** 2).mean((1, 2)))
        acc["shuf_mse"].append(((zs - zt) ** 2).mean((1, 2)))
        acc["noop_mse"].append(((zn - zt) ** 2).mean((1, 2)))
        # same comparison restricted to samples whose shuffled action sequence differs
        diff = (acts != acts[perm]).any(1)
        acc["mse_on_diff"].append(((zh - zt) ** 2).mean((1, 2))[diff])
        acc["shuf_mse_on_diff"].append(((zs - zt) ** 2).mean((1, 2))[diff])
        var.append(zt)
    agent.train()
    out = {k: float(torch.cat(v).mean()) for k, v in acc.items()}
    # fraction of target variance explained, relative to the per-dim held-out variance
    zt = torch.cat(var)
    tot = ((zt - zt.mean(0)) ** 2).mean()
    out["r2"] = float(1 - out["mse"] / tot)
    out["copy_r2"] = float(1 - out["copy_mse"] / tot)
    out["shuf_r2"] = float(1 - out["shuf_mse"] / tot)
    out["n"] = int(len(idx))
    return out
