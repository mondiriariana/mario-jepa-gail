"""Do the learned representations carry action consequences?

Linear probes from a frozen representation r_t (plus, for the action-conditioned probe,
the next K actions) to what the game does next, fitted on training-session steps and
scored on held-out-session steps. Targets come from emulator RAM:

  dx  : horizontal displacement x_{t+K} - x_t (px)          (ridge, R^2)
  dy  : vertical displacement of Mario on screen              (ridge, R^2)
  air : Mario airborne at t+K                                 (logistic, AUC)
  die : a death starts within the next 30 steps (2 s): player_state 6/11 or Mario
        below the screen, the rule validated in results/death_rule_check.json (AUC)

Representations: raw frozen V-JEPA 2 latent (token mean, 1024-d) and, per checkpoint, the
adapter output [h_t, mean(H_t)] (512-d). Each is probed with the true next actions,
with actions shuffled across samples (control), and with no actions.

python scripts/probe_representations.py --out results/probes.json NAME=ckpt ...
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mjg import common as C  # noqa: E402
from mjg.actions import N_ACTIONS  # noqa: E402

K = 4


def targets(feat_dir, rows):
    out = {k: [] for k in ("dx", "dy", "air", "die", "valid", "play")}
    for r in rows:
        m = np.load(os.path.join(feat_dir, r["play_id"] + ".meta.npz"))
        x, y = m["x"].astype(float), m["ram_player_y_pos"].astype(float)
        air, ctrl = m["ram_jump_airborne"] > 0, m["control"]
        ps, ys = m["ram_player_state"], m["ram_player_y_screen"]
        n = len(x)
        fut = np.minimum(np.arange(n) + K, n - 1)
        out["dx"].append(x[fut] - x)
        out["dy"].append(y[fut] - y)
        out["air"].append(air[fut])
        dying = np.isin(ps, (6, 11)) | (ys >= 2)
        onsets = np.flatnonzero(dying[1:] & ~dying[:-1]) + 1
        die = np.zeros(n, bool)
        for t in onsets:
            die[max(0, t - 30):t] = True
        out["die"].append(die)
        # same play, controllable now and K steps later, no wrap-around of x across areas
        ok = (np.arange(n) + K < n) & ctrl & ctrl[fut] & (np.abs(x[fut] - x) < 200)
        out["valid"].append(ok)
        out["play"].append(np.full(n, len(out["play"])))
    return {k: np.concatenate(v) for k, v in out.items()}


@torch.no_grad()
def represent(agent, demo, idx, batch=4096):
    if agent is None:
        return torch.cat([demo.norm(demo.z[idx[s:s + batch]]).mean(1).cpu() for s in range(0, len(idx), batch)]).numpy()
    outs = []
    for s in range(0, len(idx), batch):
        H, h = agent.adapter(demo.norm(demo.z[idx[s:s + batch]]))
        outs.append(torch.cat([h, H.mean(1)], -1).cpu())
    return torch.cat(outs).numpy()


def action_block(demo, idx):
    acts = np.stack([demo.a[idx + k].numpy() for k in range(K)], 1)
    oh = np.zeros((len(idx), K * N_ACTIONS), np.float32)
    for k in range(K):
        oh[np.arange(len(idx)), k * N_ACTIONS + acts[:, k]] = 1
    return oh


def probe(Xtr, Xte, ytr, binary):
    """Fit on training sessions, return held-out predictions."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if binary:
        return LogisticRegression(C=0.05, max_iter=3000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    return Ridge(alpha=100.0).fit(Xtr, ytr).predict(Xte)


def score(y, p, binary):
    from sklearn.metrics import r2_score, roc_auc_score
    return float(roc_auc_score(y, p)) if binary else float(r2_score(y, p))


BOOT = None  # list of index arrays: held-out samples of resampled plays


def boot_scores(y, p, binary):
    return np.array([score(y[i], p[i], binary) for i in BOOT])


def summary(y, p, binary):
    b = boot_scores(y, p, binary)
    return {"score": score(y, p, binary), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}


def paired_diff(y, p1, p2, binary):
    """Bootstrap over held-out plays of score(p1) - score(p2)."""
    d = boot_scores(y, p1, binary) - boot_scores(y, p2, binary)
    return {"diff": score(y, p1, binary) - score(y, p2, binary),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-train", type=int, default=60000)
    a = ap.parse_args()
    specs = [s.split("=", 1) for s in a.ckpt]
    _, _, ck0 = C.load_ckpt(specs[0][1], a.device)
    cfg, stats = ck0["cfg"], ck0["stats"]
    d = C.load_demos(cfg, a.device, stats=stats, which=("train", "heldout"))
    tr, ho = d["train"], d["heldout"]
    T = {"train": targets(C.features_dir(cfg), tr.rows), "heldout": targets(C.features_dir(cfg), ho.rows)}
    rng = np.random.default_rng(0)
    itr = np.flatnonzero(T["train"]["valid"])
    itr = np.sort(rng.choice(itr, min(a.n_train, len(itr)), replace=False))
    ite = np.flatnonzero(T["heldout"]["valid"])
    Atr, Ate = action_block(tr, torch.from_numpy(itr)), action_block(ho, torch.from_numpy(ite))
    Ate_shuf = Ate[rng.permutation(len(Ate))]
    Atr_shuf = Atr[rng.permutation(len(Atr))]
    global BOOT
    plays = T["heldout"]["play"][ite]
    by_play = [np.flatnonzero(plays == p) for p in np.unique(plays)]
    brng = np.random.default_rng(1)
    BOOT = [np.concatenate([by_play[j] for j in brng.integers(0, len(by_play), len(by_play))]) for _ in range(500)]
    TGTS = (("dx", False), ("dy", False), ("air", True), ("die", True))
    res = {"n_train": int(len(itr)), "n_heldout": int(len(ite)), "n_heldout_plays": len(by_play), "K": K,
           "results": {}, "paired": {}}
    reps = [("V-JEPA2 (frozen)", None)] + [(n, C.load_ckpt(p, a.device)[0].eval()) for n, p in specs]
    preds = {}
    for name, agent in reps:
        Rtr = represent(agent, tr, torch.from_numpy(itr))
        Rte = represent(agent, ho, torch.from_numpy(ite))
        row = {}
        for tgt, binary in TGTS:
            ytr, yte = T["train"][tgt][itr], T["heldout"][tgt][ite]
            pa = probe(np.c_[Rtr, Atr], np.c_[Rte, Ate], ytr, binary)
            ps = probe(np.c_[Rtr, Atr_shuf], np.c_[Rte, Ate_shuf], ytr, binary)
            po = probe(Rtr, Rte, ytr, binary)
            preds[(name, tgt)] = pa
            row[tgt] = {"state+actions": summary(yte, pa, binary),
                        "state+shuffled_actions": summary(yte, ps, binary),
                        "state_only": summary(yte, po, binary),
                        "action_gain": paired_diff(yte, pa, ps, binary)}
        res["results"][name] = row
        print(name, json.dumps({t: {k: round(v["score"] if "score" in v else v["diff"], 4) for k, v in r.items()}
                                for t, r in row.items()}), flush=True)
    names = [n for n, _ in reps]
    pairs = [(x, y) for x, y in [("BC+JEPA", "BC"), ("GAIL+JEPA", "GAIL"), ("GAIL+JEPA (BC reg.)", "GAIL (BC reg.)")]
             if x in names and y in names] + [(n, "V-JEPA2 (frozen)") for n in names[1:]]
    for x, y in pairs:
        res["paired"][f"{x} - {y}"] = {t: paired_diff(T["heldout"][t][ite], preds[(x, t)], preds[(y, t)], b)
                                       for t, b in TGTS}
        print(f"{x} - {y}", json.dumps(res["paired"][f"{x} - {y}"]), flush=True)
    res["actions_only"] = {t: score(T["heldout"][t][ite], probe(Atr, Ate, T["train"][t][itr], b), b) for t, b in TGTS}
    print("actions only", json.dumps(res["actions_only"]), flush=True)
    C.dump_json(a.out, res)


if __name__ == "__main__":
    main()
