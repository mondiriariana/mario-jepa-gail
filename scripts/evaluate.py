"""Evaluate trained agents by playing the real game and comparing with held-out humans.

For each checkpoint: play ``--episodes`` episodes per level (stochastic policy, as GAIL
matches a distribution), record every episode's actions and RAM trajectory, save one
gameplay video per level and the emulator's own .bk2 replays (deterministically
re-playable, same format as the CNeuroMod human data), then compute:

  * held-out human action agreement and held-out latent prediction (offline);
  * trajectory statistics vs held-out human attempts on the same levels: action
    distribution (Jensen-Shannon), action switch rate, jump rate, run-button use,
    horizontal speed, progress per attempt, death and completion rates;
  * a classifier two-sample test (C2ST): 5-fold logistic regression separating agent
    from held-out human (z, a_prev, a) samples. 0.5 = indistinguishable.

Usage:
  python scripts/evaluate.py --out results/eval --ckpt BC=runs/bc/bc.pt GAIL+JEPA=runs/x/gail_last.pt ...
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mjg import actions as A  # noqa: E402
from mjg import common as C  # noqa: E402
from mjg.encoder import FrozenVJEPA2  # noqa: E402
from mjg.offline_eval import action_metrics, jepa_metrics  # noqa: E402
from mjg.vec_env import MarioVecEnv  # noqa: E402

STEP_S = A.FRAME_SKIP / 60.0
JUMP = np.array(["A" in c for c in A.COMBOS])
RUN = np.array(["B" in c for c in A.COMBOS])
RIGHT = np.array(["RIGHT" in c for c in A.COMBOS])
LEFT = np.array(["LEFT" in c for c in A.COMBOS])


def js(p, q):
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log2(np.maximum(a, 1e-12) / np.maximum(b, 1e-12)), 0))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def action_stats(acts_list):
    """acts_list: list of int arrays (control steps of one attempt, in order)."""
    allA = np.concatenate(acts_list)
    hist = np.bincount(allA, minlength=A.N_ACTIONS).astype(float)
    switches, n_pairs, jumps, runs = 0, 0, 0, []
    for a in acts_list:
        if len(a) < 2:
            continue
        switches += int((a[1:] != a[:-1]).sum())
        n_pairs += len(a) - 1
        jumps += int((JUMP[a[1:]] & ~JUMP[a[:-1]]).sum())
        # run lengths of identical consecutive actions
        cuts = np.flatnonzero(np.diff(a)) + 1
        runs.extend(np.diff(np.r_[0, cuts, len(a)]).tolist())
    T = len(allA) * STEP_S
    return {"hist": hist, "switch_rate": switches / max(n_pairs, 1), "jumps_per_s": jumps / max(T, 1e-9),
            "run_frac": float(RUN[allA].mean()), "right_frac": float(RIGHT[allA].mean()),
            "left_frac": float(LEFT[allA].mean()), "noop_frac": float((allA == 0).mean()),
            "mean_hold_steps": float(np.mean(runs)) if runs else 0.0}


def human_attempts(feat_dir, rows, levels):
    """Split held-out human plays into attempts (one life each) on the given levels."""
    out = []
    for r in rows:
        if r["level"] not in levels:
            continue
        m = np.load(os.path.join(feat_dir, r["play_id"] + ".meta.npz"))
        lives, ctrl, x, ps = m["ram_lives"], m["control"], m["x"], m["ram_player_state"]
        cuts = np.flatnonzero(np.diff(lives) != 0) + 1
        for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(lives)]):
            c = ctrl[s:e]
            if c.sum() < 10:
                continue
            idx = np.arange(s, e)[c]
            out.append({"level": r["level"], "acts": m["actions"][idx], "steps": int(idx[-1] - idx[0] + 1),
                        "dx": int(x[idx].max() - x[idx[0]]), "speed": float((x[idx[-1]] - x[idx[0]]) / ((idx[-1] - idx[0] + 1) * STEP_S)),
                        "completed": bool((ps[s:e] == 4).any()),
                        "died": bool(e < len(lives) and lives[e] < lives[s]), "play": r["play_id"], "z_idx": idx})
    return out


def save_video(frames, path, fps=30):
    import av
    for codec in ("libx264", "h264", "mpeg4"):
        try:
            with av.open(path, "w") as c:
                s = c.add_stream(codec, rate=fps)
                s.width, s.height, s.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
                if codec == "mpeg4":
                    s.bit_rate = 4_000_000
                for f in frames:
                    for p in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                        c.mux(p)
                for p in s.encode():
                    c.mux(p)
            return codec
        except Exception as e:  # codec not available in this ffmpeg build
            err = e
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames[::2]]
    imgs[0].save(path.replace(".mp4", ".gif"), save_all=True, append_images=imgs[1:], duration=1000 // 15)
    return f"gif ({err})"


@torch.no_grad()
def play(agent, stats, enc, levels, episodes, n_envs, integration, max_steps, seed, record_dir, greedy=False):
    mu, sd = stats["mu"].cuda(), stats["sd"].cuda()
    per_env = [levels[i % len(levels)] for i in range(n_envs)]
    env = MarioVecEnv(n_envs, integration, levels, max_steps=max_steps, seed=seed, record_dir=record_dir,
                      fixed_levels=per_env)
    clips, prev = env.reset()
    need = {l: episodes for l in levels}
    eps = []
    cur = [dict(acts=[], ctrl=[], z=[], prev=[], frames=[], x=[]) for _ in range(n_envs)]
    video_done = set()
    control = env.control.copy()
    while any(v > 0 for v in need.values()):
        z = (enc(torch.from_numpy(clips)).float() - mu) / sd
        logits, _, _ = agent(z, torch.from_numpy(prev).cuda())
        act = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
        act = act.cpu().numpy()
        zc = z.mean(1).cpu().numpy().astype(np.float16)
        for i in range(n_envs):
            cur[i]["acts"].append(act[i]); cur[i]["ctrl"].append(control[i])
            cur[i]["z"].append(zc[i]); cur[i]["prev"].append(prev[i])
        lv_before = list(env.level)
        clips, prev, done, infos = env.step(act)
        for i in range(n_envs):
            lv = lv_before[i]
            cur[i]["x"].append(infos[i]["x"])
            if lv not in video_done:
                cur[i]["frames"].extend([clips[i, -2], clips[i, -1]] if not done[i] else [])
            if done[i]:
                e = cur[i]
                if need[lv] > 0:
                    need[lv] -= 1
                    acts, ctrl = np.array(e["acts"]), np.array(e["ctrl"], bool)
                    x = np.array(e["x"])
                    eps.append({"level": lv, "acts": acts[ctrl], "prev": np.array(e["prev"])[ctrl],
                                "z": np.array(e["z"])[ctrl], "steps": int(infos[i]["t"]), "dx": int(infos[i]["dx"]),
                                "speed": float((x[-1] - x[0]) / (len(x) * STEP_S)) if len(x) else 0.0,
                                "died": bool(infos[i]["died"]), "completed": bool(infos[i]["completed"]),
                                "truncated": bool(infos[i]["truncated"]), "x_traj": x, "all_acts": acts,
                                "bk2_dir": infos[i].get("bk2_dir"),
                                "frames": np.array(e["frames"]) if (lv not in video_done and e["frames"]) else None})
                    if eps[-1]["frames"] is not None:
                        video_done.add(lv)
                cur[i] = dict(acts=[], ctrl=[], z=[], prev=[], frames=[], x=[])
        control = env.control.copy()
    env.close()
    return eps


def c2st(z_a, prev_a, a_a, g_a, z_h, prev_h, a_h, g_h, seed=0, n_max=6000):
    """Classifier two-sample test. Folds are grouped by agent episode / human play, so
    steps of one trajectory never sit in both the training and the test fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    rng = np.random.default_rng(seed)
    n = min(len(a_a), len(a_h), n_max)
    ia, ih = rng.choice(len(a_a), n, replace=False), rng.choice(len(a_h), n, replace=False)
    groups = np.r_[g_a[ia], 1_000_000 + g_h[ih]]
    oh = lambda v, k: np.eye(k)[v]
    feat = lambda z, p, a: np.concatenate([z, oh(a, A.N_ACTIONS), oh(p, A.N_ACTIONS + 1)], 1)
    X = np.concatenate([feat(z_a[ia], prev_a[ia], a_a[ia]), feat(z_h[ih], prev_h[ih], a_h[ih])]).astype(np.float32)
    y = np.r_[np.zeros(n), np.ones(n)]
    out = {"n_per_class": int(n)}
    for name, cols in [("full", slice(None)), ("actions_only", slice(z_a.shape[1], None)),
                       ("latents_only", slice(0, z_a.shape[1]))]:
        clf = LogisticRegression(max_iter=2000, C=0.1)
        cv = cross_val_score(clf, X[:, cols], y, groups=groups, cv=GroupKFold(5), scoring="accuracy")
        out[name] = float(cv.mean())
        out[name + "_fold_sd"] = float(cv.std())
    return out


def summarise(eps_or_attempts):
    st = action_stats([e["acts"] for e in eps_or_attempts if len(e["acts"])])
    st.update({"n_attempts": len(eps_or_attempts), "mean_dx": float(np.mean([e["dx"] for e in eps_or_attempts])),
               "median_dx": float(np.median([e["dx"] for e in eps_or_attempts])),
               "mean_speed_px_s": float(np.mean([e["speed"] for e in eps_or_attempts])),
               "death_rate": float(np.mean([e["died"] for e in eps_or_attempts])),
               "completion_rate": float(np.mean([e["completed"] for e in eps_or_attempts])),
               "mean_steps": float(np.mean([e["steps"] for e in eps_or_attempts]))})
    st["ci95"] = bootstrap_ci(eps_or_attempts)
    return st


def bootstrap_ci(eps, n_boot=2000, seed=0):
    """95% percentile bootstrap intervals over attempts (attempts are the resampling unit)."""
    rng = np.random.default_rng(seed)
    cols = {"mean_dx": np.array([e["dx"] for e in eps], float),
            "mean_speed_px_s": np.array([e["speed"] for e in eps], float),
            "death_rate": np.array([e["died"] for e in eps], float)}
    out = {}
    for k, v in cols.items():
        bs = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
        out[k] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


def js_ci(agent_eps, human_att, n_boot=1000, seed=0):
    """95% bootstrap interval of the action-distribution JS divergence, resampling attempts on both sides."""
    rng = np.random.default_rng(seed)
    ha = [np.bincount(e["acts"], minlength=A.N_ACTIONS) for e in agent_eps if len(e["acts"])]
    hh = [np.bincount(e["acts"], minlength=A.N_ACTIONS) for e in human_att if len(e["acts"])]
    ha, hh = np.array(ha, float), np.array(hh, float)
    vals = [js(ha[rng.integers(0, len(ha), len(ha))].sum(0), hh[rng.integers(0, len(hh), len(hh))].sum(0))
            for _ in range(n_boot)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="NAME=path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", nargs="*", default=None)
    ap.add_argument("--n-levels", type=int, default=4)
    ap.add_argument("--extra-levels", nargs="*", default=[], help="added to the automatically chosen levels")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    enc = FrozenVJEPA2()
    integ = os.environ["MJG_INTEGRATION"]
    results = {}
    human = None
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        agent, _, ck = C.load_ckpt(path)
        agent.eval()
        cfg = ck["cfg"]
        demos = C.load_demos(cfg, stats=ck["stats"], which=("heldout",))
        ho = demos["heldout"]
        if human is None:
            if a.levels:
                levels = a.levels
            else:
                lv, fr = ho.level_freq()
                levels = [lv[i] for i in np.argsort(-fr)[: a.n_levels]]
            levels = levels + [l for l in a.extra_levels if l not in levels]
            att = human_attempts(C.features_dir(cfg), ho.rows, levels)
            human = {"levels": levels, "attempts": att}
            # pooled held-out human samples for the C2ST
            z_h = ho.norm(ho.z[ho.bc_idx]).mean(1).cpu().numpy()
            lv_of = np.array([ho.rows[p]["level"] for p in ho.play[ho.bc_idx].numpy()])
            keep = np.isin(lv_of, levels)
            human["c2st"] = (z_h[keep], ho.prev[ho.bc_idx].numpy()[keep], ho.a[ho.bc_idx].numpy()[keep],
                             ho.play[ho.bc_idx].numpy()[keep])
            results["human_heldout"] = {"levels": levels, "all": summarise(att),
                                        "per_level": {l: summarise([e for e in att if e["level"] == l])
                                                      for l in levels if any(e["level"] == l for e in att)}}
            print("human", json.dumps({k: v for k, v in results["human_heldout"]["all"].items() if k != "hist"}), flush=True)
        levels = human["levels"]
        res = {"checkpoint": path, "extra": ck.get("extra", {})}
        res["offline_heldout_actions"] = {k: v for k, v in action_metrics(agent, ho).items() if k != "confusion"}
        res["offline_heldout_jepa"] = jepa_metrics(agent, ho)
        rec = os.path.join(a.out, name, "bk2")
        eps = play(agent, ck["stats"], enc, levels, a.episodes, a.n_envs, integ, a.max_steps, a.seed, rec)
        res["agent"] = {"all": summarise(eps), "per_level": {l: summarise([e for e in eps if e["level"] == l]) for l in levels}}
        hh = results["human_heldout"]["all"]["hist"]
        res["agent"]["js_action_vs_human"] = js(res["agent"]["all"]["hist"], np.asarray(hh))
        res["agent"]["js_action_vs_human_ci95"] = js_ci(eps, human["attempts"])
        za = np.concatenate([e["z"] for e in eps if len(e["acts"])]).astype(np.float32)
        pa = np.concatenate([e["prev"] for e in eps if len(e["acts"])])
        aa = np.concatenate([e["acts"] for e in eps if len(e["acts"])])
        ga = np.concatenate([np.full(len(e["acts"]), i) for i, e in enumerate(eps) if len(e["acts"])])
        res["c2st"] = c2st(za, pa, aa, ga, *human["c2st"])
        # trajectories: one video per level + compact npz of all episodes
        vids = {}
        for e in eps:
            if e["frames"] is not None and e["level"] not in vids:
                p = os.path.join(a.out, name, f"{e['level']}_episode.mp4")
                vids[e["level"]] = {"file": p, "codec": save_video(e["frames"], p), "dx": e["dx"],
                                    "died": e["died"], "completed": e["completed"], "steps": e["steps"]}
        res["videos"] = vids
        np.savez_compressed(os.path.join(a.out, name, "episodes.npz"),
                            **{f"ep{i:03d}_{k}": np.asarray(e[k]) for i, e in enumerate(eps)
                               for k in ("all_acts", "x_traj")},
                            meta=json.dumps([{k: e[k] for k in ("level", "steps", "dx", "died", "completed", "truncated",
                                                                 "bk2_dir")} for e in eps]))
        results[name] = res
        print(name, json.dumps({"agent": {k: v for k, v in res["agent"]["all"].items() if k != "hist"},
                                "js": res["agent"]["js_action_vs_human"], "c2st": res["c2st"],
                                "acc": res["offline_heldout_actions"]["accuracy"]}), flush=True)
        C.dump_json(os.path.join(a.out, "eval.json"), _jsonable(results))
    C.dump_json(os.path.join(a.out, "eval.json"), _jsonable(results))


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


if __name__ == "__main__":
    main()
