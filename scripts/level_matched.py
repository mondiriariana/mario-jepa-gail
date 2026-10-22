"""Level-matched agent-vs-human comparison from an eval.json.

Agents play the same number of episodes on every evaluation level, while held-out human
attempts are spread unevenly over levels. Pooled statistics would therefore compare
different level mixes. Here every agent statistic is the per-level value weighted by the
human's number of attempts on that level, and the action-distribution JS divergence uses
per-level action histograms reweighted the same way.

python scripts/level_matched.py results/eval_final/eval.json [more eval.json ...] --out results/level_matched.json
"""
import argparse
import json

import numpy as np

KEYS = ["mean_dx", "mean_speed_px_s", "death_rate", "jumps_per_s", "switch_rate", "run_frac", "right_frac",
        "left_frac", "noop_frac", "mean_hold_steps"]


def js(p, q):
    """Jensen-Shannon divergence in bits (same definition as scripts/evaluate.py)."""
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log2(np.maximum(a, 1e-12) / np.maximum(b, 1e-12)), 0))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def matched(per_level, w):
    out = {k: float(sum(w[l] * per_level[l][k] for l in w)) for k in KEYS}
    hist = sum(w[l] * np.asarray(per_level[l]["hist"], float) / max(np.sum(per_level[l]["hist"]), 1) for l in w)
    return out, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evals", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for path in a.evals:
        with open(path) as f:
            ev = json.load(f)
        hp = ev["human_heldout"]["per_level"]
        n = {l: hp[l]["n_attempts"] for l in hp}
        w = {l: n[l] / sum(n.values()) for l in n}
        if "Human" not in res:
            h, hh = matched(hp, w)
            res["Human"] = {**h, "level_weights": w}
        _, hh = matched(hp, w)
        for m, r in ev.items():
            if m == "human_heldout" or m in res:
                continue
            s, ah = matched(r["agent"]["per_level"], w)
            s["js_action_vs_human"] = js(ah, hh)
            res[m] = s
    for m, s in res.items():
        print(m, {k: round(v, 3) for k, v in s.items() if isinstance(v, float)})
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
