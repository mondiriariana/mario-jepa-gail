"""Figures for PRELIMINARY_RESULTS.md from the training logs and eval.json.

python scripts/make_figures.py --out results/figures --eval results/eval/eval.json \
    --bc BC=runs/bc BC+JEPA=runs/bcjepa --gail GAIL=runs/gail GAIL+JEPA=runs/gailjepa
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mjg.actions import NAMES  # noqa: E402

# Fixed entity colours (reference categorical palette, light mode); human = neutral ink.
COL = {"BC": "#2a78d6", "BC+JEPA": "#eb6834", "GAIL": "#1baf7a", "GAIL+JEPA": "#6250d6",
       "GAIL (BC reg.)": "#e87ba4", "GAIL+JEPA (BC reg.)": "#008300", "Human": "#52514e"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "lines.linewidth": 2, "figure.dpi": 150, "savefig.bbox": "tight"})


def jl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def smooth(y, k=5):
    y = np.asarray(y, float)
    if len(y) < k:
        return y
    return np.convolve(np.pad(y, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def series(rows, key):
    xs = [r for r in rows if key in r and r[key] is not None]
    return [r.get("step", r.get("it")) for r in xs], [r[key] for r in xs]


def bc_curves(runs, out):
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
    for name, d in runs.items():
        rows = jl(os.path.join(d, "bc_log.jsonl"))
        x, y = series(rows, "l_bc"); ax[0].plot(x, smooth(y), color=COL[name], label=name)
        x, y = series(rows, "val_acc"); ax[1].plot(x, y, color=COL[name], marker="o", ms=3)
        x, y = series(rows, "val_jepa_r2")
        if x:
            ax[2].plot(x, y, color=COL[name], marker="o", ms=3)
            for k, lab in (("val_copy_r2", "copy z_t"), ("val_shuf_r2", "shuffled actions")):
                xc, yc = series(rows, k)
                ax[2].plot(xc, yc, color=COL[name], lw=1, ls="--" if k == "val_copy_r2" else ":")
                ax[2].text(xc[-1], yc[-1], lab, fontsize=7, color=MUTED, ha="right", va="bottom")
    ax[0].set(title="BC loss (train)", xlabel="update")
    ax[1].set(title="Validation action accuracy", xlabel="update")
    ax[2].set(title="Validation latent prediction R² (K=4)", xlabel="update")
    ax[0].legend()
    fig.savefig(os.path.join(out, "bc_curves.png")); plt.close(fig)


def gail_curves(runs, out, fname="gail_curves.png"):
    keys = [("d_acc_human", "Discriminator acc. on human"), ("d_acc_agent", "Discriminator acc. on agent"),
            ("reward_mean", "GAIL reward per controllable step"), ("entropy", "Policy entropy (nats)"),
            ("ep_dx", "Episode progress Δx (px)"), ("ep_died", "Episodes ending in death"),
            ("val_acc", "Validation human action accuracy"), ("jepa_agent", "JEPA loss on agent rollouts"),
            ("jepa_human", "JEPA loss on human demos")]
    fig, axs = plt.subplots(3, 3, figsize=(11, 8))
    for (k, title), ax in zip(keys, axs.flat):
        for name, d in runs.items():
            rows = jl(os.path.join(d, "gail_log.jsonl"))
            x, y = series(rows, k)
            if not x or (k.startswith("jepa") and not any(y)):
                continue
            ax.plot(x, y if k == "val_acc" else smooth(y), color=COL[name], label=name,
                    marker="o" if k == "val_acc" else None, ms=3)
        ax.set(title=title, xlabel="iteration (2048 env steps)")
        if k in ("ep_died", "d_acc_human", "d_acc_agent"):
            ax.set_ylim(0, 1.03)
    axs[0, 0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, fname)); plt.close(fig)


def eval_bars(ev, out):
    models = [m for m in ev if m != "human_heldout"]
    hum = ev["human_heldout"]["all"]
    metrics = [("c2st", "full", "Classifier 2-sample test acc.\n(0.5 = indistinguishable; hollow: actions only)"),
               ("agent", "js", "Action-distribution JS div.\nvs held-out human (bits)"),
               ("agent", "mean_speed_px_s", "Horizontal speed (px/s)"),
               ("agent", "jumps_per_s", "Jumps per second"),
               ("agent", "switch_rate", "Action switch rate per step"),
               ("agent", "death_rate", "Attempts ending in death"),
               ("agent", "mean_dx", "Mean progress per attempt (px)"),
               ("offline", "balanced_accuracy", "Held-out human actions:\nbalanced accuracy (offline)")]
    fig, axs = plt.subplots(2, 4, figsize=(13, 5.6))
    for (grp, k, title), ax in zip(metrics, axs.flat):
        names, vals, errs, extra = [], [], [], []
        for m in models:
            r = ev[m]
            ci = None
            if grp == "c2st":
                v = r["c2st"][k]
                sd = r["c2st"].get(k + "_fold_sd")
                ci = [v - sd, v + sd] if sd is not None else None
                extra.append(r["c2st"]["actions_only"])
            elif k == "js":
                v, ci = r["agent"]["js_action_vs_human"], r["agent"].get("js_action_vs_human_ci95")
            elif grp == "offline":
                v = r["offline_heldout_actions"][k]
            else:
                v, ci = r["agent"]["all"][k], r["agent"]["all"].get("ci95", {}).get(k)
            names.append(m); vals.append(v); errs.append(ci)
        if grp == "agent" and k != "js":
            names.append("Human"); vals.append(hum[k]); errs.append(hum.get("ci95", {}).get(k))
        xs = np.arange(len(names))
        ax.bar(xs, vals, color=[COL.get(n, MUTED) for n in names], width=0.7)
        for x, v, ci in zip(xs, vals, errs):
            if ci:
                ax.errorbar(x, v, yerr=[[v - ci[0]], [ci[1] - v]], color=INK, lw=1, capsize=2)
            top = ci[1] if ci else v
            ax.text(x, top, f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=INK)
        if extra:
            ax.plot(xs[:len(extra)], extra, marker="o", ms=7, mfc="none", mec=INK, ls="none")
        if grp == "c2st":
            ax.axhline(0.5, color=MUTED, lw=1, ls="--")
        ax.set_xticks(xs, names, rotation=30, ha="right")
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "eval_summary.png")); plt.close(fig)

    # action histograms
    fig, ax = plt.subplots(figsize=(11, 3))
    names = models + ["Human"]
    w = 0.8 / len(names)
    for j, n in enumerate(names):
        h = np.asarray(hum["hist"] if n == "Human" else ev[n]["agent"]["all"]["hist"], float)
        ax.bar(np.arange(len(h)) + (j - len(names) / 2 + 0.5) * w, h / h.sum(), width=w, color=COL.get(n, MUTED), label=n)
    ax.set_xticks(np.arange(len(NAMES)), NAMES, rotation=40, ha="right")
    ax.set(ylabel="fraction of controllable steps", title="Action distribution in play (evaluation levels)")
    ax.legend(ncol=len(names))
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(out, "action_hist.png")); plt.close(fig)

    # latent prediction: model vs controls
    rows = [(m, ev[m]["offline_heldout_jepa"]) for m in models if ev[m].get("offline_heldout_jepa")]
    if rows:
        fig, ax = plt.subplots(figsize=(6, 3))
        labels = ["predictor\n(true actions)", "predictor\n(shuffled actions)", "copy z_t"]
        w = 0.8 / len(rows)
        for j, (m, r) in enumerate(rows):
            v = [r["r2"], r["shuf_r2"], r["copy_r2"]]
            xs = np.arange(3) + (j - len(rows) / 2 + 0.5) * w
            ax.bar(xs, v, width=w, color=COL[m], label=m)
            for x, vv in zip(xs, v):
                ax.text(x, vv, f"{vv:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(np.arange(3), labels)
        ax.set(ylabel="R² of z_{t+4} (held-out)", title="Held-out latent prediction, 16 frames ahead")
        ax.legend()
        ax.grid(axis="x", visible=False)
        fig.savefig(os.path.join(out, "latent_prediction.png")); plt.close(fig)


def probe_fig(path, out):
    with open(path) as f:
        P = json.load(f)
    reps = list(P["results"])
    col = lambda n: COL.get(n, MUTED)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.2), gridspec_kw={"width_ratios": [1.2, 1.2, 1.6]})
    for ax, tgt, title in ((axs[0], "dx", "Horizontal movement 16 frames ahead\n(R², state + next 4 actions)"),
                           (axs[1], "die", "Death starts within 2 s\n(AUC, state + next 4 actions)")):
        xs = np.arange(len(reps))
        for x, n in zip(xs, reps):
            r = P["results"][n][tgt]["state+actions"]
            sh = P["results"][n][tgt]["state+shuffled_actions"]["score"]
            c = col(n if n in COL else "Human")
            ax.errorbar(x, r["score"], yerr=[[r["score"] - r["ci95"][0]], [r["ci95"][1] - r["score"]]], fmt="o",
                        ms=8, color=c, ecolor=c, lw=2, capsize=0)
            ax.plot(x, sh, marker="o", ms=7, mfc="none", mec=c, mew=1.5, ls="none")
        lo = min(P["results"][n][tgt]["state+shuffled_actions"]["score"] for n in reps)
        ax.set_ylim(lo - 0.05, None)
        ax.set_xticks(xs, [n.replace(" (frozen)", "") for n in reps], rotation=30, ha="right")
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", visible=False)
    axs[0].text(0.02, 0.03, "filled: true next actions   hollow: shuffled actions", transform=axs[0].transAxes,
                fontsize=7, color=MUTED, va="bottom")
    ax = axs[2]
    tg = ["dx", "dy", "air", "die"]
    pairs = [k for k in ("BC+JEPA - BC", "GAIL+JEPA - GAIL", "GAIL+JEPA (BC reg.) - GAIL (BC reg.)") if k in P["paired"]]
    w = 0.8 / len(pairs)
    for j, k in enumerate(pairs):
        xs = np.arange(len(tg)) + (j - len(pairs) / 2 + 0.5) * w
        d = [P["paired"][k][t]["diff"] for t in tg]
        lo = [P["paired"][k][t]["diff"] - P["paired"][k][t]["ci95"][0] for t in tg]
        hi = [P["paired"][k][t]["ci95"][1] - P["paired"][k][t]["diff"] for t in tg]
        ax.bar(xs, d, width=w, color=col(k.split(" - ")[0]), label=k.replace(" - ", " minus "))
        ax.errorbar(xs, d, yerr=[lo, hi], fmt="none", color=INK, lw=1, capsize=2)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(np.arange(len(tg)), ["Δx (R²)", "Δy (R²)", "airborne (AUC)", "death (AUC)"])
    ax.set_title("Effect of the JEPA objective on the representation\n(paired difference, 95% CI over held-out plays)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "probes.png")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--probes")
    ap.add_argument("--eval")
    ap.add_argument("--bc", nargs="*", default=[])
    ap.add_argument("--gail", nargs="*", default=[])
    ap.add_argument("--gail-fig", default="gail_curves.png")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.bc:
        bc_curves(dict(s.split("=", 1) for s in a.bc), a.out)
    if a.gail:
        gail_curves(dict(s.split("=", 1) for s in a.gail), a.out, a.gail_fig)
    if a.probes:
        probe_fig(a.probes, a.out)
    if a.eval:
        with open(a.eval) as f:
            eval_bars(json.load(f), a.out)


if __name__ == "__main__":
    main()
