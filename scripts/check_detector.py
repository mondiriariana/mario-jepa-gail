"""Stage 0 of the SAM scenario: does a text-prompted detector (SAM 3, or Grounding DINO as a
stand-in) find Mario and enemies on NES frames?

Samples screens from human replays, runs the SAM detector, and scores it against the
exact object boxes read from emulator RAM (the oracle): recall and precision per object
type (a match = same type and centres within --max-dist px) and centre error. Also scored with
the enemy types merged and by position only (any label), with a true-type x predicted-label
confusion table, so mislabelled-but-found objects are visible. --debug-dir saves annotated screens.

python scripts/check_detector.py --n 200 --out results/detector_check.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mjg import data as D  # noqa: E402
from mjg import emulator as E  # noqa: E402
from mjg import objects as O  # noqa: E402

ENEMIES = (O.TYPE_GOOMBA, O.TYPE_KOOPA, O.TYPE_ENEMY)


def match(truth, pred, same, max_dist):
    """Greedy one-to-one matching: each true object takes the nearest unused prediction within
    max_dist whose label passes same(true_type, pred_type). Yields (true obj, pred index, dist)."""
    used = set()
    for t in truth:
        best, bd = None, max_dist
        for j, q in enumerate(pred):
            d = np.hypot(q.center[0] - t.center[0], q.center[1] - t.center[1])
            if j not in used and same(t.type, q.type) and d <= bd:
                best, bd = j, d
        if best is not None:
            used.add(best)
            yield t, best, bd


def draw(frame, truth, pred, path, scale=3):
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame).resize((frame.shape[1] * scale, frame.shape[0] * scale), Image.NEAREST)
    g = ImageDraw.Draw(img)
    for o, colour, above in [(t, (0, 230, 0), False) for t in truth] + [(q, (255, 40, 40), True) for q in pred]:
        box = [o.x0 * scale, o.y0 * scale, o.x1 * scale, o.y1 * scale]
        g.rectangle(box, outline=colour, width=2)
        label = O.TYPE_NAMES[o.type] + ("" if not above else f" {o.score:.2f}")
        g.text((box[0] + 2, box[1] - 11 if above else box[3] + 1), label, fill=colour)
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MJG_DATA"))
    ap.add_argument("--integration", default=os.environ.get("MJG_INTEGRATION"))
    ap.add_argument("--subjects", nargs="*", default=["sub-01"])
    ap.add_argument("--n", type=int, default=200, help="screens to score")
    ap.add_argument("--plays", type=int, default=20, help="plays to sample screens from")
    ap.add_argument("--detector", choices=["sam", "gdino"], default="sam")
    ap.add_argument("--sam-model", default=None, help="detector weights (default: $MJG_SAM_MODEL / $MJG_GDINO_MODEL)")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--upscale", type=int, default=4)
    ap.add_argument("--max-dist", type=float, default=12.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug-dir", help="save the first --n-debug screens with true (green) and detected (red) boxes")
    ap.add_argument("--n-debug", type=int, default=8)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    plays = D.index_plays(a.data, a.subjects)
    plays = [plays[i] for i in rng.choice(len(plays), min(a.plays, len(plays)), replace=False)]
    sam = O.SamDetector(a.sam_model, threshold=a.threshold, upscale=a.upscale) if a.detector == "sam" \
        else O.GroundingDinoDetector(a.sam_model, threshold=a.threshold, upscale=a.upscale)
    per_play = int(np.ceil(a.n / len(plays)))
    # three ways to count a hit: exact type, enemy types merged, or position only (any label)
    modes = {"per_type": lambda t, q: q == t,
             "enemies_merged": lambda t, q: q == t or (t in ENEMIES and q in ENEMIES),
             "any_label": lambda t, q: True}
    tp = {m: np.zeros(O.N_TYPES) for m in modes}
    errs = {m: [[] for _ in range(O.N_TYPES)] for m in modes}
    pred_hit = np.zeros(O.N_TYPES)  # predictions matched to some object (any label), by predicted type
    confusion = np.zeros((O.N_TYPES, O.N_TYPES), int)  # true type x predicted label, position-only matches
    n_true = np.zeros(O.N_TYPES); n_pred = np.zeros(O.N_TYPES)
    n_screens = 0
    if a.debug_dir:
        os.makedirs(a.debug_dir, exist_ok=True)
    for p in plays:
        rep = E.replay_bk2(p.bk2, a.integration, ram_vars=list(dict.fromkeys(E.RAM_VARS + O.OBJ_RAM_VARS)))
        ram = rep["ram"]
        frames_with_objs = [k for k in range(len(rep["frames"]))
                            if O.oracle_objects({v: ram[v][k] for v in ram})]
        for k in rng.choice(frames_with_objs, min(per_play, len(frames_with_objs)), replace=False):
            truth = O.oracle_objects({v: ram[v][k] for v in ram})
            pred = sam(rep["frames"][k])
            n_screens += 1
            for t in truth:
                n_true[t.type] += 1
            for q in pred:
                n_pred[q.type] += 1
            for m, same in modes.items():
                for t, j, d in match(truth, pred, same, a.max_dist):
                    tp[m][t.type] += 1; errs[m][t.type].append(d)
                    if m == "any_label":
                        confusion[t.type, pred[j].type] += 1; pred_hit[pred[j].type] += 1
            if a.debug_dir and n_screens <= a.n_debug:
                draw(rep["frames"][k], truth, pred, os.path.join(a.debug_dir, f"screen_{n_screens:02d}_{p.play_id}_{k}.png"))
    res = {"n_screens": n_screens, "detector": a.detector, "model": a.sam_model,
           "threshold": a.threshold, "upscale": a.upscale, "max_dist_px": a.max_dist}
    for m in modes:
        res[m] = {}
        for t, name in enumerate(O.TYPE_NAMES):
            row = {"true": int(n_true[t]), "recall": float(tp[m][t] / n_true[t]) if n_true[t] else None,
                   "median_centre_error_px": float(np.median(errs[m][t])) if errs[m][t] else None}
            if m == "per_type":
                row.update(predicted=int(n_pred[t]), precision=float(tp[m][t] / n_pred[t]) if n_pred[t] else None)
            res[m][name] = row
    res["any_label_precision_by_predicted_label"] = {
        name: float(pred_hit[t] / n_pred[t]) if n_pred[t] else None for t, name in enumerate(O.TYPE_NAMES)}
    res["confusion_true_x_predicted"] = {
        "labels": O.TYPE_NAMES, "rows_true_cols_predicted": confusion.tolist()}
    print(json.dumps(res, indent=1))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
