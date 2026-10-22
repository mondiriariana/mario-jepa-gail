"""Summarise the encoded demo set: plays, steps, split, levels, action-space coverage."""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mjg import data as D  # noqa: E402
from mjg.actions import NAMES  # noqa: E402
from mjg.demos import split_plays  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True)
ap.add_argument("--heldout-every", type=int, default=6)
ap.add_argument("--out")
a = ap.parse_args()
rows = D.read_index(a.features)
train, val, held = split_plays(rows, a.heldout_every)
acts, exact, hold = [], [], []
for r in rows:
    m = np.load(os.path.join(a.features, r["play_id"] + ".meta.npz"))
    c = m["control"]
    acts.append(m["actions"][c]); exact.append(m["exact"][c]); hold.append(m["hold"][c])
acts, exact, hold = map(np.concatenate, (acts, exact, hold))
hist = np.bincount(acts, minlength=len(NAMES))
out = {
    "n_plays": len(rows), "n_train_plays": len(train), "n_val_plays": len(val), "n_heldout_plays": len(held),
    "val_sessions": sorted({r["ses"] for r in val}),
    "val_control_steps": int(sum(r["n_control"] for r in val)),
    "heldout_sessions": sorted({r["ses"] for r in held}),
    "train_sessions": sorted({r["ses"] for r in train}),
    "n_steps": int(sum(r["n_steps"] for r in rows)), "n_control_steps": int(sum(r["n_control"] for r in rows)),
    "train_control_steps": int(sum(r["n_control"] for r in train)),
    "heldout_control_steps": int(sum(r["n_control"] for r in held)),
    "hours_of_play": sum(r["n_steps"] for r in rows) * 4 / 60 / 3600,
    "levels": dict(sorted(Counter(r["level"] for r in rows).items())),
    "plays_with_level_completed": int(sum(r["completed"] for r in rows)),
    "action_projection_exact_frac": float(exact.mean()),
    "step_hold_consistency": float(hold.mean()),
    "steps_with_all_4_frames_same_combo": float((hold == 1).mean()),
    "action_hist": dict(zip(NAMES, hist.tolist())),
    "action_frac": dict(zip(NAMES, np.round(hist / hist.sum(), 4).tolist())),
}
print(json.dumps(out, indent=1))
if a.out:
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
