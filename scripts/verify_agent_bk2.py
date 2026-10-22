"""Check that saved agent .bk2 recordings replay to the same trajectory the agent played.

For each episode in <eval_dir>/<model>/episodes.npz that has a recording folder, replay its
.bk2 through the emulator and compare Mario's x position after every decision step with
the x trajectory logged during play. Frame layout: the environment applies one NOOP frame
at reset, then 4 frames per decision step, so step t ends on frame 4t + 4.

python scripts/verify_agent_bk2.py <eval_dir>/<model> [--max 10]
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mjg import emulator as E  # noqa: E402
from mjg.actions import FRAME_SKIP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--integration", default=os.environ.get("MJG_INTEGRATION"))
    ap.add_argument("--max", type=int, default=10)
    ap.add_argument("--out")
    a = ap.parse_args()
    z = np.load(os.path.join(a.model_dir, "episodes.npz"))
    meta = json.loads(str(z["meta"]))
    rows = []
    for i, m in enumerate(meta):
        if not m.get("bk2_dir") or len(rows) >= a.max:
            continue
        bk2 = sorted(glob.glob(os.path.join(m["bk2_dir"], "*.bk2")))
        if not bk2:
            continue
        rep = E.replay_bk2(bk2[0], a.integration, keep_frame=lambda k: False)
        x = rep["ram"]["player_x_posHi"].astype(int) * 256 + rep["ram"]["player_x_posLo"].astype(int)
        xt = z[f"ep{i:03d}_x_traj"]
        idx = FRAME_SKIP * np.arange(len(xt)) + FRAME_SKIP  # frame index ending each step
        ok = idx < len(x)
        match = float((x[idx[ok]] == xt[ok]).mean()) if ok.any() else 0.0
        rows.append({"episode": i, "level": m["level"], "bk2": bk2[0], "n_steps": int(len(xt)),
                     "replay_frames": int(len(x)), "steps_compared": int(ok.sum()), "x_match_frac": match})
        print(json.dumps(rows[-1]), flush=True)
    summary = {"n_checked": len(rows), "all_exact": all(r["x_match_frac"] == 1.0 for r in rows), "episodes": rows}
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
