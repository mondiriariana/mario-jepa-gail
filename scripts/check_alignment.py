"""Phase-1 check: do emulator replays of human .bk2 files reproduce the recorded videos,
and how do controller inputs line up with video frames and with the annotation TSVs?

Usage: python scripts/check_alignment.py --bk2 <file.bk2> [--mp4 <file.mp4>] [--events <tsv>]
Prints the best (crop, frame offset) match and the per-offset pixel error, so the frame
convention in mjg/emulator.py can be confirmed rather than assumed.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mjg import emulator  # noqa: E402


def decode_mp4(path, max_frames=None):
    import av
    out = []
    with av.open(path) as c:
        for i, f in enumerate(c.decode(video=0)):
            out.append(f.to_ndarray(format="rgb24"))
            if max_frames and i + 1 >= max_frames:
                break
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bk2", required=True)
    ap.add_argument("--mp4")
    ap.add_argument("--events")
    ap.add_argument("--integration", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()

    names, log = emulator.read_bk2_inputs(a.bk2)
    rep = emulator.replay_bk2(a.bk2, a.integration)
    res = {"bk2": a.bk2, "n_input_lines": int(len(log)), "log_buttons": names,
           "env_buttons": rep["button_names"], "n_replay_frames": int(len(rep["buttons"]))}
    # map header-order log to env-order buttons and check they agree
    env_names = rep["button_names"]
    agree = []
    for j, n in enumerate(names):
        key = n.upper()
        if key in env_names:
            m = min(len(log), len(rep["buttons"]))
            # the log can carry one extra line; test offsets 0 and 1 between log and replay
            agree.append({o: float((log[o:o + m - 1, j] == rep["buttons"][:m - 1, env_names.index(key)]).mean())
                          for o in (0, 1)})
    res["log_vs_replay_button_agreement"] = agree
    full = rep["frames"]
    res["screen_shape"] = list(full.shape[1:])

    if a.mp4:
        vid = decode_mp4(a.mp4)
        res["n_video_frames"] = int(len(vid))
        res["video_shape"] = list(vid.shape[1:])
        H, W = vid.shape[1:3]
        sel = np.linspace(200, min(len(vid), len(full)) - 10, 40).astype(int)
        best = None
        table = {}
        for y0 in range(0, full.shape[1] - H + 1, 2):
            for x0 in range(0, full.shape[2] - W + 1, 2):
                for d in range(-3, 4):
                    s = sel[(sel + d >= 0) & (sel + d < len(full))]
                    err = float(np.abs(vid[s].astype(np.float32)
                                       - full[s + d, y0:y0 + H, x0:x0 + W].astype(np.float32)).mean())
                    table[(y0, x0, d)] = err
                    if best is None or err < best[0]:
                        best = (err, y0, x0, d)
        res["best_match"] = {"mean_abs_err": best[0], "crop_y0": best[1], "crop_x0": best[2],
                             "video_frame_i_equals_screen_i_plus": best[3]}
        res["err_by_offset_at_best_crop"] = {d: table[(best[1], best[2], d)] for d in range(-3, 4)}

    if a.events:
        import pandas as pd
        ev = pd.read_csv(a.events, sep="\t")
        stem = Path(a.bk2).name
        start = ev.index[ev["stim_file"].astype(str).str.endswith(stem)]
        if len(start):
            i0 = start[0]
            nxt = ev.index[(ev["trial_type"] == "gym-retro_game") & (ev.index > i0)]
            block = ev.loc[i0 + 1:(nxt[0] - 1 if len(nxt) else ev.index[-1])]
            out = {}
            for tt, btn in [("RIGHT", "RIGHT"), ("LEFT", "LEFT"), ("A", "A"), ("B", "B")]:
                rows = block[block["trial_type"] == tt]
                col = rep["buttons"][:, env_names.index(btn)]
                hits = [float(col[int(r.frame_start):int(r.frame_stop)].mean()) for r in rows.itertuples()]
                onset_ok = [bool(col[int(r.frame_start)]) for r in rows.itertuples()]
                out[tt] = {"n_events": len(rows), "mean_frac_pressed_inside": float(np.mean(hits)) if hits else None,
                           "frac_pressed_at_frame_start": float(np.mean(onset_ok)) if onset_ok else None}
            res["annotation_check"] = out
    print(json.dumps(res, indent=1, default=str))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
