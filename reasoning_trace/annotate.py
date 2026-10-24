"""Stage 1 (CPU, no VLM): key frames, facts, decisions and simulated counterfactuals per play.

Writes one JSON line per key frame to <out>/annotations/<play_id>.jsonl, the screens the VLM
will see to <out>/images/<play_id>/, and <out>/summary_<shard>.json with counts and the
determinism check. --debug-png N also draws the RAM objects and the metatile grid (solid
tiles) on the first N key frames, to verify facts.terrain_ahead visually.

python -m reasoning_trace.annotate --subjects sub-01 --out $MJG_ROOT/reasoning_trace/sub-01 \
    --shard 0 --n-shards 8
"""
from __future__ import annotations

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
from reasoning_trace import counterfactuals as C  # noqa: E402
from reasoning_trace import facts as F  # noqa: E402
from reasoning_trace import keyframes as K  # noqa: E402
from reasoning_trace import schema as S  # noqa: E402

RAM_VARS = list(dict.fromkeys(E.RAM_VARS + O.OBJ_RAM_VARS))


def save_png(arr, path, scale=1):
    from PIL import Image
    img = Image.fromarray(arr)
    if scale > 1:
        img = img.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.NEAREST)
    img.save(path)


def debug_png(screen, ram_bytes, r, path, scale=3):
    """Objects (green) and solid metatiles (red outline) on the screen, to check the RAM facts."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(screen).resize((screen.shape[1] * scale, screen.shape[0] * scale), Image.NEAREST)
    g = ImageDraw.Draw(img)
    scroll = r["xscrollHi"] * 256 + r["xscrollLo"]
    for c in range(scroll // 16, scroll // 16 + 17):
        for row in range(F.N_ROWS):
            if F._tile(ram_bytes, c, row) in F.NON_SOLID:
                continue
            x0 = (c * 16 - scroll - O.X_OFF) * scale
            y0 = (32 + row * 16 - O.Y_OFF) * scale
            g.rectangle([x0, y0, x0 + 16 * scale, y0 + 16 * scale], outline=(255, 60, 60), width=1)
    for o in O.oracle_objects(r):
        g.rectangle([o.x0 * scale, o.y0 * scale, o.x1 * scale, o.y1 * scale], outline=(0, 230, 0), width=2)
    img.save(path)


def annotate_play(play, integration, out, n_debug=0):
    rep = E.replay_bk2(play.bk2, integration, keep_frame=lambda k: False, ram_vars=RAM_VARS)
    ram, buttons = rep["ram"], rep["buttons"]
    keys = K.select(K.score_frames(ram, buttons))
    if not keys:
        return [], {"plays": 1, "key_frames": 0}
    sim, final_x = C.replay_with_counterfactuals(play.bk2, integration, [k["frame"] for k in keys], buttons, RAM_VARS)
    img_dir = os.path.join(out, "images", play.play_id)
    os.makedirs(img_dir, exist_ok=True)
    recs, matches = [], []
    for i, kf in enumerate(keys):
        f = kf["frame"]
        if f not in sim:
            continue
        s = sim[f]
        facts = F.scene_facts(ram, f, s["ram_bytes"])
        outs = s["alternatives"]
        best = max([o["dx"] for o in outs.values() if not o["died"]] or [0])
        cf = [{"horizontal": h, "vertical": v, "outcome": o, "risk": S.risk_label(o, best)} for (h, v), o in outs.items()]
        ok = C.human_matches_recording(s["human"], ram, f)
        matches.append(ok)
        images = []
        for j, scr in enumerate(s["screens"]):
            p = os.path.join(img_dir, f"{f:06d}_{j}.png")
            save_png(scr, p)
            images.append(os.path.relpath(p, out))
        if i < n_debug and s["screens"]:
            debug_png(s["screens"][0], s["ram_bytes"], F.ram_at(ram, f - 1), os.path.join(img_dir, f"{f:06d}_debug.png"))
        recs.append({"play_id": play.play_id, "sub": play.sub, "level": play.level, "frame": f, "step": kf["step"],
                     "keyframe_score": kf["score"], "keyframe_reasons": kf["reasons"], "facts": facts,
                     "decision": S.decision_from_buttons(buttons, f), "human_outcome": s["human"],
                     "human_replay_matches": ok, "counterfactuals": cf, "images": images})
    rec_x = F.mario_level_x(F.ram_at(ram, len(buttons) - 1))
    with open(os.path.join(out, "annotations", play.play_id + ".jsonl"), "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return recs, {"plays": 1, "key_frames": len(recs), "human_replay_matches": int(sum(matches)),
                  "resumed_replay_ends_where_recording_ends": int(final_x == rec_x)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MJG_DATA"))
    ap.add_argument("--integration", default=os.environ.get("MJG_INTEGRATION"))
    ap.add_argument("--subjects", nargs="*", default=["sub-01"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--max-plays", type=int, default=None)
    ap.add_argument("--debug-png", type=int, default=0, help="debug overlays for the first N key frames of each play")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "annotations"), exist_ok=True)
    plays = D.index_plays(a.data, a.subjects)[a.shard::a.n_shards][:a.max_plays]
    total, risks = {}, {"safe": 0, "suboptimal": 0, "unsafe": 0}
    for i, p in enumerate(plays):
        if os.path.exists(os.path.join(a.out, "annotations", p.play_id + ".jsonl")):
            continue
        recs, st = annotate_play(p, a.integration, a.out, a.debug_png)
        for k, v in st.items():
            total[k] = total.get(k, 0) + v
        for r in recs:
            for c in r["counterfactuals"]:
                risks[c["risk"]] += 1
        print(f"[{i + 1}/{len(plays)}] {p.play_id}: {st}", flush=True)
    total["counterfactual_risks"] = risks
    with open(os.path.join(a.out, f"summary_{a.shard}.json"), "w") as fh:
        json.dump(total, fh, indent=1)
    print(json.dumps(total, indent=1))


if __name__ == "__main__":
    main()
