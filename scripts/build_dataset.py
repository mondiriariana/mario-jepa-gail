"""Replay human .bk2 files in the emulator and encode every decision step with frozen
V-JEPA 2.

Per play this writes, into --out:
  <play_id>.z.npy      float16 [n_steps, grid*grid, 1024]   pooled V-JEPA 2 latents
  <play_id>.meta.npz   actions, exactness, hold-consistency, control mask, RAM per step
  <play_id>.meta.json  small summary row (n_steps, level, outcome stats)

Replays run in CPU worker processes (one emulator per process, a stable-retro rule);
the GPU process gathers clips and encodes them. Use --shard i --nshards n to split the
plays across several jobs.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mjg import actions as A  # noqa: E402
from mjg import data as D  # noqa: E402
from mjg import emulator as E  # noqa: E402


def replay_worker(args):
    play, integration, tmp = args
    t0 = time.time()
    rep = E.replay_bk2(play.bk2, integration, keep_frame=lambda k: k % 2 == 1)
    n_frames = len(rep["buttons"])
    n_steps = n_frames // A.FRAME_SKIP
    fpath = os.path.join(tmp, play.play_id + ".frames.npy")
    np.save(fpath, rep["frames"])
    f = np.arange(n_steps) * A.FRAME_SKIP
    act, exact = A.buttons_to_action(rep["buttons"][f])
    # does the human hold the same projected combo over all frames of the step?
    per_frame, _ = A.buttons_to_action(rep["buttons"][: n_steps * A.FRAME_SKIP])
    hold = (per_frame.reshape(n_steps, A.FRAME_SKIP) == act[:, None]).mean(1)
    ram_idx = np.clip(f - 1, 0, n_frames - 1)
    ram = {k: v[ram_idx] for k, v in rep["ram"].items()}
    control = np.isin(ram["player_state"], D.CONTROL_STATES) & (ram["oper_mode"] == 1)
    meta = dict(actions=act, exact=exact, hold=hold.astype(np.float32), control=control,
                x=np.array([int(h) * 256 + int(l) for h, l in zip(ram["player_x_posHi"], ram["player_x_posLo"])]),
                **{"ram_" + k: v for k, v in ram.items()})
    return play, fpath, n_steps, meta, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MJG_DATA"))
    ap.add_argument("--integration", default=os.environ.get("MJG_INTEGRATION"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", nargs="*", default=["sub-01"])
    ap.add_argument("--max-plays", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--reverse", action="store_true", help="process this shard back to front (helper job)")
    a = ap.parse_args()

    import torch
    from mjg.encoder import FrozenVJEPA2

    os.makedirs(a.out, exist_ok=True)
    plays = D.index_plays(a.data, a.subjects)
    if a.max_plays:
        plays = plays[: a.max_plays]
    plays = plays[a.shard:: a.nshards]
    plays = [p for p in plays if not os.path.exists(os.path.join(a.out, p.play_id + ".meta.json"))]
    if a.reverse:
        plays = plays[::-1]
    print(f"{len(plays)} plays to process (shard {a.shard}/{a.nshards})", flush=True)
    enc = FrozenVJEPA2(grid=a.grid)
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "/tmp"))
    ctx = mp.get_context("spawn")
    t_start = time.time()
    n_done = n_steps_total = 0
    with ctx.Pool(a.workers, maxtasksperchild=20) as pool:
        for play, fpath, n_steps, meta, t_rep in pool.imap_unordered(
                replay_worker, [(p, a.integration, tmp) for p in plays]):
            t0 = time.time()
            if os.path.exists(os.path.join(a.out, play.play_id + ".meta.json")):  # done by a helper job
                os.remove(fpath)
                continue
            frames = torch.from_numpy(np.load(fpath)).cuda()
            os.remove(fpath)
            pos = torch.from_numpy(D.clip_positions(n_steps, len(frames))).cuda()
            z = np.empty((n_steps, a.grid * a.grid, enc.dim), np.float16)
            for i in range(0, n_steps, a.batch):
                z[i:i + a.batch] = enc(frames[pos[i:i + a.batch]]).cpu().numpy()
            np.save(os.path.join(a.out, play.play_id + ".z.npy"), z)
            np.savez_compressed(os.path.join(a.out, play.play_id + ".meta.npz"), **meta)
            row = dict(play_id=play.play_id, sub=play.sub, ses=play.ses, level=play.level, rep=play.rep,
                       n_steps=int(n_steps), n_control=int(meta["control"].sum()),
                       exact_frac=float(meta["exact"].mean()), hold_frac=float(meta["hold"].mean()),
                       max_x=int(meta["x"].max()), completed=bool((meta["ram_player_state"] == 4).any()))
            with open(os.path.join(a.out, play.play_id + ".meta.json"), "w") as f:
                json.dump(row, f)
            n_done += 1
            n_steps_total += n_steps
            el = time.time() - t_start
            print(f"[{n_done}/{len(plays)}] {play.play_id} steps={n_steps} replay={t_rep:.1f}s "
                  f"encode={time.time()-t0:.1f}s total {n_steps_total/el:.0f} steps/s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
