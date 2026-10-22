"""Smoke test of the parallel environment: random actions for a few hundred steps,
prints step rate, episode endings and RAM progress."""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mjg.vec_env import MarioVecEnv  # noqa: E402

if __name__ == "__main__":
    env = MarioVecEnv(4, os.environ["MJG_INTEGRATION"], ["w1l1", "w1l2"], max_steps=200)
    clips, prev = env.reset()
    print("clip batch", clips.shape, clips.dtype, flush=True)
    rng = np.random.default_rng(0)
    t0 = time.time()
    ends = []
    for s in range(400):
        a = rng.choice([1, 2, 3, 4], size=4)  # mostly right-moving actions
        clips, prev, done, info = env.step(a)
        for i in np.nonzero(done)[0]:
            ends.append({k: info[i][k] for k in ("level", "t", "x", "died", "completed", "truncated")})
    dt = time.time() - t0
    print(f"{400*4/dt:.0f} env-steps/s; episodes ended: {len(ends)}", flush=True)
    for e in ends[:10]:
        print(e)
    print("last info", info[0])
    env.close()
