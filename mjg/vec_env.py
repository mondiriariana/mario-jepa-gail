"""Parallel Super Mario Bros environments for agent rollouts.

stable-retro allows one emulator per process, so each environment lives in its own
subprocess. A step holds one combo from mjg.actions for FRAME_SKIP emulator frames and
returns the two odd-offset screens the next clip needs (same layout as mjg.data), the
RAM summary of the last frame and termination flags.

Episode = one attempt at a level from its CNeuroMod start state, ending on level
completion, the first life lost, or ``max_steps`` decision steps.
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

from . import actions as A
from . import data as D
from . import emulator as E

KEEP = ("player_state", "oper_mode", "lives", "player_y_screen", "score", "coins", "time",
        "player_x_posHi", "player_x_posLo", "player_x_speed", "player_y_pos", "stage", "world",
        "jump_airborne", "powerstate")


def _ram(info):
    r = {k: int(info.get(k, 0)) for k in KEEP}
    r["x"] = r["player_x_posHi"] * 256 + r["player_x_posLo"]
    return r


def _worker(remote, integration, record_dir):
    import retro
    E.add_integration(integration)
    env = None
    lives0 = 0
    n_ep = 0
    rec = None
    while True:
        cmd, arg = remote.recv()
        if cmd == "reset":
            if env is not None:
                env.close()
            kw = {}
            if record_dir:
                # one folder per episode: stable-retro numbers recordings per emulator instance,
                # and a new instance is made at every reset, so a shared folder would overwrite
                rec = os.path.join(record_dir, f"ep{n_ep:04d}")
                os.makedirs(rec, exist_ok=True)
                kw["record"] = rec
                n_ep += 1
            env = retro.make(game=E.GAME, state=arg, inttype=retro.data.Integrations.CUSTOM_ONLY,
                             use_restricted_actions=retro.Actions.ALL, render_mode=None, **kw)
            obs, info = env.reset()
            # one NOOP frame to populate RAM info (the reset dict is empty)
            obs, _, _, _, info = env.step(np.zeros(env.num_buttons, np.uint8))
            lives0 = int(info.get("lives", 0))
            r = _ram(info)
            r["bk2_dir"] = rec
            remote.send((obs, r))
        elif cmd == "step":
            btn = A.ACTION_BUTTONS[arg]
            frames = []
            term = False
            for i in range(A.FRAME_SKIP):
                obs, _, t1, t2, info = env.step(btn)
                if i % 2 == 1:
                    frames.append(obs)
                term |= bool(t1 or t2)
            r = _ram(info)
            died = r["lives"] < lives0 or r["player_state"] in (6, 11) or r["player_y_screen"] >= 2
            done_level = r["player_state"] in (4, 5) or term and not died
            r["died"], r["completed"] = bool(died), bool(done_level and not died)
            remote.send((np.stack(frames), r))
        elif cmd == "close":
            if env is not None:
                env.close()
            remote.close()
            return


class MarioVecEnv:
    def __init__(self, n, integration, levels, level_probs=None, max_steps=1500, seed=0,
                 record_dir=None, fixed_levels=None):
        ctx = mp.get_context("spawn")
        self.n, self.levels, self.max_steps = n, list(levels), max_steps
        self.p = None if level_probs is None else np.asarray(level_probs) / np.sum(level_probs)
        self.rng = np.random.default_rng(seed)
        self.fixed_levels = fixed_levels  # optional per-env level, kept across auto-resets
        self.remotes, self.procs = [], []
        for i in range(n):
            a, b = ctx.Pipe()
            rd = os.path.join(record_dir, f"env{i:02d}") if record_dir else None
            p = ctx.Process(target=_worker, args=(b, integration, rd), daemon=True)
            p.start()
            self.remotes.append(a)
            self.procs.append(p)
        self.clips = np.zeros((n, D.CLIP_LEN) + E.SCREEN_SHAPE, np.uint8)
        self.t = np.zeros(n, int)
        self.prev = np.full(n, A.N_ACTIONS, int)
        self.level = [None] * n
        self.ram = [None] * n
        self.ep_x0 = np.zeros(n, int)
        self.control = np.zeros(n, bool)  # is the state at the next decision controllable?
        self.bk2_dir = [None] * n  # recording folder of each env's current episode

    @staticmethod
    def state_name(level):  # "w1l1" -> "Level1-1"
        return f"Level{level[1]}-{level[3]}"

    def _reset(self, i, level=None):
        if level is None and self.fixed_levels is not None:
            level = self.fixed_levels[i]
        lv = level or self.levels[self.rng.choice(len(self.levels), p=self.p)]
        self.remotes[i].send(("reset", self.state_name(lv)))
        obs, ram = self.remotes[i].recv()
        self.clips[i] = obs[None]
        self.t[i], self.prev[i], self.level[i], self.ram[i] = 0, A.N_ACTIONS, lv, ram
        self.ep_x0[i] = ram["x"]
        self.control[i] = ram["player_state"] in D.CONTROL_STATES and ram["oper_mode"] == 1
        self.bk2_dir[i] = ram.get("bk2_dir")

    def reset(self, levels=None):
        for i in range(self.n):
            self._reset(i, None if levels is None else levels[i])
        return self.clips.copy(), self.prev.copy()

    def step(self, act):
        for i, a in enumerate(act):
            self.remotes[i].send(("step", int(a)))
        infos, dones = [], np.zeros(self.n, bool)
        for i in range(self.n):
            frames, ram = self.remotes[i].recv()
            self.clips[i] = np.concatenate([self.clips[i, len(frames):], frames])
            self.t[i] += 1
            self.prev[i] = act[i]
            ram["level"], ram["t"] = self.level[i], int(self.t[i])
            ram["control"] = ram["player_state"] in D.CONTROL_STATES and ram["oper_mode"] == 1
            trunc = self.t[i] >= self.max_steps
            done = ram["died"] or ram["completed"] or trunc
            ram["truncated"] = bool(trunc and not (ram["died"] or ram["completed"]))
            ram["dx"] = ram["x"] - int(self.ep_x0[i])
            ram["bk2_dir"] = self.bk2_dir[i]
            infos.append(ram)
            dones[i] = done
            self.control[i] = ram["control"]
            if done:
                self._reset(i)
        return self.clips.copy(), self.prev.copy(), dones, infos

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
