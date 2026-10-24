"""Simulated counterfactuals: what would each alternative meta-action have done?

nuReasoning has a VLM judge alternatives (76% agreement with experts). Here the emulator
plays them: the human replay is paused at each key frame, the state is saved, every
alternative in schema.ALTERNATIVES is played for schema.ROLLOUT frames from that state, and
the outcome (death, progress, score, power-up, level end) is read from RAM. The replay then
resumes from the saved state, so the human trajectory is unchanged.

Determinism check: the human's own buttons are also replayed from the saved state; their
outcome must equal what actually happened in the recording (``human_replay_matches``).
"""
from __future__ import annotations

import numpy as np

from mjg import emulator as E

from . import facts as F
from . import schema as S

DEATH_STATES = (6, 11)
END_STATES = (4, 5)


def _outcome(env, program: np.ndarray, start: dict) -> dict:
    lives0, x0, score0 = start["lives"], F.mario_level_x(start), start["score"]
    p0 = start.get("powerstate", 0)
    died_at = done_at = None
    info, n = start, 0
    for keys in program:
        _, _, _, _, info = env.step(keys)
        n += 1
        if info["lives"] < lives0 or info["player_state"] in DEATH_STATES or info["player_y_screen"] >= 2:
            died_at = n
            break
        if info["player_state"] in END_STATES:
            done_at = n
            break
    return {"died": died_at is not None, "died_after_frames": died_at, "completed_level": done_at is not None,
            "dx": int(F.mario_level_x(info) - x0), "score_gain": int(info["score"] - score0),
            "powered_up": int(info.get("powerstate", 0)) > p0, "frames": n}


def replay_with_counterfactuals(bk2: str, integration: str, key_frames: list[int], buttons: np.ndarray,
                                ram_vars: list[str], past=(1, 5, 9, 13)):
    """Second pass over a replay. At each key frame k (state after frame k-1) returns:

    ``ram_bytes``: full 2 KB RAM; ``screens``: screens k-1, k-5, ... (oldest last);
    ``alternatives``: {(horizontal, vertical): outcome}; ``human``: outcome of the human's
    own next ROLLOUT frames replayed from the saved state.
    Also returns ``final_x`` of the resumed replay for the end-to-end determinism check.
    """
    retro = E.add_integration(integration)
    movie = retro.Movie(bk2)
    movie.step()
    env = retro.make(game=movie.get_game(), state=retro.State.NONE,
                     inttype=retro.data.Integrations.CUSTOM_ONLY,
                     use_restricted_actions=retro.Actions.ALL, players=movie.players, render_mode=None)
    keys_set, out = set(key_frames), {}
    ring = {}
    info = None
    try:
        env.initial_state = movie.get_state()
        env.reset()
        nb = env.num_buttons
        k = 0
        while movie.step():
            keys = [movie.get_key(i, 0) for i in range(nb)]
            if k in keys_set and info is not None:
                snap = env.em.get_state()
                start = {v: int(info.get(v, 0)) for v in ram_vars}
                rec = {"ram_bytes": np.array(env.get_ram(), dtype=np.uint8),
                       "screens": [ring[k - d] for d in past if (k - d) in ring], "alternatives": {}}
                for alt, prog in S.PROGRAMS.items():
                    env.em.set_state(snap)
                    rec["alternatives"][alt] = _outcome(env, prog, start)
                env.em.set_state(snap)
                rec["human"] = _outcome(env, buttons[k:k + S.ROLLOUT].astype(np.uint8), start)
                env.em.set_state(snap)
                out[k] = rec
            obs, _, _, _, info = env.step(keys)
            ring[k] = obs
            ring.pop(k - max(past) - 1, None)
            k += 1
        final_x = F.mario_level_x({v: int(info.get(v, 0)) for v in ("player_x_posHi", "player_x_posLo")}) if info else None
    finally:
        env.close()
        del movie
    return out, final_x


def human_matches_recording(human: dict, ram: dict, k: int) -> bool:
    """Does the human's replay from the saved state end where the recording did?"""
    end = k + human["frames"] - 1
    x_rec = F.mario_level_x(F.ram_at(ram, end))
    x_start = F.mario_level_x(F.ram_at(ram, k - 1))
    return abs((x_rec - x_start) - human["dx"]) <= 1
