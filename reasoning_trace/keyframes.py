"""Decision-critical key frames, scored 1-10 with nuReasoning's behaviour-centric rubric.

Their rule: a moment is valuable only if something in the scene changes what the player
does; an object merely being visible is not enough. Candidates are decision steps
(every FRAME_SKIP frames) where the player has control. Score:

    1   base
    +3  the player's action changes at this step (projected to mjg.actions.COMBOS)
    +2  an enemy is within NEAR_PX ahead (or 16 px behind) at roughly Mario's height
    +2  ... and within 24 px
    +2  Mario dies within the next 2 s (rule of results/death_rule_check.json)

Without an action change or an imminent death the score is capped at 4 (presence alone).
Frames scoring >= MIN_SCORE are kept, at most MAX_PER_PLAY, at least MIN_GAP frames apart.
"""
from __future__ import annotations

import numpy as np

from mjg import actions as A
from mjg import data as D

from . import facts as F

MIN_SCORE, MAX_PER_PLAY, MIN_GAP = 5, 12, 30
DEATH_HORIZON = 120  # frames (2 s)


def death_frames(ram: dict) -> np.ndarray:
    """Frame indices where a death starts (player_state 6/11 or Mario below the screen)."""
    ps, ys = ram["player_state"], ram["player_y_screen"]
    dying = np.isin(ps, (6, 11)) | (ys >= 2)
    return np.flatnonzero(dying & ~np.r_[False, dying[:-1]])


def score_frames(ram: dict, buttons: np.ndarray) -> list[dict]:
    """Score every controllable decision step. Returns dicts with frame, score, reasons."""
    act, _ = A.buttons_to_action(buttons)
    deaths = death_frames(ram)
    n = len(buttons)
    out = []
    for f in range(A.FRAME_SKIP * 2, n - A.FRAME_SKIP, A.FRAME_SKIP):
        r = F.ram_at(ram, f - 1)
        if r["player_state"] not in D.CONTROL_STATES or r["oper_mode"] != 1:
            continue
        reasons, score = [], 1
        changed = act[f] != act[f - A.FRAME_SKIP]
        if changed:
            score += 3
            reasons.append(f"action change {A.NAMES[act[f - A.FRAME_SKIP]]} -> {A.NAMES[act[f]]}")
        near = [o for o in F.objects(r, F.ram_at(ram, f - 1 - F.LOOKBACK))
                if o["type"] in ("goomba", "koopa", "enemy") and -16 <= o["dx"] <= F.NEAR_PX and abs(o["dy"]) <= 32]
        if near:
            score += 2
            reasons.append(f"{near[0]['type']} {F.describe_position(near[0]['dx'], near[0]['dy'])}")
            if abs(near[0]["dx"]) <= 24:
                score += 2
        dies = bool(((deaths >= f) & (deaths < f + DEATH_HORIZON)).any())
        if dies:
            score += 2
            reasons.append("death within 2 s")
        if not (changed or dies):
            score = min(score, 4)
        out.append({"frame": f, "step": f // A.FRAME_SKIP, "score": min(score, 10), "reasons": reasons,
                    "dies_within_2s": dies})
    return out


def select(scored: list[dict]) -> list[dict]:
    """Highest-scoring frames >= MIN_SCORE, MIN_GAP apart, at most MAX_PER_PLAY, in time order."""
    keep = []
    for c in sorted((s for s in scored if s["score"] >= MIN_SCORE), key=lambda s: (-s["score"], s["frame"])):
        if all(abs(c["frame"] - k["frame"]) >= MIN_GAP for k in keep):
            keep.append(c)
        if len(keep) == MAX_PER_PLAY:
            break
    return sorted(keep, key=lambda s: s["frame"])
