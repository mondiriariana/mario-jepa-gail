"""Meta-action taxonomy and the counterfactual alternatives.

nuReasoning describes a driving decision as one longitudinal plus one lateral action. For
Mario the two axes are horizontal movement and jumping:

    horizontal: run right | walk right | stop | back off left
    vertical:   no jump | short hop | full jump

The human's decision at a key frame is read off their buttons over the next DECISION_WINDOW
frames (60 Hz). Alternatives are fixed button programs played from a saved emulator state.
"""
from __future__ import annotations

import numpy as np

from mjg.actions import ENV_BUTTONS

HORIZONTAL = ("run right", "walk right", "stop", "back off left")
VERTICAL = ("no jump", "short hop", "full jump")

DECISION_WINDOW = 16   # frames of buttons that define the horizontal decision (~0.27 s)
JUMP_WINDOW = 8        # a jump counts if A is newly pressed within this many frames
SHORT_HOP_MAX = 10     # A held for fewer frames than this = short hop (SMB jump height grows with hold)
ROLLOUT = 60           # frames each alternative is played for (1 s)

_B = {n: i for i, n in enumerate(ENV_BUTTONS) if n}


def decision_from_buttons(buttons: np.ndarray, f: int) -> dict:
    """Human meta-action at frame f from bool buttons [n_frames, 9] (env order)."""
    w = buttons[f:f + DECISION_WINDOW]
    right, left, run = w[:, _B["RIGHT"]], w[:, _B["LEFT"]], w[:, _B["B"]]
    if right.mean() >= 0.5:
        horiz = "run right" if (right & run).mean() >= 0.5 else "walk right"
    elif left.mean() >= 0.5:
        horiz = "back off left"
    else:
        horiz = "stop"
    a = buttons[:, _B["A"]]
    vert, hold = "no jump", 0
    for k in range(f, min(f + JUMP_WINDOW, len(a))):
        if a[k] and (k == 0 or not a[k - 1]):  # rising edge = a new jump
            j = k
            while j < len(a) and a[j]:
                j += 1
            hold = j - k
            vert = "short hop" if hold < SHORT_HOP_MAX else "full jump"
            break
    return {"horizontal": horiz, "vertical": vert, "jump_hold_frames": int(hold)}


def _program(horiz: str, vert: str, n: int = ROLLOUT) -> np.ndarray:
    """Button program [n, 9] for a meta-action. Frame 0 releases A so a jump can start."""
    p = np.zeros((n, len(ENV_BUTTONS)), dtype=np.uint8)
    if horiz in ("run right", "walk right"):
        p[:, _B["RIGHT"]] = 1
    if horiz == "run right":
        p[:, _B["B"]] = 1
    if horiz == "back off left":
        p[:, _B["LEFT"]] = 1
    if vert != "no jump":
        hold = 6 if vert == "short hop" else 30
        p[1:1 + hold, _B["A"]] = 1
    return p


# alternatives scored at every key frame (nuReasoning picks "meaningful" pairs; for Mario the
# full useful set is small enough to simulate exhaustively)
ALTERNATIVES = [
    ("run right", "no jump"), ("walk right", "no jump"), ("stop", "no jump"),
    ("back off left", "no jump"), ("run right", "short hop"), ("run right", "full jump"),
    ("walk right", "full jump"), ("stop", "full jump"),
]
PROGRAMS = {a: _program(*a) for a in ALTERNATIVES}


def risk_label(outcome: dict, best_dx: float) -> str:
    """Safe / suboptimal / unsafe from a simulated outcome (nuReasoning's three levels).

    unsafe: Mario dies within the rollout. suboptimal: survives but makes less than half the
    progress of the best surviving alternative (only when that one moved at least 16 px, so
    waiting is not penalised where nothing can advance). safe: otherwise.
    """
    if outcome["died"]:
        return "unsafe"
    if best_dx >= 16 and outcome["dx"] < 0.5 * best_dx:
        return "suboptimal"
    return "safe"
