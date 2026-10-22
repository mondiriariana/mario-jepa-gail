"""Discrete action space shared by the human demonstrations and the agent.

Humans press any subset of the NES buttons on every 60 Hz frame. The agent acts every
``FRAME_SKIP`` frames and holds one of ``len(COMBOS)`` button combinations for the whole
step. Human frames are projected onto the same set (``buttons_to_action``); the coverage
of this projection is reported by scripts/build_dataset.py.
"""
from __future__ import annotations

import numpy as np

FRAME_SKIP = 4  # agent decision every 4 emulator frames = 15 Hz

# stable-retro NES button order (env.buttons)
ENV_BUTTONS = ["B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]

COMBOS = [
    (),
    ("RIGHT",), ("RIGHT", "A"), ("RIGHT", "B"), ("RIGHT", "A", "B"),
    ("LEFT",), ("LEFT", "A"), ("LEFT", "B"), ("LEFT", "A", "B"),
    ("A",), ("B",), ("A", "B"),
    ("DOWN",), ("UP",),
]
NAMES = ["NOOP" if not c else "+".join(c) for c in COMBOS]
N_ACTIONS = len(COMBOS)
_INDEX = {frozenset(c): i for i, c in enumerate(COMBOS)}


def combo_to_buttons(i: int) -> list[int]:
    held = set(COMBOS[i])
    return [int(b in held) if b else 0 for b in ENV_BUTTONS]


ACTION_BUTTONS = np.array([combo_to_buttons(i) for i in range(N_ACTIONS)], dtype=np.uint8)


def buttons_to_action(pressed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project bool [N, 9] env-order button states onto combo indices.

    Rules: START/SELECT are ignored; LEFT+RIGHT together cancel; UP/DOWN are kept only
    when no horizontal direction is held (crouch / climb), and then A/B are dropped.
    Returns (action [N] int64, exact [N] bool: True when nothing had to be dropped).
    """
    pressed = np.asarray(pressed, dtype=bool)
    b = {n: pressed[:, i] for i, n in enumerate(ENV_BUTTONS) if n}
    n = len(pressed)
    out = np.zeros(n, dtype=np.int64)
    exact = np.ones(n, dtype=bool)
    for k in range(n):
        left, right = b["LEFT"][k], b["RIGHT"][k]
        if left and right:
            left = right = False
            exact[k] = False
        held = []
        if right:
            held.append("RIGHT")
        if left:
            held.append("LEFT")
        if not (left or right) and (b["DOWN"][k] or b["UP"][k]):
            held = ["DOWN"] if b["DOWN"][k] else ["UP"]
            exact[k] &= not (b["A"][k] or b["B"][k] or (b["DOWN"][k] and b["UP"][k]))
        else:
            if b["A"][k]:
                held.append("A")
            if b["B"][k]:
                held.append("B")
            exact[k] &= not (b["DOWN"][k] or b["UP"][k])
        exact[k] &= not (b["START"][k] or b["SELECT"][k])
        out[k] = _INDEX[frozenset(held)]
    return out, exact
