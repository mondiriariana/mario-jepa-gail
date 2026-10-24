"""Exact scene facts from game RAM (nuReasoning's spatial stage, computed instead of detected).

Everything here is read from RAM, never from pixels: Mario's state, every on-screen object
relative to Mario (from the RAM oracle in mjg.objects) with its motion and time to contact,
and the terrain just ahead from the level's metatile buffer.

Conventions: screen pixels (224 x 240, see mjg.objects); ``dx`` > 0 = ahead of Mario
(to the right), ``dy`` > 0 = above Mario. Speeds are px/frame (60 Hz), measured over the
last LOOKBACK frames.
"""
from __future__ import annotations

import numpy as np

from mjg import objects as O

LOOKBACK = 4
NEAR_PX = 64          # objects within this horizontal distance are reported as relevant
ENEMY_TYPES = (O.TYPE_GOOMBA, O.TYPE_KOOPA, O.TYPE_ENEMY)

# --- level metatiles (APPROXIMATE: layout per SMB disassemblies, check with --debug-png) ---
# RAM 0x0500-0x069F holds two 16-column x 13-row pages of metatiles; row 0 is NES y = 32.
TILE_BASE, PAGE_SIZE, N_ROWS = 0x0500, 0xD0, 13
NON_SOLID = {0x00, 0xC2, 0xC3}  # empty, coin, underwater coin


def ram_at(ram: dict, k: int) -> dict:
    """RAM variables (dict var -> array) at frame index k, clamped, as plain ints."""
    k = max(0, min(k, len(next(iter(ram.values()))) - 1))
    return {v: int(a[k]) for v, a in ram.items()}


def mario_level_x(r: dict) -> int:
    return r["player_x_posHi"] * 256 + r["player_x_posLo"]


def mario_state(r: dict, r_prev: dict) -> dict:
    vx = (mario_level_x(r) - mario_level_x(r_prev)) / LOOKBACK
    vy = (r_prev["player_y_pos"] - r["player_y_pos"]) / LOOKBACK  # > 0 = moving up
    size = "big" if r.get("player_size", 1) == 0 else "small"
    return {"level_x": mario_level_x(r), "speed_x": round(vx, 2), "speed_y": round(vy, 2),
            "airborne": bool(r.get("jump_airborne", 0)), "size": size,
            "powerstate": int(r.get("powerstate", 0))}


def _motion(vx_rel: float) -> str:
    if abs(vx_rel) < 0.25:
        return "not closing"
    return "closing in" if vx_rel < 0 else "moving away"


def objects(r: dict, r_prev: dict) -> list[dict]:
    """On-screen objects relative to Mario, nearest first, with motion and time to contact."""
    now, before = O.oracle_objects(r), O.oracle_objects(r_prev)
    mario = [o for o in now if o.type == O.TYPE_MARIO]
    if not mario:
        return []
    m = mario[0]
    m_prev = [o for o in before if o.type == O.TYPE_MARIO]
    prev_by_slot = {o.slot: o for o in before if o.slot >= 0}
    out = []
    for o in now:
        if o.type == O.TYPE_MARIO:
            continue
        dx = o.center[0] - m.center[0]
        dy = m.center[1] - o.center[1]
        rel = {"type": O.TYPE_NAMES[o.type], "slot": o.slot, "dx": round(dx, 1), "dy": round(dy, 1),
               "box_screen": [round(o.x0, 1), round(o.y0, 1), round(o.x1, 1), round(o.y1, 1)]}
        p = prev_by_slot.get(o.slot)
        if p is not None and m_prev:
            # change in horizontal gap per frame (screen coords: scrolling cancels out)
            gap_now, gap_prev = dx, p.center[0] - m_prev[0].center[0]
            closing = (gap_now - gap_prev) / LOOKBACK if dx >= 0 else (gap_prev - gap_now) / LOOKBACK
            rel["gap_change_per_frame"] = round(closing, 2)
            rel["motion"] = _motion(closing)
            # time to contact: frames until the boxes touch horizontally at the current rate
            edge_gap = max(0.0, abs(dx) - 16)
            rel["frames_to_contact"] = round(edge_gap / -closing, 1) if closing < -0.25 else None
        out.append(rel)
    out.sort(key=lambda d: abs(d["dx"]) + abs(d["dy"]))
    return out


def describe_position(dx: float, dy: float) -> str:
    h = "ahead" if dx > 0 else "behind"
    v = "same height" if abs(dy) < 12 else ("above" if dy > 0 else "below")
    return f"{abs(dx):.0f} px {h}, {v}" + ("" if v == "same height" else f" ({abs(dy):.0f} px)")


def _tile(ram_bytes: np.ndarray, col: int, row: int) -> int:
    page, c = (col // 16) & 1, col % 16
    return int(ram_bytes[TILE_BASE + page * PAGE_SIZE + row * 16 + c])


def terrain_ahead(ram_bytes: np.ndarray, r: dict, n_cols: int = 5) -> dict:
    """Pits and obstacles in the next n_cols 16-px columns (APPROXIMATE, verify visually).

    support row = metatile row under Mario's feet; body row = the one above it.
    """
    feet_nes = r["player_y_pos"] + 32
    support = (feet_nes - 32) // 16
    if not 0 < support < N_ROWS or r.get("player_y_screen", 1) != 1:
        return {"pit_ahead_px": None, "obstacle_ahead_px": None, "obstacle_height_tiles": None,
                "note": "Mario not over the level grid"}
    col0 = (mario_level_x(r) + 8) // 16
    x_in_col = (mario_level_x(r) + 8) % 16
    pit = obstacle = height = None
    for i in range(1, n_cols + 1):
        c = col0 + i
        dist = i * 16 - x_in_col
        column = [_tile(ram_bytes, c, row) not in NON_SOLID for row in range(N_ROWS)]
        if obstacle is None and column[support - 1]:
            obstacle, height = dist, 0
            row = support - 1
            while row >= 0 and column[row]:
                height += 1
                row -= 1
        if pit is None and not any(column[support:]):
            pit = dist
    return {"pit_ahead_px": pit, "obstacle_ahead_px": obstacle, "obstacle_height_tiles": height}


def scene_facts(ram: dict, k: int, ram_bytes: np.ndarray | None = None) -> dict:
    """All facts at frame index k (the state the player saw before acting on frame k)."""
    r, r_prev = ram_at(ram, k - 1), ram_at(ram, k - 1 - LOOKBACK)
    objs = objects(r, r_prev)
    facts = {"time_left": r.get("time", 0), "mario": mario_state(r, r_prev), "objects": objs,
             "nearby": [o for o in objs if abs(o["dx"]) <= NEAR_PX]}
    if ram_bytes is not None:
        facts["terrain"] = terrain_ahead(ram_bytes, r)
    return facts
