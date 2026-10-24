"""The trace-writing prompt: nuReasoning's decision prompt (their Figs. S4-S5), adapted to Mario.

Kept from the paper: the realised behaviour is ground truth and only gets explained; critical
components may use past and current information only; a fixed priority hierarchy decides
which constraint the one-sentence trace must name; exactly one JSON object as output.
Changed: the spatial facts are exact (game RAM), and counterfactual risks come from the
emulator, so the model only words them and may not change them.
"""
from __future__ import annotations

import json

from . import facts as F

FIELDS = ("scene", "critical_components", "decision", "trace", "counterfactuals")

SYSTEM = """You annotate a human player's decision in Super Mario Bros (NES).
You get the facts of the moment, read exactly from game memory, the decision the player actually
made, what happened next, and the simulated outcome of alternative actions.
Return a single valid JSON object with exactly these five fields and no other text:
scene, critical_components, decision, trace, counterfactuals.

Rules:
1. The player's decision is ground truth. Copy it exactly into "decision" ({"horizontal": ..., "vertical": ...})
   and explain it; never argue for another action.
2. "scene": one objective sentence about the setting (level, Mario's state, what is around). No decision words.
3. "critical_components": only the objects or terrain from the facts that directly force this decision, judged
   from the current and past moments only (never from what happens later). Each item:
   {"type": <object type from the facts, or "pit" / "obstacle">, "relative": <copy the position from the facts>,
    "impact": <how it constrains Mario>}. Use [] if nothing forces the decision (plain progress).
4. Priority hierarchy for the explanation, highest first:
   (1) survival: enemy contact, falling into a pit;
   (2) obstacles: pipes, walls, steps that block running;
   (3) opportunity: stomping an enemy, coins, power-ups;
   (4) progress: keep moving right.
5. "trace": ONE sentence naming the highest-priority constraint and why it leads to the decision.
6. "counterfactuals": for each simulated alternative, {"action": <copy as "horizontal + vertical",
   e.g. "run right + full jump">, "risk": <copy safe/suboptimal/unsafe>,
   "reason": <short, consistent with its simulated outcome>}. Never change a risk label.
"""


def _outcome_text(o: dict) -> str:
    if o["died"]:
        return f"dies after {o['died_after_frames']} frames"
    parts = [f"{o['dx']:+d} px", "survives"]
    if o["score_gain"]:
        parts.append(f"score +{o['score_gain']}")
    if o["powered_up"]:
        parts.append("powers up")
    if o["completed_level"]:
        parts.append("finishes the level")
    return ", ".join(parts)


def facts_text(rec: dict) -> str:
    f = rec["facts"]
    m = f["mario"]
    lines = [f"Level {rec['level']}, time left {f['time_left']}.",
             f"Mario: {m['size']}, {'in the air' if m['airborne'] else 'on the ground'}, "
             f"horizontal speed {m['speed_x']:+.2f} px/frame."]
    for o in f["nearby"] or []:
        s = f"- {o['type']}: {F.describe_position(o['dx'], o['dy'])}"
        if o.get("motion"):
            s += f", {o['motion']}"
        if o.get("frames_to_contact") is not None:
            s += f", contact in about {o['frames_to_contact']:.0f} frames"
        lines.append(s)
    if not f["nearby"]:
        lines.append("- no objects within 64 px")
    t = f.get("terrain") or {}
    if t.get("pit_ahead_px") is not None:
        lines.append(f"- pit: {t['pit_ahead_px']} px ahead")
    if t.get("obstacle_ahead_px") is not None:
        lines.append(f"- obstacle: {t['obstacle_ahead_px']} px ahead, {t['obstacle_height_tiles']} tile(s) high")
    return "\n".join(lines)


def user_prompt(rec: dict) -> str:
    d = rec["decision"]
    alts = "\n".join(f"- {a['horizontal']} + {a['vertical']}: {_outcome_text(a['outcome'])} -> {a['risk']}"
                     for a in rec["counterfactuals"])
    return (f"FACTS (from game memory, 1 frame = 1/60 s):\n{facts_text(rec)}\n\n"
            f"PLAYER'S DECISION (ground truth): {json.dumps({'horizontal': d['horizontal'], 'vertical': d['vertical']})}\n"
            f"WHAT HAPPENED NEXT (hindsight, 1 s): {_outcome_text(rec['human_outcome'])}\n\n"
            f"SIMULATED ALTERNATIVES (1 s each, from the same moment):\n{alts}\n\n"
            "Screens, if attached: current moment first, then earlier moments.\n"
            "Return the JSON object.")
