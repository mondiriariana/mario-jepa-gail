"""Automatic checks on a generated trace (nuReasoning's human verification, as rules).

A trace is accepted only if it agrees with everything that is known exactly: the decision
must be the player's, every critical component must exist in the facts, and counterfactual
actions and risk labels must be the simulated ones. A sample of accepted traces still needs
a human read for wording.
"""
from __future__ import annotations

import json
import re

from .prompts import FIELDS


def parse(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None


def validate(out: dict | None, rec: dict) -> list[str]:
    """Return the list of problems (empty = accepted)."""
    if out is None:
        return ["no parseable JSON object"]
    errs = []
    if set(out) != set(FIELDS):
        errs.append(f"fields {sorted(out)} != {sorted(FIELDS)}")
    d = out.get("decision") or {}
    if (d.get("horizontal"), d.get("vertical")) != (rec["decision"]["horizontal"], rec["decision"]["vertical"]):
        errs.append(f"decision {d} is not the player's")
    allowed = {o["type"] for o in rec["facts"]["objects"]} | {"pit", "obstacle"}
    for c in out.get("critical_components") or []:
        if not isinstance(c, dict) or c.get("type") not in allowed:
            errs.append(f"critical component {c} not in the facts")
    trace = out.get("trace") or ""
    if not trace or len(re.findall(r"[.!?](\s|$)", trace.strip())) > 1 or len(trace.split()) > 45:
        errs.append("trace is not one sentence of <= 45 words")
    sim = {(a["horizontal"], a["vertical"]): a["risk"] for a in rec["counterfactuals"]}
    for c in out.get("counterfactuals") or []:
        act = c.get("action") if isinstance(c, dict) else None
        key = tuple(s.strip() for s in act.split("+")) if isinstance(act, str) and "+" in act else None
        if key not in sim:
            errs.append(f"counterfactual action {act!r} was not simulated")
        elif c.get("risk") != sim[key]:
            errs.append(f"risk for {act!r} changed from {sim[key]} to {c.get('risk')}")
    return errs
