"""How much does the policy rely on the predictor's imagined consequences?

For each checkpoint with an imagination path: report the learned gate value, then compare
held-out human action agreement and the policy's action distribution with the path on
versus switched off (gate = 0). Runs on CPU or GPU.

python scripts/inspect_imagination.py --device cpu NAME=ckpt [NAME=ckpt ...]
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mjg import common as C  # noqa: E402
from mjg.offline_eval import action_metrics  # noqa: E402


def set_imagination(agent, on):
    """Switch the imagination path on/off without touching the trained weights."""
    agent.pv.imagination = on


@torch.no_grad()
def policy_shift(agent, demo, n=20000, batch=2048):
    """Mean KL(pi_on || pi_off) and argmax agreement between the two policies."""
    idx = demo.bc_idx[:n]
    kls, same = [], []
    for s in range(0, len(idx), batch):
        i = idx[s:s + batch]
        z, prev = demo.norm(demo.z[i]), demo.prev[i].to(demo.device)
        set_imagination(agent, True)
        lon, _, _ = agent(z, prev)
        set_imagination(agent, False)
        loff, _, _ = agent(z, prev)
        kls.append(F.kl_div(F.log_softmax(loff, -1), F.log_softmax(lon, -1), log_target=True, reduction="none").sum(-1))
        same.append((lon.argmax(-1) == loff.argmax(-1)).float())
    set_imagination(agent, True)
    return float(torch.cat(kls).mean()), float(torch.cat(same).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="+")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = {}
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        agent, _, ck = C.load_ckpt(path, a.device)
        agent.eval()
        if not agent.pv.imagination:
            continue
        ho = C.load_demos(ck["cfg"], a.device, stats=ck["stats"], which=("heldout",))["heldout"]
        on = action_metrics(agent, ho)
        kl, agree = policy_shift(agent, ho)
        gate = float(agent.pv.img_gate.item()) if hasattr(agent.pv, "img_gate") else None
        set_imagination(agent, False)
        off = action_metrics(agent, ho)
        set_imagination(agent, True)
        keys = ("accuracy", "nll", "balanced_accuracy", "accuracy_when_action_changes")
        res[name] = {"mode": agent.pv.imagination_mode, "gate": gate, "kl_on_vs_off": kl, "argmax_agreement_on_vs_off": agree,
                     "heldout_on": {k: on[k] for k in keys}, "heldout_off": {k: off[k] for k in keys}}
        print(name, json.dumps(res[name]), flush=True)
    if a.out:
        C.dump_json(a.out, res)


if __name__ == "__main__":
    main()
