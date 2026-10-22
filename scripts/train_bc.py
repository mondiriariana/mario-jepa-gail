"""Behavioural-cloning warm start on human demos, optionally with the JEPA objective.

  L = L_BC(adapter, policy) + lambda * L_JEPA(adapter, predictor)

With model.use_jepa=false (or bc.jepa_lambda=0 and imagination off) this is plain BC.
The checkpoint with the lowest validation-session action NLL is kept (validation sessions
are disjoint from both training and held-out test sessions). Writes <run>/bc.pt,
<run>/bc_log.jsonl and <run>/bc_eval.json (validation and held-out test metrics).
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mjg import common as C  # noqa: E402
from mjg.models import Agent  # noqa: E402
from mjg.offline_eval import action_metrics, jepa_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", required=True)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = C.load_config(a.config, a.overrides)
    C.seed_all(cfg["seed"])
    dev = "cuda"
    demos = C.load_demos(cfg, dev)
    tr, va, ho = demos["train"], demos["val"], demos["heldout"]
    print(f"train steps {len(tr)} (control {len(tr.bc_idx)}) from {len(tr.rows)} plays; "
          f"val {len(va)} from {len(va.rows)} plays; held-out {len(ho)} from {len(ho.rows)} plays", flush=True)
    best = (float("inf"), 0, None)
    agent = Agent(cfg["model"]).to(dev)
    b = cfg["bc"]
    opt = torch.optim.AdamW(agent.parameters(), lr=b["lr"], weight_decay=b["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, b["lr"], total_steps=b["steps"], pct_start=0.05)
    gen = torch.Generator().manual_seed(cfg["seed"])
    lam = b["jepa_lambda"] if agent.use_jepa else 0.0
    log = f"{a.run}/bc_log.jsonl"
    t0 = time.time()
    for it in range(1, b["steps"] + 1):
        z, prev, y = tr.bc_batch(b["batch"], gen)
        logits, _, _ = agent(z, prev)
        l_bc = F.cross_entropy(logits, y, label_smoothing=b["label_smoothing"])
        loss = l_bc
        l_j = torch.zeros(())
        if lam > 0:
            zj, acts, zt = tr.jepa_batch(b["batch"], gen)
            l_j = agent.jepa_loss(zj, acts, zt)
            loss = loss + lam * l_j
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 50 == 0:
            row = {"step": it, "l_bc": l_bc.item(), "l_jepa": l_j.item(),
                   "acc": (logits.argmax(-1) == y).float().mean().item(), "time": time.time() - t0}
            if it % b["eval_every"] == 0 or it == b["steps"]:
                am = action_metrics(agent, va)
                row.update({"val_acc": am["accuracy"], "val_nll": am["nll"], "val_bal_acc": am["balanced_accuracy"],
                            "val_acc_on_change": am["accuracy_when_action_changes"]})
                jm = jepa_metrics(agent, va, n=5000)
                if jm:
                    row.update({"val_jepa_mse": jm["mse"], "val_jepa_r2": jm["r2"], "val_copy_r2": jm["copy_r2"],
                                "val_shuf_r2": jm["shuf_r2"]})
                if am["nll"] < best[0]:
                    best = (am["nll"], it, copy.deepcopy(agent.state_dict()))
            C.jsonl(log, row)
            print(row, flush=True)
    agent.load_state_dict(best[2])
    print(f"kept step {best[1]} (val nll {best[0]:.4f})", flush=True)
    res = {"best_step": best[1], "heldout_actions": action_metrics(agent, ho), "heldout_jepa": jepa_metrics(agent, ho),
           "val_actions": action_metrics(agent, va), "val_jepa": jepa_metrics(agent, va),
           "n_train_plays": len(tr.rows), "n_val_plays": len(va.rows),
           "n_heldout_plays": len(ho.rows), "train_plays": [r["play_id"] for r in tr.rows],
           "heldout_plays": [r["play_id"] for r in ho.rows]}
    C.dump_json(f"{a.run}/bc_eval.json", res)
    C.save_ckpt(f"{a.run}/bc.pt", agent, cfg, tr.stats, {"best_step": best[1]})
    print("held-out:", {k: v for k, v in res["heldout_actions"].items() if not isinstance(v, list)}, flush=True)
    print("held-out jepa:", res["heldout_jepa"], flush=True)


if __name__ == "__main__":
    main()
