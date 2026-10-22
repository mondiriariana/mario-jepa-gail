"""GAIL + PPO in the real emulator, coupled with the JEPA latent-prediction objective.

One iteration:
  1. roll out the current policy in ``n_envs`` emulators for ``rollout`` steps; every
     observation is encoded online by the frozen V-JEPA 2 encoder;
  2. discriminator update  (D_psi only):  human (z, a_prev, a) vs agent (z, a_prev, a);
  3. reward r_t = softplus(D_psi logit) = -log(1 - D) on controllable steps, 0 otherwise;
  4. PPO epochs on the rollout. Each minibatch loss is
        L = L_PPO(adapter, policy, value) + lambda * L_JEPA(adapter, predictor)
            [+ bc_coef * L_BC(adapter, policy) if bc_coef > 0]
     where L_JEPA mixes human demo windows and the agent's own rollout windows.
The frozen V-JEPA 2 backbone is never updated.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mjg import common as C  # noqa: E402
from mjg.actions import N_ACTIONS  # noqa: E402
from mjg.encoder import FrozenVJEPA2  # noqa: E402
from mjg.models import Agent, Discriminator  # noqa: E402
from mjg.offline_eval import action_metrics, jepa_metrics  # noqa: E402
from mjg.vec_env import MarioVecEnv  # noqa: E402


def disc_update(disc, opt, demo, z_ag, prev_ag, a_ag, g, cfg, gen):
    n = min(g["disc_batch"], len(a_ag))
    stats = []
    for _ in range(g["disc_steps"]):
        zh, ph, ah = demo.disc_batch(n, gen)
        j = torch.randint(len(a_ag), (n,), device=a_ag.device)
        za, pa, aa = z_ag[j].float(), prev_ag[j], a_ag[j]
        zh = zh.requires_grad_(True)
        lh = disc(zh, ph, ah)
        la = disc(za, pa, aa)
        loss = F.binary_cross_entropy_with_logits(lh, torch.full_like(lh, 0.9)) + \
            F.binary_cross_entropy_with_logits(la, torch.zeros_like(la))
        grad = torch.autograd.grad(lh.sum(), zh, create_graph=True)[0]
        gp = grad.pow(2).flatten(1).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        (loss + g["disc_grad_pen"] * gp).backward()
        opt.step()
        stats.append([loss.item(), gp.item(), (lh > 0).float().mean().item(), (la < 0).float().mean().item(),
                      torch.sigmoid(la).mean().item()])
    s = np.mean(stats, 0)
    return {"d_loss": s[0], "d_gp": s[1], "d_acc_human": s[2], "d_acc_agent": s[3], "d_agent_prob": s[4]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", required=True)
    ap.add_argument("--init", help="BC checkpoint to start from")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = C.load_config(a.config, a.overrides)
    C.seed_all(cfg["seed"])
    dev = "cuda"
    g = cfg["gail"]
    stats = None
    if a.init and a.init != "none":
        agent, _, ck = C.load_ckpt(a.init, dev)
        stats = ck["stats"]
        cfg["model"] = ck["cfg"]["model"]
    demos = C.load_demos(cfg, dev, stats=stats, which=("train", "val"))
    tr, va = demos["train"], demos["val"]  # held-out test sessions are only touched by evaluate.py
    if stats is None:
        agent = Agent(cfg["model"]).to(dev)
    agent.train()
    disc = Discriminator(cfg["model"]["zdim"]).to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=g["lr"], eps=1e-5)
    dopt = torch.optim.Adam(disc.parameters(), lr=g["disc_lr"], betas=(0.5, 0.999))
    lam = g["jepa_lambda"] if agent.use_jepa else 0.0
    gen = torch.Generator().manual_seed(cfg["seed"] + 1)

    if g["levels"] == "auto":
        levels, freq = tr.level_freq()
    else:
        levels, freq = g["levels"], np.ones(len(g["levels"]))
    print("training levels", dict(zip(levels, freq.astype(int))), flush=True)
    enc = FrozenVJEPA2(grid=int(cfg["model"]["n_tokens"] ** 0.5))
    env = MarioVecEnv(g["n_envs"], os.environ["MJG_INTEGRATION"], levels, freq, g["max_episode_steps"],
                      seed=cfg["seed"])
    clips, prev = env.reset()
    z = tr.norm(enc(torch.from_numpy(clips)))
    prev = torch.from_numpy(prev).to(dev)
    control = torch.from_numpy(env.control.copy()).to(dev)

    T, N, K = g["rollout"], g["n_envs"], cfg["model"]["horizon"]
    log = f"{a.run}/gail_log.jsonl"
    ep_log = f"{a.run}/episodes.jsonl"
    C.dump_json(f"{a.run}/config.json", cfg)
    t_start = time.time()
    total_steps = 0
    for it in range(1, g["iterations"] + 1):
        t_it = time.time()
        Z = torch.empty((T + 1, N) + tuple(z.shape[1:]), dtype=torch.float16, device=dev)
        PREV = torch.empty((T, N), dtype=torch.long, device=dev)
        ACT, LOGP, VAL = (torch.empty((T, N), device=dev) for _ in range(3))
        DONE, CTRL = (torch.empty((T, N), dtype=torch.bool, device=dev) for _ in range(2))
        ep_rows = []
        for t in range(T):
            with torch.no_grad():
                logits, v, _ = agent(z, prev)
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
            Z[t], PREV[t], ACT[t], LOGP[t], VAL[t], CTRL[t] = z.half(), prev, act, dist.log_prob(act), v, control
            clips, prev_np, done, infos = env.step(act.cpu().numpy())
            DONE[t] = torch.from_numpy(done).to(dev)
            z = tr.norm(enc(torch.from_numpy(clips)))
            prev = torch.from_numpy(prev_np).to(dev)
            control = torch.from_numpy(env.control.copy()).to(dev)
            for i in np.nonzero(done)[0]:
                r = infos[i]
                ep_rows.append({"it": it, "level": r["level"], "steps": r["t"], "dx": r["dx"], "x": r["x"],
                                "died": r["died"], "completed": r["completed"], "truncated": r["truncated"]})
        Z[T] = z.half()
        total_steps += T * N
        with torch.no_grad():
            _, v_last, _ = agent(z, prev)
        t_roll = time.time() - t_it

        # ---- discriminator (D_psi only) ----
        m = CTRL.flatten()
        za, pa, aa = Z[:T].flatten(0, 1)[m], PREV.flatten()[m], ACT.flatten()[m].long()
        dstats = disc_update(disc, dopt, tr, za, pa, aa, g, cfg, gen) if m.any() else {}

        # ---- rewards and advantages ----
        with torch.no_grad():
            dl = disc(Z[:T].flatten(0, 1).float(), PREV.flatten(), ACT.flatten().long()).view(T, N)
            rew = F.softplus(dl).clamp(max=g["reward_clip"]) * CTRL
            adv = torch.zeros_like(rew)
            last = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                nv = v_last if t == T - 1 else VAL[t + 1]
                nonterm = (~DONE[t]).float()
                delta = rew[t] + g["gamma"] * nv * nonterm - VAL[t]
                last = delta + g["gamma"] * g["gae_lambda"] * nonterm * last
                adv[t] = last
            ret = adv + VAL

        # agent JEPA windows: t..t+K inside the rollout and the same episode
        if lam > 0:
            okw = torch.ones((T - K, N), dtype=torch.bool, device=dev)
            for k in range(K):
                okw &= ~DONE[k:T - K + k]
            wt, wn = torch.nonzero(okw, as_tuple=True)
            wacts = torch.stack([ACT[wt + k, wn] for k in range(K)], 1).long()

        # ---- PPO + lambda * JEPA ----
        b_z, b_prev, b_act = Z[:T].flatten(0, 1), PREV.flatten(), ACT.flatten().long()
        b_logp, b_adv, b_ret, b_ctrl = LOGP.flatten(), adv.flatten(), ret.flatten(), CTRL.flatten().float()
        am = b_adv[b_ctrl > 0]
        b_adv = (b_adv - am.mean()) / (am.std() + 1e-8) if len(am) > 1 else b_adv
        nb = T * N
        pstats = []
        for ep in range(g["ppo_epochs"]):
            perm = torch.randperm(nb, device=dev)
            for s in range(0, nb, g["minibatch"]):
                j = perm[s:s + g["minibatch"]]
                logits, v, H = agent(b_z[j].float(), b_prev[j])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(b_act[j])
                ratio = (lp - b_logp[j]).exp()
                w = b_ctrl[j]
                wsum = w.sum().clamp_min(1)
                pg = -(torch.min(ratio * b_adv[j], ratio.clamp(1 - g["clip"], 1 + g["clip"]) * b_adv[j]) * w).sum() / wsum
                ent = (dist.entropy() * w).sum() / wsum
                vl = F.mse_loss(v, b_ret[j])
                loss = pg + g["vf_coef"] * vl - g["ent_coef"] * ent
                lj_h = lj_a = torch.zeros((), device=dev)
                if lam > 0:
                    nj = g["minibatch"]
                    na = int(nj * g["jepa_agent_frac"]) if len(wt) else 0
                    zh, ah, zth = tr.jepa_batch(nj - na, gen)
                    lj_h = agent.jepa_loss(zh, ah, zth)
                    lj = lj_h * (nj - na) / nj
                    if na:
                        q = torch.randint(len(wt), (na,), device=dev)
                        lj_a = agent.jepa_loss(Z[wt[q], wn[q]].float(), wacts[q], Z[wt[q] + K, wn[q]].float())
                        lj = lj + lj_a * na / nj
                    loss = loss + lam * lj
                if g["bc_coef"] > 0:
                    zb, pb, yb = tr.bc_batch(g["minibatch"], gen)
                    lb, _, _ = agent(zb, pb)
                    loss = loss + g["bc_coef"] * F.cross_entropy(lb, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), g["max_grad_norm"])
                opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - (lp - b_logp[j])).mul(w).sum() / wsum
                    cf = ((ratio - 1).abs() > g["clip"]).float().mul(w).sum() / wsum
                pstats.append([pg.item(), vl.item(), ent.item(), kl.item(), cf.item(), lj_h.item(), lj_a.item()])
        ps = np.mean(pstats, 0)
        row = {"it": it, "env_steps": total_steps, "time": time.time() - t_start, "rollout_s": t_roll,
               "steps_per_s": T * N / (time.time() - t_it), "reward_mean": rew[CTRL].mean().item() if CTRL.any() else 0,
               "control_frac": CTRL.float().mean().item(), "pg_loss": ps[0], "v_loss": ps[1], "entropy": ps[2],
               "approx_kl": ps[3], "clipfrac": ps[4], "jepa_human": ps[5], "jepa_agent": ps[6],
               "agent_action_hist": torch.bincount(aa, minlength=N_ACTIONS).tolist(), **dstats}
        if ep_rows:
            row.update({"ep_n": len(ep_rows), "ep_dx": float(np.mean([e["dx"] for e in ep_rows])),
                        "ep_len": float(np.mean([e["steps"] for e in ep_rows])),
                        "ep_died": float(np.mean([e["died"] for e in ep_rows])),
                        "ep_completed": float(np.mean([e["completed"] for e in ep_rows]))})
            for e in ep_rows:
                C.jsonl(ep_log, e)
        if it % 10 == 0 or it == 1 or it == g["iterations"]:
            amx = action_metrics(agent, va)
            row.update({"val_acc": amx["accuracy"], "val_nll": amx["nll"], "val_bal_acc": amx["balanced_accuracy"],
                        "val_acc_on_change": amx["accuracy_when_action_changes"]})
            jm = jepa_metrics(agent, va, n=5000)
            if jm:
                row.update({"val_jepa_mse": jm["mse"], "val_jepa_r2": jm["r2"], "val_shuf_r2": jm["shuf_r2"]})
        C.jsonl(log, row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "agent_action_hist"},
              flush=True)
        if it % g["save_every"] == 0 or it == g["iterations"]:
            C.save_ckpt(f"{a.run}/gail_it{it:04d}.pt", agent, cfg, tr.stats, {"it": it, "env_steps": total_steps}, disc)
            C.save_ckpt(f"{a.run}/gail_last.pt", agent, cfg, tr.stats, {"it": it, "env_steps": total_steps}, disc)
    env.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
