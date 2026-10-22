# Architecture: JEPA + GAIL Mario agent

```
 emulator screens (224x240, 60 Hz)
        │  8 screens, stride 2 (the 16 frames before the decision)
        ▼
 ┌──────────────────────────┐
 │ V-JEPA 2 ViT-L (frozen)  │  facebook/vjepa2-vitl-fpc64-256, bf16, no grad
 └──────────────────────────┘
        │ tokens [4 x 16 x 16 x 1024] → mean over time → 4x4 avg-pool
        ▼
   z_t  [16 x 1024]  (normalised with training-set mean/sd)
        │                                          │
        ▼                                          ▼
 ┌──────────────┐  H_t [16x256]   ┌───────────────────────────┐
 │ adapter A_ω  │───────────────▶│ predictor F_η             │──▶ ẑ_{t+K}  vs  z_{t+K} (frozen)
 │ (2-layer     │                 │ (H_t, a_t..a_{t+K-1})     │       L_JEPA (smooth-L1)
 │ transformer) │                 │ K = 4 steps = 16 frames   │
 └──────────────┘                 └───────────────────────────┘
        │ h_t [256]                         │ g_t(a) = F's pooled state when holding a for K steps,
        ▼                                   ▼   for every action a  (no gradient into F)
 ┌──────────────────────────────────────────────────┐
 │ policy π_θ(a | h_t, a_{t-1}, g_t(·))  + value V_φ │──▶ action every 4 frames (15 Hz)
 └──────────────────────────────────────────────────┘
                                                           ┌────────────────────────────┐
  z_t, a_{t-1}, a_t (human demos  vs  agent rollouts) ───▶ │ discriminator D_ψ          │──▶ r_t = −log(1 − D)
                                                           └────────────────────────────┘
```

## Where the JEPA objective enters decision-making

1. **Shared representation.** The adapter `A_ω` is the only path from the frozen latents to the
   policy, and it is also the predictor's input. `L_JEPA` therefore shapes the features the policy
   acts on, so they must keep whatever the next 16 frames depend on given the actions.
2. **Imagined consequences.** For each of the 14 actions the policy also gets `g_t(a)`, the
   predictor's internal estimate of "what happens if I hold `a` for 4 steps". The policy scores each
   action partly by its imagined consequence. v1 (`imagination_mode: gated`, used by all GAIL runs) is
   a learned bilinear score gated by a scalar initialised at 0; v2 (`mlp`) is a small MLP on
   `[context, g_t(a)]` with no gate. This path is computed under `no_grad`, so only `L_JEPA` trains the
   predictor.

   **Measured:** the policy barely uses this path. Switching it off changes the chosen action on
   0.4–0.8% of held-out steps (v1 and v2, BC and GAIL), see `results/imagination_check_*.json`. In the
   trained agents the JEPA objective therefore reaches decisions through the shared adapter (path 1).

## Which parameters each objective updates

| Objective | adapter A_ω | predictor F_η | policy/value θ, φ | discriminator D_ψ | V-JEPA 2 |
|---|---|---|---|---|---|
| L_BC (warm start; optional regulariser) | ✔ | – | ✔ (policy) | – | frozen |
| L_PPO, with the GAIL reward (policy + value + entropy) | ✔ | – (imagination path has no grad) | ✔ | – | frozen |
| λ·L_JEPA (human windows + agent's own rollout windows) | ✔ | ✔ | – | – | frozen |
| L_D (BCE + R1 gradient penalty) | – | – | – | ✔ | frozen |

One GAIL iteration alternates: rollout → `disc_steps` discriminator updates → GAIL rewards → PPO epochs,
where every minibatch minimises `L_PPO + λ·L_JEPA (+ bc_coef·L_BC)` with one optimiser over
{A_ω, F_η, θ, φ}. The discriminator has its own optimiser and its own encoder of the frozen latents,
so the adversarial game never changes the agent's representation directly.

## Design choices and why

- **Human observations come from emulator replays, not the mp4s.** Replaying each `.bk2` through
  the same emulator the agent plays in means the discriminator cannot tell human from agent by
  video-compression artifacts. `results/alignment_check.json` shows replay screen k = video frame k.
- **Frame skip 4 and the 14-combo action set.** See `docs/DATA.md` for how many human frames each
  choice represents exactly.
- **JEPA target = the frozen latent of the next disjoint clip.** The clip at step t covers frames
  f−15..f−1 and the clip at t+4 covers f+1..f+15, so the 4 actions in between are what decides the
  difference. Predicting a frozen target, not the adapter's own output, rules out representational
  collapse without an EMA target network.
- **Discriminator inputs (z_t, a_{t−1}, a_t).** Including the previous action lets it judge button
  timing (how long humans hold a combo), which is part of human-likeness.
- **Controllable steps only.** Death animations, level intros and flagpole slides ignore input; they
  are excluded from BC, from the discriminator and from the policy-gradient term
  (`player_state ∈ {8, 9, 10, 12}` and `oper_mode = 1` from RAM).
- **Episodes** start from CNeuroMod's level states and end at level completion, the first life lost
  or 1500 steps (100 s).
