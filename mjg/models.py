"""Trainable modules on top of the frozen V-JEPA 2 latents.

Parameter groups and which objective updates them (see docs/ARCHITECTURE.md):

  adapter    A_w  : z_t (16x1024, frozen V-JEPA 2) -> tokens H_t (16xd) + summary h_t (d)
                    updated by L_BC, L_PPO (policy + value) and lambda * L_JEPA
  predictor  F_eta: (H_t, a_t..a_{t+K-1}) -> z_hat_{t+K} (16x1024)
                    updated by L_JEPA only
  policy     pi_th: (h_t, a_{t-1}, g_t(a) for every a) -> logits over actions,
                    where g_t(a) = pooled predictor state for "hold a for K steps"
                    (the imagination path is computed without gradient, so PPO/BC never
                    update F_eta through it); updated by L_BC and L_PPO
  value      V_phi: (h_t, a_{t-1}) -> scalar; updated by L_PPO value loss
  discrim.   D_psi: (z_t, a_{t-1}, a_t) -> logit "human"; its own small encoder of the
                    frozen latents, updated by L_D only (never touches the adapter)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .actions import N_ACTIONS

START = N_ACTIONS  # index of the "no previous action" token


def _encoder(d, layers, heads):
    layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True,
                                       activation="gelu")
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class Adapter(nn.Module):
    def __init__(self, zdim=1024, n_tokens=16, d=256, layers=2, heads=4):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(zdim), nn.Linear(zdim, d))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.enc = _encoder(d, layers, heads)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, z):
        x = self.inp(z.float())
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1) + self.pos
        x = self.out_norm(self.enc(x))
        return x[:, 1:], x[:, 0]  # tokens H_t, summary h_t


class Predictor(nn.Module):
    """Action-conditioned latent predictor: predicts the frozen V-JEPA 2 latent of the
    clip K steps ahead from the adapter tokens and the K actions taken in between.

    The output is a residual on the current latent, z_hat_{t+K} = z_t + out(.), with the
    output layer initialised at zero, so training starts from the "nothing changes" (copy)
    prediction and has to learn what the actions and the game add to it."""

    def __init__(self, zdim=1024, n_tokens=16, d=256, horizon=4, layers=2, heads=4):
        super().__init__()
        self.horizon = horizon
        self.act = nn.Embedding(N_ACTIONS, d)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + horizon, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.enc = _encoder(d, layers, heads)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, zdim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def hidden(self, H, acts):
        x = torch.cat([H, self.act(acts)], 1) + self.pos
        return self.norm(self.enc(x))[:, : H.shape[1]]

    def forward(self, H, acts, z):
        """H [B,16,d], acts [B,K], z [B,16,zdim] (normalised z_t) ->
        (z_hat_{t+K} [B,16,zdim] in normalised latent units, hidden)."""
        h = self.hidden(H, acts)
        return z.float() + self.out(h), h

    @torch.no_grad()
    def imagine_all(self, H):
        """g(a) for every action held for K steps: [B, N_ACTIONS, d] (no gradient)."""
        B, T, d = H.shape
        Hr = H.detach().unsqueeze(1).expand(B, N_ACTIONS, T, d).reshape(B * N_ACTIONS, T, d)
        acts = torch.arange(N_ACTIONS, device=H.device).repeat(B)[:, None].expand(-1, self.horizon)
        return self.hidden(Hr, acts).mean(1).view(B, N_ACTIONS, d)


class PolicyValue(nn.Module):
    """imagination_mode "gated" (v1): logits += gate * <c, W g(a)>, gate initialised at 0.
    "mlp" (v2): logits += MLP([c, g(a)]) per action, no gate."""

    def __init__(self, d=256, hidden=256, imagination=True, imagination_mode="gated"):
        super().__init__()
        self.imagination = imagination
        self.imagination_mode = imagination_mode
        self.prev = nn.Embedding(N_ACTIONS + 1, d)
        self.trunk = nn.Sequential(nn.Linear(2 * d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.logits = nn.Linear(hidden, N_ACTIONS)
        if imagination and imagination_mode == "gated":
            # score each imagined consequence g(a) against the current context
            self.img = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden))
            self.img_gate = nn.Parameter(torch.zeros(1))
        elif imagination:
            self.img_norm = nn.LayerNorm(d)
            self.img_mlp = nn.Sequential(nn.Linear(hidden + d, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.value = nn.Sequential(nn.Linear(2 * d, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, h, prev_a, g=None):
        x = torch.cat([h, self.prev(prev_a)], -1)
        c = self.trunk(x)
        logits = self.logits(c)
        if self.imagination and g is not None and self.imagination_mode == "gated":
            logits = logits + self.img_gate * torch.einsum("bh,bah->ba", c, self.img(g)) / c.shape[-1] ** 0.5
        elif self.imagination and g is not None:
            cg = torch.cat([c.unsqueeze(1).expand(-1, g.shape[1], -1), self.img_norm(g)], -1)
            logits = logits + self.img_mlp(cg).squeeze(-1)
        return logits, self.value(x).squeeze(-1)


class Agent(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d"]
        self.adapter = Adapter(cfg["zdim"], cfg["n_tokens"], d, cfg["adapter_layers"])
        self.use_jepa = cfg["use_jepa"]
        self.predictor = Predictor(cfg["zdim"], cfg["n_tokens"], d, cfg["horizon"], cfg["pred_layers"]) \
            if self.use_jepa else None
        self.pv = PolicyValue(d, cfg["hidden"], imagination=self.use_jepa and cfg["imagination"],
                              imagination_mode=cfg.get("imagination_mode", "gated"))

    def forward(self, z, prev_a):
        H, h = self.adapter(z)
        g = self.predictor.imagine_all(H) if (self.pv.imagination and self.predictor is not None) else None
        logits, v = self.pv(h, prev_a, g)
        return logits, v, H

    def jepa_loss(self, z, acts, z_target, H=None):
        """Normalized-latent L1 between predicted and actual frozen latents K steps ahead."""
        if H is None:
            H, _ = self.adapter(z)
        zh, _ = self.predictor(H, acts, z)
        return F.smooth_l1_loss(zh, z_target.float(), beta=1.0)


class Discriminator(nn.Module):
    def __init__(self, zdim=1024, d=128, hidden=256):
        super().__init__()
        sn = nn.utils.parametrizations.spectral_norm
        self.tok = nn.Sequential(nn.LayerNorm(zdim), sn(nn.Linear(zdim, d)), nn.GELU())
        self.act = nn.Embedding(N_ACTIONS, 32)
        self.prev = nn.Embedding(N_ACTIONS + 1, 32)
        self.mlp = nn.Sequential(sn(nn.Linear(2 * d + 64, hidden)), nn.GELU(), sn(nn.Linear(hidden, hidden)),
                                 nn.GELU(), sn(nn.Linear(hidden, 1)))

    def forward(self, z, prev_a, a):
        t = self.tok(z.float())
        x = torch.cat([t.mean(1), t.amax(1), self.act(a), self.prev(prev_a)], -1)
        return self.mlp(x).squeeze(-1)
