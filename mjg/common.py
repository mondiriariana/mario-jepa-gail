"""Config, seeding, checkpoint helpers shared by the training scripts."""
from __future__ import annotations

import copy
import json
import os
import random

import numpy as np
import torch
import yaml

from . import data as D
from .demos import DemoSet, split_plays


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for o in overrides:  # "gail.lr=1e-4"
        k, v = o.split("=", 1)
        node = cfg
        *parents, leaf = k.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(v)
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def features_dir(cfg):
    """Feature folder from the config, with environment variables such as ${MJG_ROOT} expanded."""
    path = os.path.expandvars(cfg["data"]["features"])
    if "$" in path:
        raise RuntimeError(f"unset environment variable in data.features: {path} (see slurm/paths.example.sh)")
    return path


def load_demos(cfg, device="cuda", stats=None, which=("train", "val", "heldout")):
    feat = features_dir(cfg)
    rows = D.read_index(feat)
    rows = [r for r in rows if r["sub"] in cfg["data"]["subjects"]]
    train, val, held = split_plays(rows, cfg["data"]["heldout_sessions_every"])
    if cfg["data"].get("max_train_plays"):
        rng = np.random.default_rng(cfg["seed"])
        keep = sorted(rng.choice(len(train), min(len(train), cfg["data"]["max_train_plays"]), replace=False))
        train = [train[i] for i in keep]
    out = {}
    K = cfg["model"]["horizon"]
    if "train" in which:
        out["train"] = DemoSet(feat, train, K, stats=stats, device=device)
        stats = out["train"].stats
    if "val" in which:
        out["val"] = DemoSet(feat, val, K, stats=stats, device=device)
    if "heldout" in which:
        out["heldout"] = DemoSet(feat, held, K, stats=stats, device=device)
    out["stats"] = stats
    return out


def save_ckpt(path, agent, cfg, stats, extra=None, disc=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"agent": agent.state_dict(), "disc": disc.state_dict() if disc else None,
                "cfg": cfg, "stats": stats, "extra": extra or {}}, path)


def load_ckpt(path, device="cuda"):
    from .models import Agent, Discriminator
    ck = torch.load(path, map_location=device, weights_only=False)
    agent = Agent(ck["cfg"]["model"]).to(device)
    agent.load_state_dict(ck["agent"])
    disc = None
    if ck.get("disc"):
        disc = Discriminator(ck["cfg"]["model"]["zdim"]).to(device)
        disc.load_state_dict(ck["disc"])
    return agent, disc, ck


def dump_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)


def jsonl(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")


def snapshot_cfg(cfg):
    return copy.deepcopy(cfg)
