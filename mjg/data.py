"""Human demonstration data: play index, step/clip layout, feature loading.

Step layout (identical for humans and the agent):
  decision step t starts at emulator frame f = FRAME_SKIP * t;
  observation o_t = screens [f-15, f-13, ..., f-1] (CLIP_LEN=8, stride 2), i.e. only what
  was on screen before the step's input; frames before 0 are clamped to the first screen;
  action a_t = button state on frame f, projected to mjg.actions.COMBOS;
  ram_t = RAM variables after frame f-1 (clamped to 0).
Only odd frames are ever needed for clips, so replays keep odd screens only.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass

import numpy as np

from .actions import FRAME_SKIP

CLIP_LEN = 8
CLIP_STRIDE = 2

# SMB player_state (RAM 0x000E): 8 = normal control; 9 = changing size, 10 = injury
# blink, 12 = getting fire flower (player still steers). Others are cut-scenes/deaths.
CONTROL_STATES = (8, 9, 10, 12)

_PAT = re.compile(r"(sub-\d+)_(ses-\d+)_task-mario_level-(w\dl\d)_rep-(\d+)\.bk2$")


@dataclass
class Play:
    play_id: str
    sub: str
    ses: str
    level: str
    rep: int
    bk2: str


def index_plays(data_root: str, subjects=None) -> list[Play]:
    out = []
    for p in sorted(glob.glob(os.path.join(data_root, "sub-*", "ses-*", "*.bk2"))):
        m = _PAT.search(os.path.basename(p))
        if not m or os.path.getsize(p) == 0:
            continue
        sub, ses, level, rep = m.groups()
        if subjects and sub not in subjects:
            continue
        out.append(Play(f"{sub}_{ses}_{level}_rep-{rep}", sub, ses, level, int(rep), p))
    return out


def clip_positions(n_steps: int, n_odd: int) -> np.ndarray:
    """Indices into the odd-frame array for every step's clip: [n_steps, CLIP_LEN].

    Odd frame k sits at position (k-1)//2. For step t (f=4t) the clip frames are
    f-1-2j (j = CLIP_LEN-1..0) -> positions 2t-1-j.
    """
    t = np.arange(n_steps)[:, None]
    j = np.arange(CLIP_LEN - 1, -1, -1)[None, :]
    pos = (FRAME_SKIP // 2) * t - 1 - j * (CLIP_STRIDE // 2)
    return np.clip(pos, 0, n_odd - 1)


def load_split(feat_dir: str, play_ids: list[str], mmap: bool = True):
    """Load features + metadata for the given plays.

    Returns dict of concatenated arrays plus ``play_start`` offsets. ``z`` is a list of
    per-play memory-mapped arrays to avoid holding every play in RAM twice.
    """
    zs, metas = [], []
    for pid in play_ids:
        zs.append(np.load(os.path.join(feat_dir, pid + ".z.npy"), mmap_mode="r" if mmap else None))
        metas.append(dict(np.load(os.path.join(feat_dir, pid + ".meta.npz"))))
    return zs, metas


def read_index(feat_dir: str) -> list[dict]:
    rows = []
    for p in sorted(glob.glob(os.path.join(feat_dir, "*.meta.json"))):
        with open(p) as f:
            rows.append(json.load(f))
    return rows
