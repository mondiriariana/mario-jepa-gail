"""Object-level ("segment") features to add to the whole-frame V-JEPA 2 latents.

For each decision step we find the objects on the last screen of the clip, cut the same
box out of all 8 clip frames, and encode that small clip with the frozen V-JEPA 2 (mean
over its tokens). Each object becomes one row of ``zdim + 6`` numbers:

    [ raw V-JEPA 2 crop latent (zdim) | cx, cy, w, h (fractions of the screen) | type | valid ]

and a step holds ``K`` such rows (Mario first, then the others by distance to Mario,
padded with valid = 0). An agent observation is then the 16 whole-frame tokens (padded
with 6 zeros) followed by the K object rows: ``[16 + K, zdim + 6]``, see ``pack_obs``.

Two detectors share one interface (a list of ``Obj`` per screen):

* ``oracle``: boxes read from emulator RAM (Mario, the 5 enemy slots, the power-up slot).
  Exact and free; it is also the reference SAM is scored against (scripts/check_detector.py).
* ``sam``: Meta SAM 3 with text prompts, through Hugging Face ``transformers`` (>= 5.0).
  Gated model: needs approved access, the weights on the cluster and the SAM environment.
* ``gdino``: Grounding DINO, an ungated open-vocabulary detector with the same text prompts;
  a stand-in while SAM 3 access is pending.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .emulator import SCREEN_SHAPE

SCREEN_H, SCREEN_W = SCREEN_SHAPE[:2]  # 224 x 240
# The NES draws 256 x 240; stable-retro's screen drops 8 px on each side. RAM coordinates
# are in NES pixels, so screen = RAM - offset. Checked visually on a box overlay
# (scripts/build_object_features.py --debug-png): enemy boxes sit on the sprites.
X_OFF, Y_OFF = 8, 8

TYPE_MARIO, TYPE_GOOMBA, TYPE_KOOPA, TYPE_ENEMY, TYPE_ITEM, TYPE_SCENERY = range(6)
N_TYPES = 6
TYPE_NAMES = ["mario", "goomba", "koopa", "enemy", "item", "scenery"]
GOOMBA_IDS = {0x06}
KOOPA_IDS = {0x00, 0x01, 0x03, 0x04, 0x09, 0x0E, 0x0F, 0x10}
SCENERY_MIN_ID = 0x24  # lifts / platforms are enemy-slot objects too
META = 6  # cx, cy, w, h, type, valid

OBJ_RAM_VARS = (
    ["player_x_posHi", "player_x_posLo", "player_y_pos", "player_y_screen", "player_size",
     "xscrollHi", "xscrollLo"]
    + [f"enemy_drawn{15 + i}" for i in range(6)]
    + [f"enemy_id_{i}" for i in range(6)]
    + [f"enemy_x_{i}" for i in range(6)]
    + [f"enemy_pageloc_{i}" for i in range(6)]
    + [f"enemy_y_{i}" for i in range(6)]
    + [f"enemy_y_high_{i}" for i in range(5)]
)


@dataclass
class Obj:
    type: int
    x0: float  # screen pixels, box of the object itself
    y0: float
    x1: float
    y1: float
    score: float = 1.0
    slot: int = -1  # oracle only: RAM enemy slot 0-5 (-1 for Mario / detector boxes)

    @property
    def center(self):
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2


def _on_screen(o: Obj) -> bool:
    return o.x1 > 0 and o.x0 < SCREEN_W and o.y1 > 0 and o.y0 < SCREEN_H


def oracle_objects(ram: dict) -> list[Obj]:
    """Objects on screen from one frame's RAM values (dict of ints)."""
    g = lambda k: int(ram.get(k, 0))
    scroll = g("xscrollHi") * 256 + g("xscrollLo")
    out = []
    if g("player_y_screen") == 1:
        mx = g("player_x_posHi") * 256 + g("player_x_posLo") - scroll - X_OFF
        # player_y_pos is the top of a 32-px-tall slot; small Mario is drawn in its lower half
        # (checked on the debug overlay: without the +16 the box sat one sprite above him)
        my = g("player_y_pos") - Y_OFF
        big = g("player_size") == 0
        o = Obj(TYPE_MARIO, mx, my if big else my + 16, mx + 16, my + 32)
        if _on_screen(o):
            out.append(o)
    for i in range(6):
        if g(f"enemy_drawn{15 + i}") == 0:
            continue
        if i < 5 and g(f"enemy_y_high_{i}") != 1:
            continue
        ex = g(f"enemy_pageloc_{i}") * 256 + g(f"enemy_x_{i}") - scroll - X_OFF
        # enemy_y is the top of the 16-px body: no Y_OFF (checked on the SAM debug overlay,
        # where the boxes sat 8 px above the goombas); koopas are 24 px tall, 8 px above that
        ey = g(f"enemy_y_{i}")
        eid = g(f"enemy_id_{i}")
        if i == 5:
            t = TYPE_ITEM
        elif eid in GOOMBA_IDS:
            t = TYPE_GOOMBA
        elif eid in KOOPA_IDS:
            t = TYPE_KOOPA
        elif eid >= SCENERY_MIN_ID:
            t = TYPE_SCENERY
        else:
            t = TYPE_ENEMY
        o = Obj(t, ex, ey - 8 if t == TYPE_KOOPA else ey, ex + 16, ey + 16, slot=i)
        if _on_screen(o):
            out.append(o)
    return out


def order_objects(objs: list[Obj], k: int) -> list[Obj]:
    """Mario first, then the rest by distance to Mario (screen centre if Mario is absent)."""
    mario = [o for o in objs if o.type == TYPE_MARIO][:1]
    ref = mario[0].center if mario else (SCREEN_W / 2, SCREEN_H / 2)
    rest = sorted((o for o in objs if o not in mario),
                  key=lambda o: (o.center[0] - ref[0]) ** 2 + (o.center[1] - ref[1]) ** 2)
    return (mario + rest)[:k]


def crop_box(o: Obj, min_side=48, scale=2.0):
    """Square crop around an object, at least ``min_side`` px and ``scale`` x its size, kept
    inside the screen. Fixed over the 8-frame clip so motion shows up inside the crop."""
    cx, cy = o.center
    side = int(max(min_side, scale * max(o.x1 - o.x0, o.y1 - o.y0)))
    side = min(side, SCREEN_H)
    x0 = int(np.clip(round(cx - side / 2), 0, SCREEN_W - side))
    y0 = int(np.clip(round(cy - side / 2), 0, SCREEN_H - side))
    return x0, y0, side


class ObjectFeaturizer:
    """V-JEPA 2 latent of a clip cropped around each object: rows of [zdim + META]."""

    def __init__(self, enc, k=8, min_side=48, batch=64):
        self.enc, self.k, self.min_side, self.batch = enc, k, min_side, batch

    @torch.no_grad()
    def __call__(self, clips_u8: torch.Tensor, objs_per_clip: list[list[Obj]]) -> torch.Tensor:
        """clips uint8 [B, T, H, W, 3] (any device), objects per clip -> float16 [B, k, zdim+6]."""
        B = len(objs_per_clip)
        out = torch.zeros(B, self.k, self.enc.dim + META, dtype=torch.float16)
        crops, where = [], []
        for b, objs in enumerate(objs_per_clip):
            for j, o in enumerate(order_objects(objs, self.k)):
                x0, y0, side = crop_box(o, self.min_side)
                c = clips_u8[b, :, y0:y0 + side, x0:x0 + side]  # [T, side, side, 3]
                c = c.to(self.enc.device).permute(0, 3, 1, 2).float().div_(255)
                crops.append(F.interpolate(c, size=(64, 64), mode="bilinear", align_corners=False))
                cx, cy = o.center
                out[b, j, self.enc.dim:] = torch.tensor(
                    [cx / SCREEN_W, cy / SCREEN_H, (o.x1 - o.x0) / SCREEN_W, (o.y1 - o.y0) / SCREEN_H, o.type, 1.0])
                where.append((b, j))
        for s in range(0, len(crops), self.batch):
            lat = self.enc.encode_crops(torch.stack(crops[s:s + self.batch]))
            for (b, j), v in zip(where[s:s + self.batch], lat.cpu()):
                out[b, j, :self.enc.dim] = v
        return out


def pack_obs(z_global: torch.Tensor, obj: torch.Tensor | None) -> torch.Tensor:
    """Whole-frame tokens [B, 16, zdim] (+ object rows [B, K, zdim+6]) -> one observation
    tensor. Without objects this is z_global unchanged (the baseline layout)."""
    if obj is None:
        return z_global
    pad = torch.zeros(*z_global.shape[:2], META, dtype=z_global.dtype, device=z_global.device)
    return torch.cat([torch.cat([z_global, pad], -1), obj.to(z_global.device, z_global.dtype)], 1)


class SamDetector:
    """Meta SAM 3 concept segmentation with text prompts (untested until weights are available).

    Each prompt maps to a coarse object type. Frames are upscaled (nearest) before
    detection because NES sprites are only 16 px. Boxes are returned in screen pixels.
    """

    PROMPTS = {"mario": TYPE_MARIO, "goomba": TYPE_GOOMBA, "koopa troopa": TYPE_KOOPA,
               "enemy": TYPE_ENEMY, "mushroom": TYPE_ITEM, "pipe": TYPE_SCENERY}

    def __init__(self, model_id=None, threshold=0.4, upscale=4, device="cuda", prompts=None):
        import os
        model_id = model_id or os.environ.get("MJG_SAM_MODEL") or "facebook/sam3"
        try:
            from transformers import Sam3Model, Sam3Processor
        except ImportError as e:  # pragma: no cover - depends on the installed transformers
            raise ImportError("SAM 3 needs a transformers version with Sam3Model / Sam3Processor") from e
        self.proc = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(device).eval()
        self.threshold, self.up, self.device = threshold, upscale, device
        self.prompts = prompts or self.PROMPTS

    @torch.no_grad()
    def __call__(self, frame_u8: np.ndarray) -> list[Obj]:
        from PIL import Image
        img = Image.fromarray(frame_u8).resize((SCREEN_W * self.up, SCREEN_H * self.up), Image.NEAREST)
        found = []
        for text, typ in self.prompts.items():
            inputs = self.proc(images=img, text=text, return_tensors="pt").to(self.device)
            res = self.proc.post_process_instance_segmentation(
                self.model(**inputs), threshold=self.threshold, mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist())[0]
            for box, score in zip(res["boxes"].tolist(), res["scores"].tolist()):
                x0, y0, x1, y1 = (v / self.up for v in box)
                found.append(Obj(typ, x0, y0, x1, y1, score))
        return _nms(found)


class GroundingDinoDetector:
    """Open-vocabulary boxes from text prompts with Grounding DINO (ungated stand-in for SAM 3;
    same interface: screen frame -> list of Obj)."""

    PROMPTS = {"mario": TYPE_MARIO, "goomba": TYPE_GOOMBA, "koopa troopa": TYPE_KOOPA, "turtle": TYPE_KOOPA,
               "enemy": TYPE_ENEMY, "mushroom": TYPE_ITEM, "pipe": TYPE_SCENERY}

    def __init__(self, model_id=None, threshold=0.3, text_threshold=0.25, upscale=4, device="cuda", prompts=None):
        import os
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        model_id = model_id or os.environ.get("MJG_GDINO_MODEL") or "IDEA-Research/grounding-dino-base"
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        self.threshold, self.text_threshold, self.up, self.device = threshold, text_threshold, upscale, device
        self.prompts = prompts or self.PROMPTS
        self.text = ". ".join(self.prompts) + "."

    @torch.no_grad()
    def __call__(self, frame_u8: np.ndarray) -> list[Obj]:
        from PIL import Image
        img = Image.fromarray(frame_u8).resize((SCREEN_W * self.up, SCREEN_H * self.up), Image.NEAREST)
        inputs = self.proc(images=img, text=self.text, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        res = self.proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=self.threshold, text_threshold=self.text_threshold,
            target_sizes=[(img.height, img.width)])[0]
        labels = res.get("text_labels", res.get("labels"))
        found = []
        for box, score, label in zip(res["boxes"].tolist(), res["scores"].tolist(), labels):
            typ = next((t for name, t in self.prompts.items() if name in str(label)), None)
            if typ is None:
                continue
            x0, y0, x1, y1 = (v / self.up for v in box)
            found.append(Obj(typ, x0, y0, x1, y1, score))
        return _nms(found)


def make_detector(name, model_id=None):
    """"oracle" -> None (boxes come from RAM); "sam" -> SamDetector; "gdino" -> GroundingDinoDetector."""
    if name == "oracle":
        return None
    if name == "sam":
        return SamDetector(model_id)
    if name == "gdino":
        return GroundingDinoDetector(model_id)
    raise ValueError(f"unknown detector {name}")


def _iou(a: Obj, b: Obj) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = ix * iy
    union = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter
    return inter / union if union > 0 else 0.0


def _nms(objs: list[Obj], iou=0.5) -> list[Obj]:
    keep = []
    for o in sorted(objs, key=lambda o: -o.score):
        if all(_iou(o, k) < iou for k in keep):
            keep.append(o)
    return keep


class ObsBuilder:
    """Live agent observations while it plays: whole-frame V-JEPA 2 latents (+ object rows
    when the agent uses objects), normalised with the demo statistics so they match the
    observations the agent was trained on. ``rams``: RAM of each clip's last screen
    (MarioVecEnv.obs_ram), used by the oracle detector."""

    def __init__(self, enc, stats, obj_mode="none", detector="oracle", k=8, sam_model=None):
        from .demos import normalize_obs
        self._norm = normalize_obs
        self.enc, self.use_obj = enc, obj_mode != "none"
        dev = enc.device
        self.mu, self.sd = stats["mu"].to(dev), stats["sd"].to(dev)
        self.omu = stats["obj_mu"].to(dev) if "obj_mu" in stats else None
        self.osd = stats["obj_sd"].to(dev) if "obj_sd" in stats else None
        self.feat = ObjectFeaturizer(enc, k=k) if self.use_obj else None
        self.sam = make_detector(detector, sam_model) if self.use_obj else None  # None = oracle RAM boxes

    @torch.no_grad()
    def __call__(self, clips_np, rams=None):
        clips = torch.from_numpy(clips_np)
        zg = self.enc(clips)
        if not self.use_obj:
            return self._norm(zg, self.mu, self.sd)
        objs = [self.sam(c[-1]) for c in clips_np] if self.sam is not None else [oracle_objects(r) for r in rams]
        return self._norm(pack_obs(zg, self.feat(clips, objs)), self.mu, self.sd, self.omu, self.osd)

    def raw_global(self, x):
        """Un-normalised whole-frame latents of an observation: [B, 16, zdim]."""
        return x[:, :16, :self.mu.shape[0]].float() * self.sd + self.mu
