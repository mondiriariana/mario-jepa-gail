"""Frozen V-JEPA 2 video encoder (HF ``facebook/vjepa2-vitl-fpc64-256``).

A decision step sees a short clip of ``clip_len`` screens (stride ``clip_stride`` raw
frames) ending just before the step. The clip is resized to 256x256 and encoded by the
frozen ViT-L. The token grid [T/2, 16, 16, 1024] is averaged over time and pooled to a
``grid x grid`` spatial map, giving ``z_t`` of shape [grid*grid, 1024]. The backbone is
never trained: it is put in eval mode, its parameters have requires_grad=False and it
runs under torch.no_grad.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


class FrozenVJEPA2(torch.nn.Module):
    def __init__(self, model_id: str = MODEL_ID, grid: int = 4, size: int = 256,
                 dtype=torch.bfloat16, device="cuda", attn="sdpa"):
        super().__init__()
        from transformers import VJEPA2Model
        self.model = VJEPA2Model.from_pretrained(model_id, dtype=dtype, attn_implementation=attn).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.grid, self.size, self.dtype, self.device = grid, size, dtype, device
        self.dim = self.model.config.hidden_size
        self.patch = self.model.config.patch_size
        self.tubelet = self.model.config.tubelet_size

    @torch.no_grad()
    def preprocess(self, clips_u8: torch.Tensor) -> torch.Tensor:
        """uint8 [B, T, H, W, 3] (any device) -> normalized [B, T, 3, size, size]."""
        x = clips_u8.to(self.device, non_blocking=True)
        B, T, H, W, _ = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(B * T, 3, H, W).float().div_(255)
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False,
                          antialias=True)
        x = x.view(B, T, 3, self.size, self.size)
        x = (x - MEAN.to(x.device)) / STD.to(x.device)
        return x.to(self.dtype)

    @torch.no_grad()
    def forward(self, clips_u8: torch.Tensor) -> torch.Tensor:
        """uint8 clips [B, T, H, W, 3] -> pooled latents float16 [B, grid*grid, D]."""
        x = self.preprocess(clips_u8)
        B, T = x.shape[:2]
        h = self.model(pixel_values_videos=x, skip_predictor=True).last_hidden_state
        n = self.size // self.patch
        h = h.view(B, T // self.tubelet, n, n, self.dim).float().mean(1)  # [B, n, n, D]
        h = h.permute(0, 3, 1, 2)
        h = F.adaptive_avg_pool2d(h, self.grid)  # [B, D, g, g]
        return h.flatten(2).transpose(1, 2).to(torch.float16)
