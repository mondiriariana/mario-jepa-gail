"""Measure frozen V-JEPA 2 throughput (clips/s) for a few clip lengths and batch sizes,
to size the feature-extraction and online-rollout jobs before launching them."""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mjg.encoder import FrozenVJEPA2  # noqa: E402

attn = sys.argv[1] if len(sys.argv) > 1 else "sdpa"
enc = FrozenVJEPA2(attn=attn)
print(torch.cuda.get_device_name(), attn, "dim", enc.dim, "patch", enc.patch, "tubelet", enc.tubelet, flush=True)
x = torch.randint(0, 255, (16, 8, 224, 240, 3), dtype=torch.uint8, device="cuda")
torch.cuda.synchronize(); t0 = time.time()
for _ in range(5):
    enc.preprocess(x)
torch.cuda.synchronize(); print("preprocess only", 16 * 5 / (time.time() - t0), "clips/s", flush=True)
for T in (8,):
    for B in (16, 32, 64):
        x = torch.randint(0, 255, (B, T, 224, 240, 3), dtype=torch.uint8, device="cuda")
        z = enc(x)
        torch.cuda.synchronize()
        t0 = time.time()
        n = 5
        for _ in range(n):
            z = enc(x)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / n
        print(f"T={T} B={B} out={tuple(z.shape)} {B/dt:.1f} clips/s  mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
