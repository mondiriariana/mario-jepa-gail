"""Stage 2 (GPU): write the one-sentence traces with a local VLM / LLM, then check them.

Reads <ann>/annotations/*.jsonl from annotate.py and writes <out> (JSON lines) with the
model's JSON, ``accepted`` and the ``problems`` found by checks.validate. --mode image sends
the screens with the prompt (needs a vision-language model); --mode text sends only the RAM
facts, which already hold every exact quantity. --dry-run prints prompts without a model.

python -m reasoning_trace.generate_traces --ann $MJG_ROOT/reasoning_trace/sub-01 \
    --model <local model folder> --mode text --out traces.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reasoning_trace import checks  # noqa: E402
from reasoning_trace import prompts as P  # noqa: E402


def load_records(ann, max_n=None):
    recs = []
    for path in sorted(glob.glob(os.path.join(ann, "annotations", "*.jsonl"))):
        with open(path) as fh:
            recs += [json.loads(l) for l in fh if l.strip()]
        if max_n and len(recs) >= max_n:
            break
    return recs[:max_n] if max_n else recs


class Writer:
    """Thin wrapper over a Hugging Face chat model, in image or text mode."""

    def __init__(self, model_path, mode, max_new_tokens):
        import torch
        from transformers import AutoProcessor
        self.mode, self.max_new_tokens = mode, max_new_tokens
        if mode == "image":
            from transformers import AutoModelForImageTextToText as M
        else:
            from transformers import AutoModelForCausalLM as M
        self.proc = AutoProcessor.from_pretrained(model_path)
        self.model = M.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto").eval()

    def __call__(self, rec, ann_root):
        import torch
        content = []
        if self.mode == "image":
            from PIL import Image
            for p in rec["images"]:
                content.append({"type": "image", "image": Image.open(os.path.join(ann_root, p)).convert("RGB")})
        content.append({"type": "text", "text": P.user_prompt(rec)})
        msgs = [{"role": "system", "content": [{"type": "text", "text": P.SYSTEM}]},
                {"role": "user", "content": content}]
        kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        try:
            inputs = self.proc.apply_chat_template(msgs, enable_thinking=False, **kw)
        except TypeError:
            inputs = self.proc.apply_chat_template(msgs, **kw)
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="output folder of annotate.py")
    ap.add_argument("--model", help="local model folder (a Hugging Face snapshot)")
    ap.add_argument("--mode", choices=["image", "text"], default="text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=None, help="only the first N key frames")
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    recs = load_records(a.ann, a.max)
    if a.dry_run:
        for r in recs[:3]:
            print("=" * 80, f"\n{r['play_id']} frame {r['frame']}\n", P.SYSTEM, "\n---\n", P.user_prompt(r))
        print(f"\n{len(recs)} key frames would be sent")
        return
    write = Writer(a.model, a.mode, a.max_new_tokens)
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as fh:
            done = {(j["play_id"], j["frame"]) for j in map(json.loads, fh)}
    n_ok = n = 0
    with open(a.out, "a") as fh:
        for r in recs:
            if (r["play_id"], r["frame"]) in done:
                continue
            raw = write(r, a.ann)
            out = checks.parse(raw)
            problems = checks.validate(out, r)
            n += 1
            n_ok += not problems
            fh.write(json.dumps({"play_id": r["play_id"], "frame": r["frame"], "trace": out, "raw": raw,
                                 "accepted": not problems, "problems": problems}) + "\n")
            fh.flush()
            if n % 20 == 0:
                print(f"{n} written, {n_ok} accepted", flush=True)
    print(f"done: {n} written, {n_ok} accepted ({n_ok / max(n, 1):.0%})")


if __name__ == "__main__":
    main()
