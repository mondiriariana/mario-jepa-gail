# Reasoning traces for Mario

Grounded reasoning traces for human Super Mario Bros play, following **nuReasoning**
(Huang et al., arXiv 2605.31572), which builds spatial / decision / counterfactual reasoning
annotations for driving. The structure is theirs; what changes for Mario is that the facts come
exactly from game RAM, and counterfactuals are **simulated in the emulator** instead of
judged by a VLM.

These are explanations consistent with what the player did and what followed (hindsight), not
the player's actual thoughts.

## Pipeline

| Stage | nuReasoning | Here | File |
|---|---|---|---|
| Key frames | VLM scores clips 1-10, humans pick the key frame | rules on RAM + buttons, same 1-10 behaviour-centric rubric | `keyframes.py` |
| Spatial facts | 2D detections matched to 3D boxes, map | Mario state, objects relative to Mario, motion, time to contact, terrain ahead, all from RAM | `facts.py` |
| Decision | 9 longitudinal x 7 lateral actions, must match the real trajectory | horizontal (run right / walk right / stop / back off left) x vertical (no jump / short hop / full jump), read off the player's buttons | `schema.py` |
| Counterfactuals | VLM labels alternatives safe / suboptimal / unsafe (76% agreement) | 8 alternatives replayed for 1 s from a saved emulator state; label from the outcome | `counterfactuals.py` |
| Trace | Gemini 3.1 Pro, one-sentence causal chain, priority hierarchy | local model, same prompt structure (`prompts.py`), priority: survival > obstacles > opportunity > progress | `generate_traces.py` |
| Verification | two-stage human review | automatic checks (`checks.py`): decision = player's, components exist in RAM, risks = simulated; then a human read of a sample | `checks.py` |

## Running (on Andromeda, from the code folder; never on the login node)

```bash
# stage 1, CPU: 8 shards over sub-01 -> $MJG_ROOT/reasoning_trace/sub-01/
sbatch reasoning_trace/slurm/annotate.sbatch
# a quick look first: 1 play per shard, with debug overlays
MAX_PLAYS=1 DEBUG_PNG=3 sbatch --array=0 reasoning_trace/slurm/annotate.sbatch

# stage 2, GPU: check the prompts without a model, then generate
$MJG_SAM_PY -m reasoning_trace.generate_traces --ann <stage-1 folder> --out x.jsonl --dry-run   # via srun
MODEL=<local model folder> MODE=text sbatch reasoning_trace/slurm/traces.sbatch
```

Output of stage 1, per key frame (`annotations/<play>.jsonl`): `facts`, `decision`,
`human_outcome`, `counterfactuals` (each with outcome and risk), `human_replay_matches`
(the player's own buttons replayed from the saved state reproduce the recording), and the
screens in `images/`. `summary_<shard>.json` counts key frames, risks and the determinism checks.

## To verify before relying on it

- `facts.terrain_ahead` reads the level's metatile buffer (RAM 0x0500-0x069F) using the layout
  from SMB disassemblies; check it on the `--debug-png` overlays (red = solid tiles).
- Jump-height thresholds (`SHORT_HOP_MAX`, the 6 / 30-frame programs) and the
  suboptimal rule in `schema.risk_label` are first guesses.
- The trace model: a local model must fit on one GPU; with `--mode image` it must be a
  vision-language model.

## Using the output for training (not implemented yet)

Like nuVLA, the annotations can supervise the agent without generating text at play time:
auxiliary heads predicting the decision, the critical object and each alternative's risk;
and the counterfactual rollouts (same state, different actions, real futures) as extra
training data for the JEPA predictor.
