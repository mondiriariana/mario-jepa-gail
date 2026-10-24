#!/bin/bash
#SBATCH -J rt-traces
#SBATCH --export=ALL
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o logs/rt_traces_%j.out
# Stage 2 of reasoning_trace (GPU): one-sentence traces from a local model, checked automatically.
# Submit from the code folder: MODEL=<local model folder> sbatch reasoning_trace/slurm/traces.sbatch
# Needs a recent transformers (the SAM env, $MJG_SAM_PY, has 5.5 + torch 2.7 cu128).
# Optional: ANN=<annotate.py output>, MODE=image|text, MAX=<N key frames>.
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
: "${MODEL:?set MODEL to a local model folder}"
ANN=${ANN:-$MJG_ROOT/reasoning_trace/sub-01}
${PY:-$MJG_SAM_PY} -m reasoning_trace.generate_traces --ann $ANN --model $MODEL --mode ${MODE:-text} \
  --out $ANN/traces_$(basename $MODEL)_${MODE:-text}.jsonl ${MAX:+--max $MAX}
