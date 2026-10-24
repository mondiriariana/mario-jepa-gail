#!/bin/bash
#SBATCH -J rt-annotate
#SBATCH --export=ALL
#SBATCH -p short
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 06:00:00
#SBATCH -o logs/rt_annotate_%A_%a.out
#SBATCH --array=0-7
# Stage 1 of reasoning_trace (CPU only): key frames, RAM facts, simulated counterfactuals.
# Submit from the code folder: sbatch reasoning_trace/slurm/annotate.sbatch
# Optional: SUBJECTS="sub-01 ...", OUT=<folder>, MAX_PLAYS=<N per shard>, DEBUG_PNG=<N per play>.
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
S=${SUBJECTS:-sub-01}
$MJG_PY -m reasoning_trace.annotate --subjects $S --out ${OUT:-$MJG_ROOT/reasoning_trace/${S// /_}} \
  --shard $SLURM_ARRAY_TASK_ID --n-shards ${SLURM_ARRAY_TASK_COUNT:-8} \
  ${MAX_PLAYS:+--max-plays $MAX_PLAYS} --debug-png ${DEBUG_PNG:-0}
