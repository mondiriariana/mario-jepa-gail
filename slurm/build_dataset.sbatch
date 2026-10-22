#!/bin/bash
#SBATCH -J mjg-build
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o logs/build_%A_%a.out
#SBATCH --array=0-3
# Replay + encode human plays; 4 shards in parallel, one GPU each.
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
$MJG_PY scripts/build_dataset.py --out $MJG_ROOT/features/vjepa2_l_c8s2_g4 \
  --subjects ${SUBJECTS:-sub-01} --shard $SLURM_ARRAY_TASK_ID --nshards 4 --workers 6
