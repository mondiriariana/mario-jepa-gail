#!/bin/bash
#SBATCH -J mjg-gail2
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=110G
#SBATCH -t 06:00:00
#SBATCH -o logs/gailpair_%j.out
# Runs the GAIL-only and GAIL+JEPA arms side by side on one GPU (same seed, same config).
# usage: [SUFFIX=_x] sbatch slurm/train_gail_pair.sbatch [overrides...]   (runs/gail$SUFFIX, runs/gailjepa$SUFFIX)
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
$MJG_PY scripts/train_gail.py --run $MJG_ROOT/runs/gail${SUFFIX:-} --init $MJG_ROOT/runs/bc/bc.pt "$@" \
  > $MJG_ROOT/logs/gail${SUFFIX:-}_${SLURM_JOB_ID}.out 2>&1 &
sleep 240  # stagger demo loading so the two processes' peak RAM does not overlap
$MJG_PY scripts/train_gail.py --run $MJG_ROOT/runs/gailjepa${SUFFIX:-} --init $MJG_ROOT/runs/bcjepa/bc.pt "$@" \
  > $MJG_ROOT/logs/gailjepa${SUFFIX:-}_${SLURM_JOB_ID}.out 2>&1 &
wait
