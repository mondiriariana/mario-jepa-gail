#!/bin/bash
#SBATCH -J mjg-gail
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o logs/gail_%j.out
# usage: sbatch slurm/train_gail.sbatch <run_name> <init_ckpt relative to MJG_ROOT> [overrides...]
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
RUN=$1; INIT=$2; shift 2
$MJG_PY scripts/train_gail.py --run $MJG_ROOT/runs/$RUN --init $MJG_ROOT/$INIT "$@"
