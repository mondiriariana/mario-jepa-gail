#!/bin/bash
#SBATCH -J mjg-bc
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o logs/bc_%j.out
# usage: sbatch slurm/train_bc.sbatch <run_name> [config overrides...]
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
RUN=$1; shift
$MJG_PY scripts/train_bc.py --run $MJG_ROOT/runs/$RUN "$@"
