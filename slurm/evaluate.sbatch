#!/bin/bash
#SBATCH -J mjg-eval
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o logs/eval_%j.out
# usage: sbatch slurm/evaluate.sbatch <out_name> NAME=ckpt [NAME=ckpt ...]  (ckpt relative to MJG_ROOT)
source "$SLURM_SUBMIT_DIR/slurm/env.sh"
OUT=$1; shift
CK=(); for s in "$@"; do CK+=("${s%%=*}=$MJG_ROOT/${s#*=}"); done
$MJG_PY scripts/evaluate.py --out $MJG_ROOT/results/$OUT --ckpt "${CK[@]}" ${EVAL_ARGS:-}
