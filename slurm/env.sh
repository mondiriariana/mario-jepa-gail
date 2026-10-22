# Sourced by every Slurm script (jobs are submitted from the code folder on the cluster).
# Site paths come from slurm/paths.local.sh, which is not in git: copy slurm/paths.example.sh.
_MJG_SLURM_DIR="${SLURM_SUBMIT_DIR:-$PWD}/slurm"
[ -f "$_MJG_SLURM_DIR/paths.local.sh" ] && source "$_MJG_SLURM_DIR/paths.local.sh"
: "${MJG_ROOT:?set MJG_ROOT in slurm/paths.local.sh}"
export MJG_CODE=${MJG_CODE:-$MJG_ROOT/code}
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
mkdir -p "$MJG_CODE/logs"
cd "$MJG_CODE"
