#!/bin/bash
# Copy the code (tracked + untracked-not-ignored files except results/, plus slurm/paths.local.sh) to the
# cluster code folder $MJG_ROOT/code, over the ssh alias $MJG_SSH_HOST (both from slurm/paths.local.sh).
set -e
cd "$(dirname "$0")/.."
[ -f slurm/paths.local.sh ] && source slurm/paths.local.sh
DEST=${DEST:-${MJG_ROOT:?set MJG_ROOT in slurm/paths.local.sh}/code}
HOST=${MJG_SSH_HOST:-andromeda}
{ git ls-files -co --exclude-standard | grep -v '^results/'; [ -f slurm/paths.local.sh ] && echo slurm/paths.local.sh; } \
  | tar czf - -T - | ssh "$HOST" "mkdir -p $DEST/logs && tar xzf - -C $DEST"
