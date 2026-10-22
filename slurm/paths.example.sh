# Site-specific paths. Copy to slurm/paths.local.sh (git-ignored) and fill in.
export MJG_ROOT=/path/to/project/space/mario-jepa-gail       # runs/, features/, results/ on the cluster
export MJG_PY=/path/to/envs/mjg/bin/python                    # venv with torch, transformers, stable-retro
export MJG_DATA=/path/to/cneuromod/data/mario                 # .bk2 replays (+ mp4) of the mario dataset
export MJG_ANNOT=/path/to/cneuromod/data/mario.annotations    # annotated events
export MJG_INTEGRATION=/path/to/cneuromod/mario.stimuli       # folder holding SuperMarioBros-Nes/ (ROM + states)
export HF_HOME=/path/to/huggingface                           # cache with facebook/vjepa2-vitl-fpc64-256
export MJG_SSH_HOST=andromeda                                 # ssh alias used by scripts/sync_to_cluster.sh
