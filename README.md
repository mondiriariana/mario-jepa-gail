# mario-jepa-gail

A prototype Super Mario Bros agent that learns to play **like a specific human** (adversarial
imitation: GAIL with PPO, behavioural-cloning warm start) while learning **what its actions will do**
(JEPA-style action-conditioned latent prediction), on top of a frozen, pretrained
[V-JEPA 2](https://github.com/facebookresearch/vjepa2) video encoder. The human demonstrations are
CNeuroMod's `mario` dataset (Paugam et al., 2025, bioRxiv 10.1101/2025.11.28.691119).

- **Architecture, and which parameters each loss updates:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Data, alignment checks, action space:** [docs/DATA.md](docs/DATA.md)
- **Results so far:** [PRELIMINARY_RESULTS.md](PRELIMINARY_RESULTS.md)

## Layout

| Path | What |
|---|---|
| `mjg/emulator.py` | stable-retro + CNeuroMod integration, `.bk2` parsing and replay |
| `mjg/actions.py` | 14-combo action space, frame skip 4, projection of human button states |
| `mjg/data.py`, `mjg/demos.py` | step/clip layout, held-out split by session, batch samplers |
| `mjg/encoder.py` | frozen V-JEPA 2 ViT-L clip encoder → `z_t` [16 × 1024] |
| `mjg/models.py` | adapter, latent predictor, policy/value, discriminator |
| `mjg/vec_env.py` | one emulator per subprocess, 15 Hz decision steps |
| `mjg/offline_eval.py` | held-out action agreement and latent prediction metrics |
| `scripts/check_alignment.py` | replay-vs-video and inputs-vs-annotation checks |
| `scripts/build_dataset.py` | replay human plays and encode every decision step |
| `scripts/train_bc.py` | BC (+ λ·JEPA) warm start |
| `scripts/train_gail.py` | GAIL + PPO in the emulator, + λ·JEPA |
| `scripts/evaluate.py` | play the game, compare with held-out humans, save videos and `.bk2`s (not committed) |
| `scripts/make_figures.py` | figures in `results/figures/` |
| `slurm/` | job scripts for BC Andromeda |
| `configs/default.yaml` | every hyperparameter |
| `results/` | metrics (JSON), figures, agent trajectories, final checkpoints (`results/checkpoints/`); gameplay videos and `.bk2` replays are kept locally |

## Local setup (laptop)

For editing, analysis, figures and loading checkpoints on CPU:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements-local.txt
.venv/bin/python scripts/make_figures.py --out results/figures --eval results/eval_all.json --probes results/probes.json
```

Training and evaluation need a GPU, the emulator ROM and the encoded features, so they run on the
cluster. `scripts/sync_to_cluster.sh` copies the working tree to Andromeda.

## Environment (BC Andromeda)

```bash
E=/m31/modules/miniconda/3/envs/m31_pytorch          # torch 2.5.1 + CUDA 12.6, Python 3.13
$E/bin/python -m venv --system-site-packages /scratch/$USER/envs/mjg
/scratch/$USER/envs/mjg/bin/pip install "transformers==4.57.1" "huggingface_hub<1.0" safetensors av \
    matplotlib pandas scipy scikit-learn pyyaml tqdm einops stable-retro==1.0.1
```

On Andromeda, download wheels on the login node (`pip download`, one stream, `nice -n 19`) and
install them from a Slurm job; compute nodes have very slow internet and the login node enforces a
CPU policy. V-JEPA 2 weights: `huggingface_hub.snapshot_download("facebook/vjepa2-vitl-fpc64-256")`
into `$HF_HOME`.

**Game integration.** Clone `https://github.com/courtois-neuromod/mario.stimuli` and fetch the
annexed `SuperMarioBros-Nes/rom.nes` and `Level*.state` files (git-annex / datalad, or the CONP
`httpalso` mirror listed in the repo's `git-annex:remote.log`). The ROM is Nintendo's copyrighted
game, shared by CNeuroMod for research; it is **not** in this repository. Check:
`tail -c +17 rom.nes | sha1sum` = `facee9c577a5262dbe33ac4930bb0b58c8c037f7`. Point
`MJG_INTEGRATION` at the clone's root.

## Reproduce

Copy `slurm/paths.example.sh` to `slurm/paths.local.sh` (git-ignored) and fill in your cluster
paths; `scripts/sync_to_cluster.sh` copies it to the cluster with the code, and every Slurm script
reads it. Submit jobs from the code folder on the cluster (`$MJG_ROOT/code`); logs go to `logs/`
there. All steps run under Slurm.

```bash
sbatch slurm/build_dataset.sbatch                                 # replay + encode human plays (4 GPU shards)
sbatch slurm/train_bc.sbatch bc     model.use_jepa=false          # BC warm start
sbatch slurm/train_bc.sbatch bcjepa model.use_jepa=true           # BC + lambda*JEPA warm start
sbatch slurm/train_gail_pair.sbatch                               # GAIL (from bc) and GAIL+JEPA (from bcjepa) on one GPU
SUFFIX=_bc01 sbatch --export=ALL,SUFFIX=_bc01 slurm/train_gail_pair.sbatch gail.bc_coef=0.1   # same, with BC regulariser
sbatch slurm/evaluate.sbatch eval_final BC=runs/bc/bc.pt BC+JEPA=runs/bcjepa/bc.pt \
    GAIL=runs/gail/gail_last.pt GAIL+JEPA=runs/gailjepa/gail_last.pt
python scripts/probe_representations.py --out results/probes.json BC=runs/bc/bc.pt ...      # CPU is fine
python scripts/inspect_imagination.py --device cpu BC+JEPA=runs/bcjepa/bc.pt ...
python scripts/make_figures.py --out results/figures --eval results/eval_final/eval.json \
    --bc BC=runs/bc BC+JEPA=runs/bcjepa --gail GAIL=runs/gail GAIL+JEPA=runs/gailjepa
```

Seeds are fixed in the config (`seed: 0`); every run writes its resolved config, logs and
checkpoints under `runs/<name>/`.
