# Checkpoints

Final weights of every agent in PRELIMINARY_RESULTS.md (seed 0). Each file holds the agent's
state dict, the discriminator (GAIL runs), the resolved config and the latent normalisation
statistics. Load with `mjg.common.load_ckpt(path)`. The frozen V-JEPA 2 encoder is not
included; it is `facebook/vjepa2-vitl-fpc64-256` from Hugging Face.

| File | Agent |
|---|---|
| `bc/bc.pt` | BC (validation-selected, update 4,000) |
| `bcjepa/bc.pt` | BC+JEPA (update 4,500) |
| `bcjepa_v2/bc.pt` | BC+JEPA with the MLP imagination variant (diagnostic only) |
| `gail/gail_last.pt` | GAIL from BC, 500 iterations / 1.02 M env steps |
| `gailjepa/gail_last.pt` | GAIL+JEPA from BC+JEPA, 500 iterations |
| `gail_bc01/gail_last.pt` | GAIL (BC reg. 0.1), 500 iterations |
| `gailjepa_bc01/gail_last.pt` | GAIL+JEPA (BC reg. 0.1), 500 iterations: best agent overall |

Intermediate GAIL checkpoints (every 25 iterations) are on Andromeda under
`$MJG_ROOT/runs/`.
