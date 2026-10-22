# Data

## Source

CNeuroMod `mario` (Paugam et al., 2025, bioRxiv 10.1101/2025.11.28.691119), `algonauts2027` branch
of the public datalad repository (commit b778ff6). Each *play* (repetition) is one attempt sequence at
one level, stored as

- `sub-XX_ses-YYY_task-mario_level-wXlY_rep-NNN.bk2`: stable-retro replay (zip): start savestate
  `Core.bin` plus the controller log (`Input Log.txt`, one line per 60 Hz frame);
- `..._recording.mp4`: 240×224 h264 video at 60 fps, with audio;
- `mario.annotations/.../*_desc-annotated_events.tsv`: per-run events (button intervals, kills,
  power-ups, deaths, level boundaries) with emulator frame indices.

The game integration (`SuperMarioBros-Nes`: ROM, `data.json` RAM map, 32 level states) is from
`courtois-neuromod/mario.stimuli` (commit e3ecccf), whose annexed files we fetched from CNeuroMod's
public CONP mirror. ROM sha1 (without the 16-byte iNES header) =
`facee9c577a5262dbe33ac4930bb0b58c8c037f7`, as listed in the integration's `rom.sha`.

## What this prototype uses

Subject **sub-01**, all 797 plays available on 2026-10-21 (21 sessions, 22 levels, 13.7 h of play,
739,002 decision steps, of which 465,469 are controllable). Numbers from
`results/dataset_summary.json`.

| Split | Sessions | Plays | Controllable steps | Used for |
|---|---|---|---|---|
| train | the other 14 sessions | 529 | 309,127 | fitting |
| validation | ses-003, 009, 015, 021 | 184 | 106,408 | checkpoint choice, training curves |
| held-out test | ses-006, 012, 018 | 84 | 49,934 | final numbers only |

Splits are by whole session, so validation and test plays come from days never seen in training
(same 22 levels). Nothing is selected on the test sessions.

## Alignment checks (`results/alignment_check.json`, scripts/check_alignment.py)

On `sub-01_ses-001_..._w1l1_rep-000`:

- Replaying the `.bk2` in stable-retro yields 4,831 frames; the mp4 has 4,830 frames, and the
  annotation row spans frames 0–4831.
- **Replay screen k = video frame k.** Mean absolute pixel error between replay and video is lowest at
  offset 0 with no crop (6.5 grey levels, which is h264 compression; offsets −1 / +1 give 7.8 / 9.8).
  stable-retro's screen is already the 224×240 overscan-cropped image the videos show.
- **Inputs.** The input log has one leading line more than the replay: log line k+1 = the input applied
  on replay frame k (100% agreement on all 8 buttons at that offset, 97.8–100% at offset 0).
- **Annotations.** Button-interval events (RIGHT, LEFT, A, B) cover the replay's pressed frames
  93–98% of the time, and every interval's `frame_start` is one frame before the replay's first
  pressed frame, i.e. annotation frame indices count log lines.

We therefore take everything (screens, inputs, RAM) from the emulator replay itself, which keeps human
and agent observations pixel-identical in format: the agent's screens come from the same emulator.

## Steps, clips, actions

- Decision step t starts at frame f = 4t (15 Hz). Observation = 8 screens f−15, f−13, …, f−1.
- Action label = the human's buttons on frame f, projected onto 14 combos (NOOP; RIGHT/LEFT with
  any of A, B; A; B; A+B; DOWN; UP). 99.05% of controllable steps are represented exactly (the rest
  had e.g. LEFT+RIGHT together, or UP/DOWN with a direction). On 84.9% of controllable steps the human
  holds the same combo for all 4 frames (92.3% of frames match the step's label).
- Controllable steps: RAM `player_state ∈ {8, 9, 10, 12}` and `oper_mode = 1`. 63% of steps qualify;
  the rest are level intros, death animations, pipe transitions and flagpole sequences.
- Human sub-01's action use on controllable steps: RIGHT+B 36.5%, RIGHT+A+B 22.8%, B 16.6%, NOOP 8.7%,
  LEFT+B 8.4%, A+B 3.3%, LEFT+A+B 2.5%; everything else < 1% each. This player almost always holds
  the run button.

## Not used here

- Other subjects (02, 03, 05, 06): their files were still downloading. The pipeline takes
  `--subjects` and the build skips already-encoded plays.
- The mp4 audio track, fMRI (`mario.fmriprep`), and the not-yet-public `_variables.json` /
  `_lowlevel.npy` files. Symbolic state comes from emulator RAM instead (`data.json` map; per-step RAM
  is stored in every `<play>.meta.npz`).
