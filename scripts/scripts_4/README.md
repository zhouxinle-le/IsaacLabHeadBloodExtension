# Dreamer replay evaluation

This folder contains replay/evaluation utilities for State Dreamer and Vision
Wrist Dreamer runs. The main entry point is:

```bash
conda run -n isaacsim-4.2 python scripts/scripts_4/evaluate_state_vision_dreamer_replay.py --help
```

Default runs:

- `state_dreamer_run0`: `logs/r2dreamer/ur3_blood_pipe_state_dreamer/seed_0_800k`
- `state_dreamer_run1`: `logs/r2dreamer/ur3_blood_pipe_state_dreamer/seed_1_800k`
- `vision_dreamer_run0`: `logs/r2dreamer/ur3_blood_pipe_vision_wrist_dreamer/seed_0_800k`
- `vision_dreamer_run1`: `logs/r2dreamer/ur3_blood_pipe_vision_wrist_dreamer/2026-05-12_21-14-29_seed_0_600k`

The script evaluates one run per worker process to avoid reusing an Isaac
SimulationContext across runs. Outputs are written under
`scripts/scripts_4/results/{timestamp}_state_vision_dreamer_replay/`.

Small smoke run example:

```bash
conda run -n isaacsim-4.2 python scripts/scripts_4/evaluate_state_vision_dreamer_replay.py \
  --methods state_dreamer \
  --run-indices 0 \
  --eval-episodes 2 \
  --headless
```

Full default evaluation:

```bash
conda run -n isaacsim-4.2 python scripts/scripts_4/evaluate_state_vision_dreamer_replay.py \
  --eval-episodes 100 \
  --headless
```
