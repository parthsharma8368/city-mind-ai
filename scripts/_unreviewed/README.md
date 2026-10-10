# scripts/_unreviewed

Scripts parked here have NOT passed review and must not be run for, or cited in,
any result. Do not promote or commit their outputs.

## generate_training_video.py

Renders an "explainer" animation whose numbers are synthetic or stale
(hard-coded pre-2026-09-21 midnight-window "Empirical Results", sine-wave
observation bars, a fake 40M-step counter, "12-dimensional" observation where
the environment is 18-dimensional). It contradicts the current finding that no
RL controller beats the SUMO-actuated baseline (see `artifacts/metrics/drain_scores.md`).
It refuses to run without `--allow-synthetic`, and the untracked outputs
`artifacts/rl_training_explanation.webp`, `artifacts/rl_training_visualizer.html`,
`artifacts/rl_training_visualizer.png`, `artifacts/gemini_nano_banana_training_video.webp`
should not be published either (the HTML embeds the same stale table).

To make it publishable: drive the table from `artifacts/metrics/drain_scores.json`
(lag per scheduled car, due-completion ratio), drop the "winner" highlight, fix the
observation size, and label the remaining animation "illustrative".
