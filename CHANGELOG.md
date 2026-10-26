# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: SemVer (pre-1.0). No release has been tagged yet.
Dates below come from `git log`.

## [Unreleased]
### Showcase (2026-10-26): one-click presentation page
- `/` is now the Showcase: **Run the showcase** plays the recorded peak-hour run of all four controllers at 10x (canvas, interpolated between the 5 s frames,
  60 fps; one master clock), loads each controller's stored results below with a dissolve while it plays, and mounts the benchmarks only once all
  metrics are in (a pure, tested schedule). The dashboard moved to `/dashboard`.
- API: `GET /api/showcase/{config,frames,metrics/<controller>}` (validated window, gzip, cached); `GZipMiddleware`.
- **Watch it in SUMO**: `scripts/run_visual_demo.py` runs the real SUMO-GUI for sumo_actuated, fixed_time, model A or model B (the models are fixed server-side, the
  client sends only a controller id); one at a time or all four tiled. Restored the launch control the dashboard dropped; sumo-gui is found by absolute path, a
  crashed window reports its last log line in the UI. `CITYMIND_USE_TRACI=1` makes the code use socket traci (libsumo cannot drive a window).
### Simplified (2026-10-23): one model, one algorithm
- The repository now contains only the approach that beats the SUMO-actuated baseline: parameter-shared IPPO (PPO, split actor/critic
  gradient clipping), a 22-number road-graph-aware observation, a level-based delay reward and a keep/advance action behind the
  safety layer. Removed: centralized PPO, the phase-select MaskablePPO agent, Max-Pressure, the XAI module, three generations of
  observation / reward code, the `version` / `action_mode` switches, 15 one-off scripts and ~20 configs. History stays in git.
- No road or signal ids in the code: neighbours, feeder roads and the "most blocked" entry roads in the report are derived from the
  network and from each run; the observation's slot count is checked against the network instead of assumed.
- One evaluation entry point: `scripts/evaluate.py`; `scripts/report_eval.py` turns many runs into the paired multi-seed report. One training entry point: `scripts/run_ippo_training.py` (+ `resume_` / `finetune_`).
- Configs: `configs/environment.yaml` + `configs/ippo.yaml` (the trained recipe); `*_pending` / `*_finetune` are an experiment.
### Dashboard (2026-10-25): shows the final models
- `api/results.py` reads the 10-seed evaluation (`artifacts/eval10_run`, or `CITYMIND_RESULTS_ROOT`): SUMO actuated, fixed time, model A, model B.
  `/api/metrics/{summary,comparison,ten-seed}` and `/api/replay/*` serve them; the replay animates the recorded 07:00 run of any of the four controllers,
  side by side (baseline vs model) or alone. Dashboard, Comparison and Story show the paired 10-seed verdicts (`scripts/report_eval.py`).
- Removed from the UI: Experiments and XAI pages (they described retired models), the SUMO-GUI launcher card (its script was already retired). The `/api/comparison` router
  (unused by the UI) is gone; the experiments / xai API routers and page sources are still in the tree, unreferenced.
### Results
- Stage 1 (two independently trained models, seeds 42 and 1042) beats `sumo_actuated` on lag per scheduled car by ~26 s on three seeds and on backlog,
  time loss, waiting time, queues and speed; cars never inserted stayed ~100 above the baseline on those seeds.
- Stage 2 (fine-tune with cars waiting outside the network in the demand input; 10 updates for seed 42, 20 for seed 1042) gives, over 10 simulation seeds
  (`docs/TEST_REPORT_10_SEEDS.md`): lag -41.4 s (A) / -36.9 s (B) on 10 of 10 seeds, trips completed +258 / +206, waiting time -41% / -40%,
  11-12 of 14 measures beaten, none worse. Cars never inserted is level (501 / 481 vs 503) and teleports are too noisy to call.
### Added
- Final observation (22 numbers: active phase, elapsed vs own min/max green, per-phase demand incl. unsignalised feeder roads and
  cars waiting outside, queue, entry backlog, road-connected neighbour queues), basis `artifacts/eda/observation_relevance/REPORT.md`.
- Level-based delay reward; `split_clip` (separate actor / critic gradient clipping); `scripts/evaluate.py`,
  `scripts/finetune_ippo.py` (stage 2), `scripts/resume_ippo_training.py`, `scripts/summarize_training_logs.py`.
- Repository tooling: Apache-2.0 LICENSE and NOTICE, CI workflow, Dockerfile/compose, Makefile,
  pre-commit, `.env.example`, `requirements.lock`, SECURITY/CONTRIBUTING, issue/PR templates,
  `docs/MODELS.md` + `artifacts/models/MANIFEST.json`, tracked `configs/sumo/InTAS_train.sumocfg`.
- Broader ruff rule set and a non-strict mypy config (`pyproject.toml`).
### Changed
- `scripts/setup_sumo.sh` pins InTAS to a commit, is idempotent/portable, and writes `.env.sumo`.
- README corrected (18-dim local observation, MaskablePPO, `Discrete(6)`, six controllers).
- `werkzeug>=3.1.9` floor (CVE in `safe_join`); wheel now packages `src` and `api`.
- `Less_priority.md` moved to `docs/archive/` with a historical banner.

## [0.1.0] - development history to 2026-09-30 (untagged)
### Results (negative)
- **No RL controller beat SUMO's calibrated actuated baseline** on completed trips or mean trip timeloss.
  The gate-G1 phase-selection retrain (`ippo_phase_select`) improved on its first attempt
  (+22% trips, -61% teleports) and reached the second-best mean trip timeloss, but still had
  ~90x the baselines' teleports and fewer completed trips.
### 2026-09-27 to 2026-09-30
- Phase-selection action space with MaskablePPO (`Discrete(6)`, action masks); reward fix
  ("neglected pressure" term, restored waiting delta, lower switching penalty); resumable training;
  honest frontend narrative with starvation warnings; 5-minute replay window; metrics summary
  ranks by trip metrics instead of waiting time.
### 2026-09-23/24
- Fixed a ~5x departure/arrival undercount in all controllers; added trip-level metrics
  (tripinfo), backlog and teleport counts; peak-hour (07:00-08:00) evaluation window.
### 2026-09-20 to 2026-09-21
- Safety-layer mechanics repair; true fixed-time baseline (the earlier "fixed-time" was SUMO's actuated
  control); parameter-shared IPPO with neighbour visibility and spillback-aware reward; XAI for IPPO;
  parallel multi-instance training; README rewritten for the PPO/IPPO comparison.
### 2026-09-18/19
- IPPO architecture introduced to address centralized PPO's credit-assignment failure; IPPO
  evaluation wired in; live multi-controller SUMO-GUI comparison; UI/UX and accessibility fixes;
  backend hardening.
### 2026-09-18 (initial)
- Initial commit: centralized RL traffic-signal control platform on InTAS/SUMO with FastAPI + React
  dashboard; security fixes closing unauthenticated file-read and process-spawn endpoints;
  Max-Pressure baseline hardening; real per-frame signal-state replay.
