# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: SemVer (pre-1.0). No release has been tagged yet.
Dates below come from `git log`.

## [Unreleased]
### Added
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
