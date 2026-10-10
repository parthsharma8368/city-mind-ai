# CityMind AI - Reinforcement Learning for Traffic Signal Control

[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml) <!-- TODO: replace OWNER/REPO -->
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

CityMind AI is a reproducible research and engineering platform evaluating whether deep reinforcement learning (RL) can optimize network-wide traffic signals across a real-world city grid (the Ingolstadt InTAS scenario in Eclipse SUMO), compared with classical controllers. **Headline finding: no RL controller here beats SUMO's calibrated actuated control** (reported honestly below).

![IPPO policy replaying on the InTAS Ingolstadt network](artifacts/traffic_simulation_showcase.gif)

## Core research question

> Can deep-RL traffic-signal control reduce congestion and travel delay compared to fixed-time, actuated and Max-Pressure control under identical simulation conditions, and does giving each intersection its own agent and reward (instead of one policy jointly controlling all of them) change the answer?

Three RL architectures were trained and compared against three classical baselines (six controllers in total) on the same 98-intersection network, same seeds, same peak-hour demand:

| Controller id | Kind | Action space |
|---|---|---|
| `fixed_time` | baseline: InTAS static programs | - |
| `sumo_actuated` | baseline: SUMO's calibrated vehicle-actuated control | - |
| `max_pressure` | baseline: queue-pressure differential | keep / next phase |
| `ppo` | RL: centralized SB3 `PPO`, one policy for the whole network | `MultiBinary(98)` |
| `ippo` | RL: parameter-shared SB3 `PPO`, one shared policy invoked per intersection with its own local reward | `Discrete(2)` (extend / advance) |
| `ippo_phase_select` | RL: parameter-shared sb3-contrib `MaskablePPO` | `Discrete(6)`, action-masked per intersection |

IPPO was built after the centralized design was diagnosed with a credit-assignment failure; phase selection was built after the keep/advance action space was identified as the limiting factor (see Results).

---

## Results (07:00-08:00 peak demand, mean ± std over 3 seeds)

Two corrections behind this table, both load-bearing: earlier results were measured on a **midnight window** (1,486 departures/hour vs. peak's 14,500+) against a baseline mislabeled "fixed-time" that was actually **SUMO's own calibrated vehicle-actuated control** (`sumo_actuated` below) — a materially higher bar than a real fixed timer (`fixed_time` below, InTAS's own static programs). Separately, every controller's per-step departure/arrival counters were undercounting by roughly 5× (sampled once every 5 simulation seconds using TraCI's since-last-call counters, without accumulating the 4 seconds in between) — fixed alongside adding trip-level metrics (`trips_completed`, `mean_trip_timeloss`, from SUMO's own `--tripinfo-output`) and backlog/teleport counts, which is what surfaced the finding below in the first place.

| Controller | Trips Completed | Mean TimeLoss (s) | Backlog | Teleports | Mean Wait (s) | Switches |
|---|---|---|---|---|---|---|
| **SUMO-Actuated (calibrated)** | **11,600** | **284.4 ± 1.3** | **3,257** | 7.7 | 6.06 ± 0.06 | 0 |
| Fixed-Time (static) | 11,454 | 301.8 ± 5.4 | 3,472 | 7.7 | 6.97 ± 0.18 | 0 |
| Max-Pressure | 10,408 | 342.9 ± 4.9 | 3,755 | 50.3 | 12.62 ± 0.21 | ~11,962 |
| PPO (Centralized) | 9,546 | 354.7 ± 0.6 | 4,112 | 19.0 | 7.61 ± 0.25 | ~31,265 |
| IPPO (Parameter-Shared) | 9,520 | 347.8 ± 2.1 | 4,086 | 10.0 | **5.26 ± 0.13** | ~40,342 |
| IPPO Phase-Select (`ippo_phase_select`, see Update below) | 9,441 ± 42 | 291.5 ± 1.5 | 4,420 ± 30 | 712.7 ± 7 | 34.10 ± 0.26 | — |

**None of the RL variants (PPO, IPPO, IPPO phase-select) beats the best baseline** — on trips completed or mean trip timeloss (the two metrics sourced from completed trips, not in-network sampling), the first two rank at or near last. IPPO's mean waiting time *looks* best (5.26s vs. actuated's 6.06s), but that number is measured only over vehicles currently in the network — and IPPO completes 18% fewer trips than actuated control while leaving 26% more vehicles stuck in the network at episode end (backlog). That combination is starvation, not faster service: some vehicles get through quickly while a growing pool never clears. The mechanism is visible in the switch count — IPPO's shared policy changes phase roughly 40,000 times across the hour (98 agents × 720 decisions ≈ 57% of all opportunities), well past anything a real signal engineer would run, which resets in-network waiting-time counters without moving the queue that matters.

The root cause is the action space, not the algorithm: every controller here (RL and baseline alike) can only keep or advance to the *next* phase in a fixed cycle — never jump straight to whichever approach is actually congested. Three independent training runs (centralized PPO, IPPO, IPPO with neighbor observations) converged to the same result under this constraint.

**Update — phase-selection, final result:** a phase-selection action space (`Discrete(6)`, action-masked per intersection via MaskablePPO, controller id `ippo_phase_select`) was built and trained to address the action-space limitation above. The first attempt failed worse than every baseline (severe gridlock — see the full diagnosis on the app's **Story** page or `PROJECT_RESEARCH_AND_VISION.md`), traced to a reward that didn't credit the agent for relieving congestion it wasn't currently serving. That reward was fixed (a "neglected pressure" term, restored waiting-delta, a 6× cut to the switching penalty) and the model was retrained for the full 100 episodes (interrupted twice by infrastructure issues along the way, resumed cleanly from checkpoint both times — not a policy failure).

The fix worked, measurably and reproducibly (±0.4% variance across 3 seeds): trips completed rose 22% and teleports fell 61% versus the first attempt, and mean trip timeloss (291.5s) is now the second-best of all six controllers, beaten only by the calibrated actuated baseline. **It still doesn't beat the baselines** — teleports remain ~90× every baseline's, and trips completed/backlog still trail every other controller. The honest state of this repo is: two real measurement bugs found and fixed, a harder action-space redesign built and proven to work mechanically, a reward flaw diagnosed and corrected with a real, reproducible improvement — and still no RL controller here beats calibrated actuated signal control. Reported directly, not around.

---

## Architecture and formulation

- **Simulator**: Eclipse SUMO + TraCI. **Scenario**: InTAS (Ingolstadt), 98 signalized intersections.
- **Actions**: keep/advance-phase (`Discrete(2)` per intersection, `MultiBinary(98)` for centralized PPO) or direct phase selection (`Discrete(6)` with action masking) for `ippo_phase_select`.
- **Safety layer**: shared by every controller; enforces minimum/maximum green and yellow/all-red clearance.
- **Observations**:
  - Centralized PPO: one large vector concatenating every intersection's lane queues, densities, speeds and phase state, plus global stats.
  - IPPO / phase-select: a fixed-size local vector per intersection of `12 + MAX_GREEN_PHASES = 18` dimensions (phase timing, own-lane aggregates, downstream-neighbour signal, network totals), so one shared network serves every intersection despite TLS ranging over 4-19 lanes and 3-11 phases (`src/environment/local_observation.py`).
- **Training**: PettingZoo `ParallelEnv` + SuperSuit so 98 agents share one set of weights. `ppo` and `ippo` use SB3 `PPO`; `ippo_phase_select` uses sb3-contrib `MaskablePPO`.
- **Evaluation**: 3 seeds (42, 123, 777), 07:00-08:00 peak window, trip-level metrics from SUMO's tripinfo output.

```text
SUMO/TraCI  <->  safety layer  <->  env (centralized | local per-TLS)  <->  PPO / MaskablePPO
     |                                                                         |
 tripinfo, trajectories -> artifacts/  ->  FastAPI (api/)  ->  React dashboard (frontend/)
```

---

## Repository structure

```text
configs/                  hyperparameters (ppo, ippo, ippo_phase_select, env, eval) and sumo/InTAS_train.sumocfg
data/                     processed network GeoJSON, metadata (data/raw/intas is cloned by setup, git-ignored)
src/{scenario,environment,controllers,agents,evaluation}/
api/                      FastAPI backend (metrics, replay, network geometry, XAI, simulation launch)
frontend/                 React + TypeScript + Vite dashboard
artifacts/                tracked metrics/reports/scenario JSON; models and trajectories git-ignored
scripts/                  CLI entry points (see Workflow)
tests/                    pytest suites
docs/                     MODELS.md, LOGGING.md, archive/
```

---

## Quickstart

Requirements: macOS or Linux, Python 3.11 or 3.12 (3.11 is what CI and development use), Eclipse SUMO 1.20+ (`brew install sumo`, `apt install sumo sumo-tools`, or `pip install eclipse-sumo`), Node LTS for the dashboard.

```bash
make install            # .venv + editable install with dev extras (uses uv)
make sumo               # checks SUMO, clones InTAS at a PINNED commit, installs configs/sumo/InTAS_train.sumocfg
source .env.sumo        # exports SUMO_HOME / PATH / PYTHONPATH in your shell
make test lint          # backend tests + ruff
make run-api            # API on 127.0.0.1:8000
make run-frontend       # dashboard on http://127.0.0.1:5173
```

`make install-lock` installs exactly the versions in `requirements.lock`. Run `make help` for all targets.

**Docker**: `cp .env.example .env && docker compose up --build` serves the API on `127.0.0.1:8000` and the dashboard on `127.0.0.1:8080` (needs `artifacts/` and `data/` on the host).

### Workflow

1. Inspect the scenario: `python -m src.scenario.inspect_intas`
2. Generate the true fixed-time programs: `python scripts/generate_static_programs.py`
3. Train:
   - Centralized PPO: `python scripts/run_training.py --ppo-config configs/ppo.yaml`
   - IPPO: `python scripts/run_ippo_training.py --ippo-config configs/ippo.yaml`
   - Phase-select IPPO: `python scripts/run_ippo_maskable_training.py --sumo-cfg data/raw/intas/scenario/InTAS_train.sumocfg --tls-json artifacts/scenario/tls_metadata.json` (resume with `scripts/resume_ippo_maskable_training.py`)
4. Evaluate baselines and PPO/IPPO: `python scripts/run_evaluation.py --sumo-cfg data/raw/intas/scenario/InTAS_buildings.sumocfg --tls-json artifacts/scenario/tls_metadata.json --ppo-model <path> --output-dir artifacts/metrics --plot`
   Phase-select (needs `action_masks()`, so it has its own script): `python scripts/run_ippo_maskable_eval.py --sumo-cfg data/raw/intas/scenario/InTAS_train.sumocfg --tls-json artifacts/scenario/tls_metadata.json --model <path>`
   Note: the `ippo` checkpoint was trained on an older observation layout and cannot be re-evaluated with `--ippo-model` under the current 18-dim env.
5. Watch live in SUMO-GUI: `python scripts/run_visual_demo.py --controller <fixed_time|sumo_actuated|max_pressure|ppo|ippo>` (`ippo_phase_select` is not offered there).
6. XAI: `python scripts/run_xai.py --controller ppo --model <path>`
7. Dashboard: see Quickstart.

Model weights are not in git: see [docs/MODELS.md](docs/MODELS.md) for run IDs, SHA-256 and distribution.

### Troubleshooting

- `SUMO_HOME` not set: run `source .env.sumo` (after `make sumo`) or set it manually.
- `InTAS_train.sumocfg` not found: re-run `make sumo`; it copies the tracked `configs/sumo/InTAS_train.sumocfg` into the clone.
- API warns about `CITYMIND_API_KEY`: fine on loopback; set it before exposing the API (see [SECURITY.md](SECURITY.md)).

---

## Licence

Our code is Apache-2.0 ([LICENSE](LICENSE)). InTAS is GPL-3.0 and is *not* redistributed here (fetched at setup); a few small InTAS-derived data files are tracked and should be treated as GPL-3.0. Details and attribution: [NOTICE](NOTICE). Security policy: [SECURITY.md](SECURITY.md). Contributing: [CONTRIBUTING.md](CONTRIBUTING.md). History: [CHANGELOG.md](CHANGELOG.md).

## Further reading
- [`docs/MODELS.md`](docs/MODELS.md) - model manifest.
- [`CLAUDE_AGENT_HANDOVER.md`](CLAUDE_AGENT_HANDOVER.md) - historical (pre-IPPO) notes on the SUMO/TraCI performance work.
- [`PROJECT_RESEARCH_AND_VISION.md`](PROJECT_RESEARCH_AND_VISION.md) - original research proposal (historical; scope changed).
