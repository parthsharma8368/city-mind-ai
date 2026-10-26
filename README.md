# CityMind AI - shared-policy reinforcement learning for traffic-signal control

One neural network, shared by all 98 signals of the Ingolstadt InTAS network (Eclipse SUMO), controls the morning peak
(07:00-08:00, ~15,000 scheduled cars) and is scored against SUMO's own vehicle-actuated signal control.
Algorithm: **PPO used as IPPO** (independent agents, shared weights) with separate actor / critic gradient clipping.

## Result (07:00-08:00 peak, 10 simulation seeds, paired by seed)

Two stages: **stage 1** trains the shared policy from scratch (100 PPO updates); **stage 2** continues it with one extra input (cars waiting outside the
network count towards the demand of the lanes they will queue at), 10 updates for seed 42 and 20 for seed 1042. "Final" = stage 2. Two independently trained models,
tested on 10 seeds (40 runs, `scripts/evaluate.py`; full table in [docs/TEST_REPORT_10_SEEDS.md](docs/TEST_REPORT_10_SEEDS.md)):

| Metric (per hour) | SUMO actuated | Model A | Model B | Verdict (10 seeds) |
|---|---|---|---|---|
| Lag per scheduled car (s) | 276.8 | **235.4** | **239.9** | beaten, 10/10 seeds each (-41.4 / -36.9 s) |
| Time loss per finished car (s) | 336.6 | **286.4** | **289.4** | beaten, 10/10 |
| Backlog at 08:00 (cars) | 3,108 | **2,852** | **2,921** | beaten, 10/10 |
| Mean waiting time (s) | 6.39 | **3.77** | **3.86** | beaten, 10/10 (-41% / -40%) |
| Mean queue / speed (m/s) | 0.440 / 8.70 | **0.318 / 9.37** | **0.314 / 9.30** | beaten, 10/10 |
| Due completion | 0.839 | **0.861** | **0.856** | beaten, 10/10 |
| Trips completed | 11,585 | **11,842** | **11,791** | beaten, 10/10 (+258 / +206) |
| Throughput (arrived / departed) | 0.796 | **0.813** | **0.809** | beaten, 10/10 |
| Trip duration / overdue cars | 745 s / 1,853 | **699 / 1,597** | **702 / 1,649** | beaten, 10/10 |
| Depart delay of inserted cars (s) | 15.7 | **14.1** | 14.9 | A beaten (10/10); B better (8/10) |
| Teleports (gridlock) | 13.3 | 7.5 | 10.0 | lower on average, 6/10 seeds: too noisy to call |
| Cars never inserted | 503 | 501 | 481 | **level, not won** (5/10 and 7/10 seeds) |

"Beaten" = the 95% interval of the per-seed difference excludes zero AND the sign test is significant (9-10 of 10 seeds). No measure is worse than the baseline.
A first 3-seed test showed a win on cars never inserted (473 vs 539); the baseline happened to be unusually high on those three seeds, so the honest statement is that the
fix closed the gap without opening a lead. Held-out windows (never trained on, one seed, model A only): 08:00-09:00 lag 230.7 -> 177.8, 17:00-18:00 108.5 -> 50.0.

Stage 1 alone (lag 251 s / 251 s) was already ~26 s ahead but lost ~100 more cars at the network entries (656 / 640 vs 539 on the first three seeds): one entry road's queue formed
outside the network where the agent could not see it. Not tested: another network, real traffic, degraded sensing.

## What the agent sees, does and is rewarded for

* **Observation (22 numbers per signal, each squashed into [-1, 1])**: active green phase (one-hot), in-transition flag, time in phase relative to
  its own minimum / maximum green, demand of each green phase, worst queue, cars waiting outside the network, number of phases, queues of the
  2 strongest road-connected upstream and downstream signals (`src/environment/local_observation.py`).
  Phase demand `d_i = sum over green movements [ halted(lane) + halted(feeder lanes) + pending share - halted(downstream lanes) ]`, squashed `d/(|d|+8)`.
  Feeder lanes are the unsignalised roads feeding a short stop-line lane, walked upstream up to 300 m: without them a 20 m lane (~2 cars) hides a 60-car queue.
* **Action**: keep the current green or advance to the next one, every 5 s. A safety layer enforces minimum / maximum green and the program's own yellow / all-red.
* **Reward** (`src/environment/local_reward.py`): `r = -scale * min( w_q Q + w_d D + w_s S + w_b P_own + w_g P_all/N , clip )` with Q stopped cars on the
  signal's lanes (+ feeders), D mean stopped cars per downstream lane, S = 1 if it switched, P cars waiting outside the network; weights 1, 2, 1, 0.5, 0.5, scale 0.1, clip 200.
  A level of delay (not its change), so the return is minus total vehicle-delay and cannot be gamed by timing a queue release.
* **Learning** (`src/agents/`): PPO with GAE (gamma 0.99, lambda 0.95), clip 0.2, value coefficient 0.5, entropy 0.01, networks 22-256-256 (tanh) for actor and critic,
  105,840 samples per update (180 steps x 98 signals x 6 SUMO runs), 100 updates. The actor's and the critic's gradient norms are clipped **separately** at 0.5: under joint clipping
  the critic's gradient (norm ~3,800) shrank the actor's by ~7,600x and the policy never learned.
* **Metrics** (`src/evaluation/drain_metrics.py`): `lag = mean over ALL scheduled cars of (completion time, or 08:00 if unfinished / never inserted) - (zero-delay completion time)`;
  due completion; never inserted = scheduled - inserted; depart delay; teleports; backlog; time loss.

## Repository

```text
configs/environment.yaml, ippo.yaml     the trained recipe (environment_pending.yaml, ippo_finetune.yaml: an experiment)
src/environment/                        env (one agent per signal), observation, reward, safety layer, lane statistics, metrics
src/agents/                             IPPO training, split gradient clipping, model contract
src/controllers/, src/evaluation/       baselines (sumo_actuated, fixed_time) and the drain-aware scoring
src/scenario/                           network / signal metadata
scripts/                                run_ippo_training.py, resume_ippo_training.py, finetune_ippo.py, evaluate.py, ...
api/, frontend/                         Showcase (/), results dashboard (/dashboard): `make run-api` + `make run-frontend` (reads artifacts/eval10_run)
tests/                                  pytest (runs without SUMO)
```

## Quickstart

Requirements: macOS or Linux, Python 3.11, Eclipse SUMO 1.20+.

```bash
make install && make sumo && source .env.sumo    # venv, SUMO check, pinned InTAS scenario
make test lint
make train                                       # ~7 h: configs/ippo.yaml + configs/environment.yaml
python scripts/evaluate.py --controller ippo --model <run>/model_final.zip --out artifacts/eval/ippo
python scripts/evaluate.py --controller sumo_actuated --out artifacts/eval/sumo_actuated
python scripts/evaluate.py --controller ippo --model <...> --begin 28800 --out artifacts/eval/ippo_0800   # another window
```

A run interrupted by a restart continues with `scripts/resume_ippo_training.py --checkpoint <ckpt> --run-dir <run dir>`.
Model weights are not in git: see [docs/MODELS.md](docs/MODELS.md).

### Showcase (presenting the project)

`make run-api` and `make run-frontend`, then open <http://localhost:5173/> and press **Run the showcase**: the recorded 07:20-07:30 run of
SUMO actuated, fixed time, model A and model B plays side by side at 10x; the stored results of each controller dissolve in below while
it plays, and the benchmarks (paired 10-seed verdicts, lag per seed, mini-bars, paired test) load only after all four are in.
**Watch it in SUMO** (bottom of the page) starts the real SUMO-GUI on this computer for one controller, or all four tiled, through
`scripts/run_visual_demo.py` (needs SUMO with the GUI; logs in `artifacts/demo_logs/`). Frames are served by `/api/showcase/*` from `artifacts/eval10_run`.

## Licence

Our code is Apache-2.0 ([LICENSE](LICENSE)). InTAS is GPL-3.0 and is not redistributed here (fetched at setup); a few small InTAS-derived data files
are tracked and should be treated as GPL-3.0 ([NOTICE](NOTICE)). Security: [SECURITY.md](SECURITY.md). Contributing: [CONTRIBUTING.md](CONTRIBUTING.md). History: [CHANGELOG.md](CHANGELOG.md).
