> **Historical document (pre-2026-09-19).** This is the original pre-implementation research proposal — written before the project scaled from InTAS's ~20-intersection subset to the full 98-intersection network, and before the parameter-shared IPPO architecture (built to fix a credit-assignment failure in the centralized design below) existed. Several ideas here — the full DQN/DoubleDQN/DuelingDQN/A2C training matrix across multiple demand levels, an explicit Actuated baseline — were superseded by the centralized-vs-decentralized PPO comparison actually run. See [`README.md`](README.md) for current results and architecture.

# CityMind AI — Centralized Reinforcement Learning for Traffic Signal Control

**Document:** Initial research, project purpose, vision, dataset/environment, training and evaluation plan  
**Primary scenario:** InTAS (Ingolstadt Traffic Scenario) + Eclipse SUMO / libsumo  
**RL formulation:** Centralized single-agent control of the complete signalized network  
**Intended consumer:** Antigravity / AI coding agent  
**Status:** Initial research and engineering specification  
**Research cutoff:** September 2026

---

## 1. Executive Summary

CityMind AI is a research and engineering project for evaluating whether a **single centralized reinforcement-learning (RL) policy can improve traffic-signal coordination across an entire road network compared with conventional fixed-time signal control**.

The central design constraint is deliberate:

> **There is one RL agent/policy for the entire controlled network. The policy receives a global representation of the traffic state and jointly decides signal-control actions for all controlled intersections. Individual intersections are not independent RL agents.**

The project will use **InTAS — the Ingolstadt Traffic Scenario for SUMO** as the primary realistic scenario. InTAS is an open SUMO traffic scenario based on Ingolstadt's real road network. Its authors report real traffic-light locations and programs, 24-hour simulation, public transport, buildings and parking, traffic demand modeled using ActivityGen plus demographic/real-traffic information, and validation against real traffic measurements at 24 measurement points. The InTAS repository currently reports 20 traffic-light systems and an NRMSE of 0.33 for the scenario validation. [InTAS GitHub](https://github.com/silaslobo/InTAS) · [InTAS paper](https://doi.org/10.52825/scp.v1i.102).

SUMO is not merely a visualization tool in this project. It is the **microscopic traffic simulator and environment dynamics engine**. The RL system will interact with SUMO through TraCI during development and preferably libsumo during large-scale training/evaluation. SUMO explicitly supports externally controlled traffic-light algorithms through TraCI, while libsumo provides the same general API without socket communication overhead.

The project should **not build a second traffic simulator**. Instead, the engineering contribution is the centralized RL environment, controller implementations, experimental protocol, metrics pipeline, artifact/replay system, and visualization layer built around SUMO.

---

## 2. Project Purpose

### 2.1 Problem

Fixed-time traffic signal controllers use predetermined signal schedules. They cannot directly adapt their decisions to the instantaneous traffic state.

A traffic network, however, is dynamic:

- vehicle arrivals change over time;
- queues propagate between intersections;
- congestion at one intersection can affect downstream intersections;
- phase decisions at one signal influence traffic arriving at neighboring signals;
- demand intensity varies throughout the day.

The project investigates whether an RL controller can exploit these dynamic relationships by observing the network as a whole and adapting signal decisions.

### 2.2 Core research question

> **Can a centralized deep-RL policy that observes the complete traffic network and jointly controls all selected traffic signals reduce network congestion and travel delay compared with fixed-time signal control under the same traffic demand and simulation conditions?**

Secondary questions:

1. How does PPO compare with value-based discrete-action methods?
2. Does centralized global observation improve network-level coordination?
3. How robust is the learned policy to unseen traffic-demand levels?
4. How does RL compare with stronger non-RL baselines such as actuated control and Max-Pressure?
5. How stable are results across random seeds?
6. What traffic conditions cause the learned policy to outperform or fail to outperform conventional control?

---

# 3. Project Vision

The long-term vision is **CityMind AI: a reproducible traffic-intelligence platform that learns network-level traffic-signal policies, evaluates them against conventional controllers, and visually explains the resulting traffic behavior.**

The system should eventually support:

```text
Traffic scenario
      ↓
SUMO / libsumo
      ↓
Centralized RL environment
      ↓
RL policy
      ↓
Joint control of all signals
      ↓
Simulation trajectory
      ↓
Metrics + statistical analysis
      ↓
Replay / visualization
```

The final demonstration should allow a user to compare the same traffic scenario under:

- fixed-time control;
- actuated control;
- Max-Pressure;
- PPO;
- other selected RL policies.

The visualization should show both traffic behavior and quantitative performance.

---

# 4. What We Mean by "Dataset"

A critical conceptual distinction:

**InTAS is primarily a simulation scenario/environment, not a conventional supervised-learning dataset.**

It provides the ingredients needed to reproduce a traffic world inside SUMO:

- road network;
- traffic demand/routes;
- traffic-light systems/programs;
- scenario configuration;
- other simulated infrastructure and traffic components.

The RL policy is **not trained by fitting labels to a static table**.

Instead:

```text
InTAS scenario
      ↓
SUMO simulation
      ↓
Current traffic state
      ↓
RL action
      ↓
SUMO advances
      ↓
New traffic state + reward
      ↓
RL update
      ↓
repeat
```

This is reinforcement learning through environment interaction.

The InTAS paper describes a realistic traffic scenario whose road topology represents Ingolstadt's real road map; twenty traffic-light systems were simulated according to real deployed programs; traffic demand was modeled using ActivityGen with demographic and real-traffic information; and the scenario was validated against real traffic data from 24 measurement points.

Source: [InTAS paper](https://doi.org/10.52825/scp.v1i.102).

---

# 5. Primary Scenario: InTAS

## 5.1 Official repository

**GitHub:**  
https://github.com/silaslobo/InTAS

The repository describes InTAS as a realistic Ingolstadt SUMO scenario and currently documents:

- 24-hour simulation;
- actual Ingolstadt road network;
- real traffic-light locations;
- real traffic-light systems;
- public transport routes/timetables;
- public parking areas;
- buildings;
- pedestrian simulation support;
- validation using 24 traffic measurement points;
- 20 traffic-light systems;
- reported scenario NRMSE of 0.33.

Source: [InTAS GitHub](https://github.com/silaslobo/InTAS).

## 5.2 Academic reference

Lobo, S., Neumeier, S., Fernandez, E. M. G., & Facchi, C.

**"InTAS - The Ingolstadt Traffic Scenario for SUMO."**

SUMO Conference Proceedings.

DOI:

https://doi.org/10.52825/scp.v1i.102

Source: [SUMO Conference Proceedings](https://www.tib-op.org/ojs/index.php/scp/article/view/102).

## 5.3 Why InTAS is appropriate

InTAS is useful because the project needs more than a toy grid.

The scenario has real-world grounding in:

- road topology;
- traffic-light locations;
- traffic-light programs;
- traffic measurements;
- traffic demand;
- public transport.

This gives the project a stronger experimental foundation than a purely synthetic network.

## 5.4 Important verification requirement

The repository and related Ingolstadt simulation projects have evolved over time. The exact files, signal IDs, available demand files and compatible SUMO version **must be verified after downloading the repository**.

Do not hard-code claims such as "19 intersections" until the actual selected network/configuration has been inspected.

The original InTAS repository currently reports **20 TLS systems**. A separate research implementation has used a 19-intersection subset. The project should define its final controlled subset programmatically after inspecting the downloaded network and document the selection explicitly.

---

# 6. Relevant Alternative / Secondary Benchmark

## CityFlow traffic-signal benchmark

Official benchmark:

https://traffic-signal-control.github.io/

GitHub:

https://github.com/traffic-signal-control/RL_signals

CityFlow:

https://github.com/cityflow-project/CityFlow

The benchmark provides road-network and traffic-flow files and reports benchmark results using Average Travel Time.

Relevant scenarios include:

- Hangzhou 4×4 — 16 intersections;
- Jinan 3×4 — 12 intersections;
- synthetic grids;
- Manhattan — 2,510 intersections in the benchmark description.

The Hangzhou/Jinan datasets are useful for **early environment development and sanity checking** because they are established traffic-signal-control benchmarks with explicit road-network and traffic-flow files.

However:

> **CityFlow/Hangzhou is a secondary development benchmark. InTAS remains the primary realistic final scenario for CityMind AI.**

Sources:

- https://traffic-signal-control.github.io/
- https://github.com/traffic-signal-control/RL_signals
- https://github.com/cityflow-project/CityFlow

---

# 7. Simulator Choice

## Eclipse SUMO

Official documentation:

https://sumo.dlr.de/docs/

SUMO is a microscopic road-traffic simulator capable of modeling individual vehicles and traffic-signal systems.

The SUMO traffic-light documentation explicitly supports:

- built-in traffic-light algorithms;
- custom traffic-light programs;
- external control through TraCI.

Source:

https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html

## TraCI

https://sumo.dlr.de/docs/TraCI/

TraCI (Traffic Control Interface) allows an external program to retrieve simulation values and manipulate simulated objects while the simulation is running.

This is the interface through which the RL environment can:

- inspect vehicle states;
- inspect lanes;
- inspect traffic signals;
- change signal control;
- advance the simulation.

## libsumo

https://sumo.dlr.de/docs/Libsumo.html

libsumo embeds SUMO as a library and avoids the socket communication overhead associated with normal TraCI client/server coupling.

Recommended usage:

```text
Development / debugging:
SUMO-GUI + TraCI

Training:
libsumo

Final visual replay:
SUMO-GUI or recorded trajectory replay
```

The SUMO documentation states that existing TraCI scripts can often be reused by importing:

```python
import libsumo as traci
```

subject to documented libsumo limitations.

---

# 8. High-Level System Architecture

```text
                     InTAS
                       │
          network + demand + signals
                       │
                       ▼
              ┌─────────────────┐
              │ SUMO / libsumo  │
              └────────┬────────┘
                       │
                    TraCI API
                       │
                       ▼
          ┌────────────────────────┐
          │ Centralized RL Env     │
          │                        │
          │ Global observation     │
          │ Global reward          │
          │ Joint action           │
          │ Safety layer           │
          └────────────┬───────────┘
                       │
                       ▼
                 ONE RL POLICY
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        TLS 1        TLS 2        TLS N
          │            │            │
          └────────────┼────────────┘
                       ▼
                     SUMO
                       │
                       ▼
                 New state
                       │
                       └─────────► RL loop
```

There must be **one policy**.

The system must not accidentally create:

```text
Agent 1 → TLS 1
Agent 2 → TLS 2
...
Agent N → TLS N
```

That is a multi-agent formulation and is outside the primary research question.

---

# 9. Project Phases

## Phase 0 — Environment setup

### Objectives

- install Python;
- install SUMO;
- install TraCI/libsumo;
- install Gymnasium;
- install Stable-Baselines3;
- download InTAS;
- run the original scenario.

### Acceptance criteria

```text
[ ] Python environment works
[ ] SUMO works
[ ] SUMO-GUI opens
[ ] TraCI import works
[ ] libsumo import works
[ ] InTAS repository downloaded
[ ] Original InTAS configuration runs
[ ] Vehicles move
[ ] Traffic lights operate
```

**Do not begin RL training before these conditions are satisfied.**

---

# 10. Phase 1 — Reverse-engineer the InTAS scenario

Before writing the RL environment, programmatically inspect:

```text
network file
route files
additional files
SUMO configuration
traffic-light IDs
traffic-light programs
phases
lanes
edges
junctions
vehicle types
simulation duration
demand files
```

Create a machine-readable metadata artifact:

```text
artifacts/scenario/tls_metadata.json
artifacts/scenario/network_metadata.json
artifacts/scenario/demand_metadata.json
```

Example:

```json
{
  "tls_id": "example_tls",
  "phases": [
    {
      "index": 0,
      "state": "GGrr",
      "duration": 30
    }
  ]
}
```

Do not assume all traffic signals have identical phase structures.

---

# 11. Phase 2 — Establish the fixed-time baseline

This must be implemented **before RL**.

The baseline should preferably use the documented/available InTAS traffic-light program rather than inventing arbitrary timings.

The objective is to reproduce the conventional controller under exactly the same:

- road network;
- vehicle demand;
- vehicle behavior;
- simulation duration;
- random seed.

Record:

- average travel time;
- average waiting time;
- total waiting time;
- average queue length;
- maximum queue length;
- throughput;
- stops.

Save raw results.

Example:

```text
results/
└── fixed_time/
    ├── seed_001.parquet
    ├── seed_002.parquet
    ├── seed_003.parquet
    ├── seed_004.parquet
    └── seed_005.parquet
```

Acceptance criterion:

> A fixed-time simulation can be reproduced deterministically for a known seed/configuration and produces valid traffic metrics.

---

# 12. Phase 3 — Build the centralized Gymnasium environment

Create:

```text
src/environment/
├── centralized_sumo_env.py
├── observation.py
├── action.py
├── reward.py
├── safety.py
└── metrics.py
```

The environment should implement the standard Gymnasium interface:

```python
reset()
step(action)
```

The environment owns:

- SUMO lifecycle;
- simulation stepping;
- state extraction;
- action application;
- reward calculation;
- termination;
- episode metadata.

---

# 13. Global Observation Design

The policy must observe the entire controlled network.

For each controlled signal/intersection, collect appropriate features such as:

### Signal state

- current phase;
- elapsed phase time;
- time since last phase transition.

### Lane state

- vehicle count;
- density/occupancy;
- queue length;
- mean speed.

### Network state

Optionally include:

- total vehicles;
- total network waiting time;
- total queued vehicles;
- throughput;
- network-level delay.

A conceptual observation is:

```text
GlobalObservation =
[
    TLS_1 features,
    TLS_2 features,
    ...
    TLS_N features,
    global network features
]
```

Normalize continuous features before feeding them to the policy where appropriate.

---

# 14. Observation Architecture

### Initial prototype

Use:

```text
Global feature vector
        ↓
MLP
        ↓
Policy
```

### Later research model

Use a shared intersection encoder:

```text
TLS 1 ──┐
TLS 2 ──┤
TLS 3 ──┤
...     ├──► shared encoder ─► global aggregation ─► policy
TLS N ──┘
```

This remains a **single centralized policy**.

The encoder is not an independent agent.

---

# 15. Action-Space Design

Avoid an unrestricted joint phase-number action.

If every intersection has four possible phases and there are 19 intersections:

\[
4^{19}=274,877,906,944
\]

joint configurations.

That is unnecessarily difficult.

A preferred initial formulation is a per-signal control decision such as:

```text
0 = maintain current phase
1 = request/advance to next valid phase
```

or a small discrete control vocabulary.

The environment's safety layer must translate the policy decision into a legal traffic-light transition.

---

# 16. Safety Layer

The RL policy must not be allowed to generate physically or operationally invalid signal transitions.

The safety layer should enforce:

- minimum green;
- maximum green;
- yellow duration;
- all-red clearance where applicable;
- minimum red;
- valid phase transitions.

Architecture:

```text
RL action
    ↓
Safety validator
    ↓
Legal signal command
    ↓
SUMO
```

The safety layer should be deterministic and independently testable.

---

# 17. RL Decision Frequency

SUMO can simulate at a fine time resolution while the RL policy acts less frequently.

Initial recommendation:

```text
SUMO timestep = 1 second
RL decision interval = 5 seconds
```

Meaning:

```text
policy action
    ↓
SUMO advances 5 steps
    ↓
new global observation
    ↓
policy action
```

Treat the decision interval as a hyperparameter for later ablation.

---

# 18. Reward Design

The reward should be **network-level**, not based only on one intersection.

A useful initial formulation is:

\[
R_t = -\Delta W_t
\]

where \(W_t\) is total network waiting time.

A richer formulation can include queue length and unnecessary signal switching:

\[
R_t =
-\alpha \Delta W_t
-\beta Q_t
-\gamma S_t
\]

where:

- \(W_t\) = total network waiting time;
- \(Q_t\) = network queue measure;
- \(S_t\) = signal-switching penalty.

An alternative potential-based form is:

\[
R_t =
\alpha(W_{t-1}-W_t)
+
\beta(Q_{t-1}-Q_t)
-
\gamma S_t
\]

The final coefficients must be treated as experimental hyperparameters and reported.

Avoid reward definitions that make the agent optimize one intersection at the expense of the network.

---

# 19. Phase 4 — Random-policy environment validation

Before training:

```python
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break
```

Verify:

```text
[ ] observation shape is stable
[ ] action shape is stable
[ ] actions reach all controlled TLS
[ ] safety layer works
[ ] SUMO continues without invalid transitions
[ ] reward is finite
[ ] metrics are populated
[ ] episode termination works
```

This catches environment bugs before they contaminate training.

---

# 20. Phase 5 — PPO as the first RL algorithm

Start with PPO.

Reason:

- it handles policy learning well for many discrete control formulations;
- it can work with vectorized environments;
- it is easier to adapt to a large structured action space than a naive DQN joint-action formulation.

Stable-Baselines3 documentation:

https://stable-baselines3.readthedocs.io/

PPO documentation:

https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Start with a simple MLP:

```text
Observation
    ↓
256
    ↓
256
    ↓
Policy + value heads
```

Start with approximately:

```text
500k steps
```

and scale to:

```text
1M
2M
```

based on learning curves rather than choosing an arbitrary budget.

---

# 21. DQN-family algorithms

DQN is naturally suited to discrete actions.

However, vanilla Stable-Baselines3 DQN supports a `Discrete` action space rather than arbitrary `MultiDiscrete`/`MultiBinary` action spaces. Therefore, **do not create an enormous joint discrete action ID for all intersections**.

Stable-Baselines3 DQN documentation:

https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html

For research comparisons, implement or use an appropriate factorized centralized Q architecture if Double-DQN/Dueling-DQN are required.

Conceptually:

```text
Global observation
        ↓
Shared centralized encoder
        ↓
┌───────┬───────┬───────┐
Q_TLS1  Q_TLS2  ... Q_TLSN
```

This is still one centralized network.

Do not interpret the per-TLS output heads as separate agents.

---

# 22. A2C

A2C can be included as another policy-gradient baseline.

Stable-Baselines3:

https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html

It is useful as a comparison point but should not distract from the primary PPO experiment.

---

# 23. SAC and TD3

Do **not** make SAC or TD3 primary algorithms in the initial experiment.

Their standard formulations target continuous action spaces, while traffic-signal phase control is naturally discrete.

They become relevant only if the project explicitly changes the action representation to continuous variables, for example:

```text
green extension duration ∈ [0, 10] seconds
```

or another continuous signal-timing parameterization.

That would be a different MDP and should be treated as a separate experiment, not mixed with the main discrete-action benchmark.

---

# 24. Training Data / Demand Strategy

Do not train on one fixed traffic trace and then evaluate on the same trace.

Instead create multiple demand conditions.

Conceptual training set:

```text
0.70 × nominal demand
0.85 × nominal demand
1.00 × nominal demand
1.15 × nominal demand
1.30 × nominal demand
```

Validation:

```text
0.90 × nominal
1.10 × nominal
```

Test:

```text
0.75 × nominal
1.00 × nominal
1.25 × nominal
1.50 × nominal
```

The exact multipliers must be implemented only if the InTAS demand representation permits valid scaling without violating scenario assumptions.

If multiple real-day calibrated demand files are available in the selected InTAS-compatible scenario, prefer those over synthetic scaling.

---

# 25. Avoiding Data Leakage

Do not split individual vehicles randomly.

That can allow the model to encounter effectively identical traffic patterns during training and testing.

Prefer scenario/seed/demand-level separation:

```text
TRAIN
 ├── demand A
 ├── demand B
 ├── demand C
 └── demand D

VALIDATION
 └── demand E

TEST
 ├── demand F
 ├── demand G
 └── demand H
```

The test configurations must never influence:

- reward tuning;
- hyperparameter selection;
- model checkpoint selection;
- reward coefficients;
- feature engineering.

---

# 26. Random Seeds

Every major experiment should be repeated over multiple seeds.

Minimum:

```text
5 independent seeds
```

For each seed, record:

- Python seed;
- NumPy seed;
- PyTorch seed;
- environment seed;
- traffic-demand seed where applicable;
- RL algorithm seed.

Report:

\[
mean \pm standard\ deviation
\]

For final claims, bootstrap confidence intervals can be added.

---

# 27. Baseline Controllers

The benchmark should contain at least:

### Baseline 1 — Fixed-time

Conventional signal program.

### Baseline 2 — Actuated

If an appropriate SUMO actuated controller is available/configurable for the selected network.

### Baseline 3 — Max-Pressure

Important because it is a strong traffic-signal-control baseline and is widely used in traffic-signal-control research.

### RL

- PPO;
- DQN/factorized DQN;
- Double-DQN;
- Dueling-DQN;
- optionally A2C.

Do not claim RL superiority merely because it beats fixed-time. A comparison against Max-Pressure makes the experiment more informative.

---

# 28. Evaluation Metrics

## Primary metrics

### Average travel time

Time required for completed trips.

### Average waiting time

Vehicle waiting time averaged across relevant vehicles.

### Total network waiting time

Network-wide accumulated waiting.

### Average queue length

Average number of queued vehicles.

### Maximum queue length

Useful for detecting localized congestion.

### Throughput

Number of completed/arrived vehicles during the evaluation period.

### Number of stops

Useful as an additional traffic-flow quality metric.

---

## Optional metrics

If supported consistently:

- fuel consumption;
- CO₂;
- emissions;
- network delay;
- speed distribution;
- intersection-level fairness.

The same metric definitions must be used for every controller.

---

# 29. Fair Evaluation Protocol

For every controller:

```text
same network
same demand
same simulation duration
same vehicle behavior
same initial conditions
same random seed set
```

Only the controller should change.

Conceptually:

```text
Scenario S + seed 1
       ├── Fixed-time
       ├── Max-Pressure
       └── PPO

Scenario S + seed 2
       ├── Fixed-time
       ├── Max-Pressure
       └── PPO
```

This paired evaluation is much stronger than comparing unrelated simulation runs.

---

# 30. Statistical Reporting

For each controller and demand condition report:

```text
mean
standard deviation
median where useful
95% confidence interval where appropriate
```

Example table:

| Controller | Demand | Travel Time | Waiting Time | Queue | Throughput |
|---|---:|---:|---:|---:|---:|
| Fixed-time | 1.0× | measured | measured | measured | measured |
| Max-Pressure | 1.0× | measured | measured | measured | measured |
| PPO | 1.0× | measured | measured | measured | measured |

Do not insert expected numbers into the document or code.

---

# 31. Required Experiment Artifacts

Every training run should produce:

```text
artifacts/
└── experiments/
    └── ppo/
        └── run_YYYYMMDD_HHMMSS/
            ├── config.yaml
            ├── model.zip
            ├── metrics.parquet
            ├── training_metrics.csv
            ├── evaluation_metrics.parquet
            ├── scenario_metadata.json
            ├── git_commit.txt
            └── README.md
```

The experiment configuration should contain:

```yaml
algorithm: PPO
seed: 42
total_timesteps: 1000000
decision_interval: 5

observation:
  lane_features:
    - queue
    - density
    - mean_speed
  signal_features:
    - phase
    - phase_elapsed

reward:
  waiting_delta_weight: 1.0
  queue_weight: 0.1
  switching_penalty: 0.01
```

Values are examples and must be tuned/reported experimentally.

---

# 32. Logging Schema

At minimum, log:

```text
episode_id
step
simulation_time
controller
seed
demand_id

total_vehicles
departed_vehicles
arrived_vehicles

total_waiting_time
average_waiting_time

average_queue
maximum_queue

average_speed
travel_time

signal_switch_count
reward
```

For signal-level analysis:

```text
tls_id
phase
phase_elapsed
queue
density
vehicles
```

Store large trajectory data in **Parquet**.

Use JSON/YAML for configuration and metadata.

---

# 33. Recommended Repository Structure

```text
citymind-ai/
│
├── README.md
├── PROJECT_RESEARCH_AND_VISION.md
├── pyproject.toml
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── intas/
│   ├── processed/
│   └── metadata/
│
├── src/
│   ├── environment/
│   │   ├── centralized_sumo_env.py
│   │   ├── observation.py
│   │   ├── action.py
│   │   ├── reward.py
│   │   ├── safety.py
│   │   └── metrics.py
│   │
│   ├── controllers/
│   │   ├── fixed_time.py
│   │   ├── actuated.py
│   │   └── max_pressure.py
│   │
│   ├── agents/
│   │   ├── ppo.py
│   │   ├── dqn.py
│   │   ├── double_dqn.py
│   │   └── dueling_dqn.py
│   │
│   ├── training/
│   │   ├── train_ppo.py
│   │   └── train_dqn.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── statistics.py
│   │   └── compare.py
│   │
│   └── scenario/
│       ├── inspect_intas.py
│       ├── tls_metadata.py
│       └── demand.py
│
├── configs/
│   ├── environment.yaml
│   ├── ppo.yaml
│   └── evaluation.yaml
│
├── experiments/
│
├── artifacts/
│   ├── models/
│   ├── trajectories/
│   ├── metrics/
│   └── reports/
│
├── api/
│   └── ...
│
└── frontend/
    └── ...
```

---

# 34. Simulation / Visualization Architecture

Do not build a new traffic simulator.

Use:

```text
SUMO = simulation engine
FastAPI = backend/replay API
React = visualization
```

Recommended flow:

```text
SUMO
  ↓
evaluation
  ↓
trajectory.parquet
  ↓
FastAPI
  ↓
React
  ↓
Canvas/WebGL visualization
```

This gives deterministic replay.

A later real-time mode can be added after the offline evaluation system works.

---

# 35. Visualization Requirements

The eventual UI should allow:

```text
Controller:
[ Fixed-time ▼ ]

Scenario:
[ InTAS ]

Demand:
[ 1.0× ]

Seed:
[ 42 ]

[ Play ] [ Pause ] [ Reset ]
```

Display:

```text
Network map
Traffic vehicles
Signal colors
Queue/congestion heatmap
Current simulation time
```

Metrics:

```text
Average waiting time
Average travel time
Queue length
Throughput
Stops
```

Comparison:

```text
Fixed-time vs PPO
Fixed-time vs Max-Pressure
```

The visualization is **not the research result**. The reproducible evaluation artifacts are.

---

# 36. Milestones and Acceptance Criteria

## Milestone 1 — InTAS works

```text
[ ] repository downloaded
[ ] configuration identified
[ ] SUMO simulation runs
[ ] traffic signals operate
[ ] traffic vehicles operate
```

## Milestone 2 — Scenario understanding

```text
[ ] TLS IDs extracted
[ ] phases extracted
[ ] network metadata extracted
[ ] demand metadata extracted
```

## Milestone 3 — Fixed-time baseline

```text
[ ] controller reproducible
[ ] metrics extracted
[ ] results stored
```

## Milestone 4 — Centralized environment

```text
[ ] one global observation
[ ] one global action
[ ] network-level reward
[ ] safety layer
[ ] Gymnasium compliance
```

## Milestone 5 — PPO

```text
[ ] PPO trains
[ ] reward curve generated
[ ] checkpoint saved
[ ] evaluation works
```

## Milestone 6 — Comparative benchmark

```text
[ ] fixed-time
[ ] Max-Pressure
[ ] PPO
[ ] additional RL algorithms
[ ] multiple seeds
[ ] multiple demand levels
```

## Milestone 7 — Research analysis

```text
[ ] confidence intervals
[ ] demand stress test
[ ] generalization test
[ ] ablation studies
```

## Milestone 8 — CityMind visualization

```text
[ ] trajectory artifacts
[ ] FastAPI
[ ] React
[ ] replay
[ ] controller comparison
```

---

# 37. Important Research Pitfalls

## Pitfall 1 — Accidentally creating multi-agent RL

Bad:

```text
19 PPO models
19 policies
19 independent agents
```

Correct:

```text
one centralized policy
global observation
joint control
```

## Pitfall 2 — Training on the test scenario

Never select hyperparameters based on test performance.

## Pitfall 3 — Using only fixed-time vs RL

Add Max-Pressure if feasible.

## Pitfall 4 — One random seed

Traffic simulation results can vary. Use multiple seeds.

## Pitfall 5 — Giant unrestricted joint action space

Avoid \(4^{19}\)-type action spaces.

## Pitfall 6 — Unsafe signal transitions

Use a deterministic safety layer.

## Pitfall 7 — Reward hacking

A controller could reduce one metric while creating severe congestion elsewhere.

Always monitor multiple network-level metrics.

## Pitfall 8 — Overclaiming

Do not state:

> "RL is always better."

Instead evaluate:

> "Under the specified scenarios, demand levels, seeds and metrics, controller X achieved the following measured performance."

## Pitfall 9 — Treating InTAS as static supervised data

The scenario is the simulated environment. RL learns from interaction.

## Pitfall 10 — Building the UI before validating the science

The order should be:

```text
Simulation
→ baseline
→ environment
→ RL
→ evaluation
→ statistics
→ visualization
```

---

# 38. Recommended Initial Research Benchmark

The first complete experiment should be:

```text
Scenario:
InTAS

Controlled network:
explicitly documented subset of InTAS TLS after programmatic inspection

Agent:
one centralized policy

Observation:
global lane + signal state

Action:
safe joint signal-control decisions

Primary RL:
PPO

Baselines:
fixed-time
Max-Pressure

Seeds:
≥5

Demand:
multiple training and held-out test conditions

Primary metrics:
average travel time
average waiting time
average queue
throughput

Secondary:
stops
maximum queue
delay
fuel/emissions where reliable
```

---

# 39. Recommended Research Progression

### Stage A — Proof of environment

```text
InTAS
↓
SUMO
↓
Fixed-time
```

### Stage B — RL feasibility

```text
InTAS
↓
centralized environment
↓
PPO
```

### Stage C — Algorithm comparison

```text
Fixed-time
Max-Pressure
PPO
DDQN
Dueling-DDQN
A2C
```

### Stage D — Robustness

```text
low demand
normal demand
high demand
extreme demand
```

### Stage E — Generalization

```text
unseen demand
unseen seeds
possibly unseen day/scenario configuration
```

### Stage F — Explainability

Investigate:

- which network regions influence decisions;
- when the policy changes phases;
- relationship between queue pressure and decisions;
- whether decisions coordinate neighboring intersections.

---

# 40. Longer-Term Architecture

The final system can evolve toward:

```text
                    CITYMIND AI
                         │
        ┌────────────────┼────────────────┐
        │                │                │
     Scenario         Training        Evaluation
        │                │                │
      InTAS             PPO           Fixed-time
      CityFlow          DDQN          MaxPressure
      SUMO              A2C           RL
        │                │                │
        └────────────────┼────────────────┘
                         │
                    Artifact Layer
                         │
                  Parquet + JSON
                         │
                    FastAPI API
                         │
                  React Dashboard
```

The artifact layer should make every experiment reproducible.

---

# 41. Instructions to Antigravity

The following constraints are mandatory for the implementation.

### A. Simulator

Use **Eclipse SUMO** as the traffic simulator.

Use:

- SUMO-GUI for debugging/visual inspection;
- TraCI for initial development;
- libsumo for high-throughput training/evaluation.

Do not implement a replacement microscopic traffic simulator.

### B. Primary scenario

Use **InTAS** as the primary scenario:

https://github.com/silaslobo/InTAS

Do not fabricate missing scenario metadata.

First inspect the downloaded repository and configuration files.

### C. Centralization

Implement exactly **one RL policy/agent**.

The policy must receive a representation of the complete controlled network.

The policy must jointly control all selected traffic signals.

Do not create one RL agent per intersection.

### D. Environment

Implement a custom Gymnasium-compatible centralized environment around SUMO/libsumo.

Required components:

```text
reset()
step()
observation extraction
action application
reward
safety layer
termination
metrics
```

### E. Baseline

Implement and validate fixed-time control before RL.

Add Max-Pressure if feasible.

### F. RL

Implement PPO first.

Then implement appropriate discrete-action comparisons.

Do not force SAC/TD3 into the main experiment unless a separate continuous-action formulation is intentionally designed.

### G. Action space

Avoid exponential unrestricted joint phase selection.

Use a compact, safe signal-control action representation.

### H. Safety

Enforce legal phase transitions and minimum/maximum timing constraints outside the neural policy.

### I. Evaluation

Every controller must use:

- identical scenario;
- identical demand;
- identical simulation duration;
- comparable seeds;
- identical metric definitions.

### J. Reproducibility

Every run must save:

```text
config
seed
model
metrics
scenario metadata
git commit
evaluation results
```

### K. Data

Do not overwrite raw InTAS files.

Use:

```text
data/raw/
data/processed/
artifacts/
```

### L. Training

Use multiple demand conditions and held-out test conditions where the scenario representation permits valid demand variation.

### M. Metrics

At minimum:

```text
average travel time
average waiting time
total waiting time
average queue
maximum queue
throughput
stops
```

### N. Visualization

Do not build the visualization before the evaluation pipeline.

After evaluation is correct, expose trajectory artifacts through FastAPI and visualize them with React.

---

# 42. Useful Official and Research Links

## Primary scenario

- **InTAS GitHub:** https://github.com/silaslobo/InTAS
- **InTAS paper:** https://doi.org/10.52825/scp.v1i.102
- **InTAS publication page:** https://www.tib-op.org/ojs/index.php/scp/article/view/102

## SUMO

- **SUMO documentation:** https://sumo.dlr.de/docs/
- **SUMO GitHub:** https://github.com/eclipse-sumo/sumo
- **TraCI:** https://sumo.dlr.de/docs/TraCI/
- **libsumo:** https://sumo.dlr.de/docs/Libsumo.html
- **Traffic lights:** https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html
- **TraCI traffic-light control:** https://sumo.dlr.de/docs/TraCI/Change_Traffic_Lights_State.html

## RL environment/reference

- **SUMO-RL:** https://github.com/LucasAlegre/sumo-rl
- **SUMO-RL documentation:** https://lucasalegre.github.io/sumo-rl/

## CityFlow secondary benchmark

- **CityFlow:** https://github.com/cityflow-project/CityFlow
- **Traffic signal benchmark:** https://traffic-signal-control.github.io/
- **Benchmark datasets:** https://traffic-signal-control.github.io/dataset.html
- **RL signal repository:** https://github.com/traffic-signal-control/RL_signals

## RL framework

- **Stable-Baselines3:** https://stable-baselines3.readthedocs.io/
- **PPO:** https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- **DQN:** https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
- **A2C:** https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html

## Gymnasium

- https://gymnasium.farama.org/

---

# 43. Research Sources and Evidence

### InTAS

Lobo et al., *InTAS - The Ingolstadt Traffic Scenario for SUMO*.

The paper describes the real-road topology, 20 traffic-light systems, real traffic-light programs, ActivityGen-based demand using demographic/real-traffic information, public transport, and validation against 24 traffic measurement points.

https://doi.org/10.52825/scp.v1i.102

### InTAS repository

The repository provides the downloadable SUMO scenario and documents its current features and validation status.

https://github.com/silaslobo/InTAS

### SUMO / TraCI

SUMO documentation states that TraCI provides access to a running road-traffic simulation, allowing external programs to retrieve simulated-object values and manipulate behavior online.

https://sumo.dlr.de/docs/TraCI/

### SUMO / libsumo

SUMO documentation describes libsumo as an embedded alternative to TraCI that avoids socket communication overhead and exposes similar APIs.

https://sumo.dlr.de/docs/Libsumo.html

### SUMO traffic lights

SUMO explicitly supports external traffic-light control through TraCI and supports traffic-light modeling for research.

https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html

### CityFlow benchmark

The benchmark provides road-network and traffic-flow files and established traffic-signal-control scenarios including Hangzhou, Jinan and Manhattan.

https://traffic-signal-control.github.io/

### CityFlow

CityFlow is a microscopic traffic simulator with a Python interface designed for large-scale traffic RL research.

https://github.com/cityflow-project/CityFlow

### Stable-Baselines3

Stable-Baselines3 provides implementations of PPO, DQN, A2C and other RL algorithms. Its DQN implementation uses a discrete action space, which is an important constraint when designing the centralized joint action representation.

https://stable-baselines3.readthedocs.io/

---

# 44. Final Project Definition

The project should ultimately demonstrate a complete, reproducible pipeline:

```text
                REAL-WORLD-GROUNDED SCENARIO
                           │
                         InTAS
                           │
                           ▼
                   SUMO / libsumo
                           │
                           ▼
              CENTRALIZED RL ENVIRONMENT
                           │
              ┌────────────┴────────────┐
              │                         │
       GLOBAL OBSERVATION         GLOBAL REWARD
              │                         │
              └────────────┬────────────┘
                           ▼
                    ONE RL POLICY
                           │
                    JOINT SIGNAL ACTION
                           │
                           ▼
                         SUMO
                           │
                           ▼
                    TRAFFIC EVOLUTION
                           │
                           ▼
                  METRICS + TRAJECTORY
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
        Statistical analysis       Visualization
               │                       │
               └───────────┬───────────┘
                           ▼
              Fixed-time vs RL comparison
```

The scientific goal is not to assume that RL wins.

The goal is to build an experimental system in which the question can be answered **fairly, reproducibly and quantitatively**.

The final result should be a defensible statement of the form:

> Under the specified InTAS/SUMO network, demand distributions, controller constraints, random seeds and evaluation protocol, the centralized RL controller produced the measured changes in travel time, waiting time, queue length and throughput relative to conventional controllers.

That is the core of CityMind AI.
