# CityMind AI — Reinforcement Learning for Traffic Signal Control

CityMind AI is a reproducible research and engineering platform evaluating whether deep reinforcement learning (RL) can optimize network-wide traffic signals across a real-world city grid (the Ingolstadt InTAS scenario in Eclipse SUMO), compared with conventional fixed-time and Max-Pressure controllers.

![IPPO policy replaying on the InTAS Ingolstadt network](artifacts/traffic_simulation_showcase.gif)

---

## 🎯 Core Research Question

> **Can deep-RL traffic-signal control reduce congestion and travel delay compared to fixed-time and Max-Pressure control under identical simulation conditions — and does giving each intersection its own agent and reward (rather than one policy jointly controlling all of them) change the answer?**

The project trained and evaluated **two RL architectures** against the same two classical baselines, on the same 98-intersection network, same seed, same demand:

1. **PPO (Centralized)** — one policy observes the entire network and jointly outputs an action for every intersection (`MultiBinary(98)`). This is the project's original design.
2. **IPPO (Parameter-Shared, decentralized)** — one shared policy is invoked independently per intersection, each seeing only a fixed-size local observation and receiving its own local reward (`Discrete(2)`: extend or switch). Built after the centralized design was diagnosed with a credit-assignment failure (see [Results](#-results-seed-42-10-demand) below) — one global reward gave no per-intersection agent a way to tell whether *its own* action helped or hurt.

---

## 📊 Results (seed 42, 1.0× demand)

| Controller | Mean Wait (s) | Mean Queue | Throughput | Mean Speed (m/s) | Total Arrived |
|---|---|---|---|---|---|
| Fixed-Time (baseline) | **3.02** | 0.048 | 82.7% | 10.53 | 268 |
| **IPPO (Parameter-Shared)** | 4.15 | 0.067 | 74.7% | **13.02** | 233 |
| PPO (Centralized) | 13.47 | 0.108 | 81.4% | 12.76 | 253 |
| Max-Pressure | 37.38 | 0.143 | 74.2% | 6.74 | 242 |

Switching from a centralized policy to a parameter-shared decentralized one (IPPO) cut mean waiting time from 13.47s to 4.15s — a 3.2× improvement — and pushed mean speed above every other controller, including the professionally-tuned fixed-time baseline. It still trails fixed-time on raw wait time; each IPPO agent only ever sees and optimizes its own intersection, with no explicit coordination channel between neighbors. See [`PROJECT_RESEARCH_AND_VISION.md`](PROJECT_RESEARCH_AND_VISION.md) and the in-app XAI Analysis page for the coordination-side evidence, and the RL-alternatives write-up for concrete next steps (MAPPO, CoLight, PressLight, FRAP) that add an explicit coordination mechanism on top of this result.

---

## 🏛️ Architecture & Formulation

- **Microscopic Traffic Engine**: Eclipse SUMO + TraCI.
- **Scenario**: InTAS (Ingolstadt Traffic Scenario) — a realistic, validated urban network with **98 signalized intersections**.
- **Action Space**: Binary per intersection — `0` (maintain current phase) or `1` (advance to next phase).
- **Safety Layer**: Shared by every controller (RL and baseline alike) — enforces minimum/maximum green time and mandatory yellow/all-red clearance intervals; no controller can request an unsafe transition.
- **Observation Space**:
  - *Centralized PPO*: one large vector concatenating every intersection's per-lane queues, densities, speeds, and phase state, plus global network stats.
  - *Decentralized IPPO*: a fixed 10-dimensional vector per intersection (phase timing, own-lane queue/speed/density aggregates, a downstream-neighbor queue signal, and two network-wide totals) — fixed-size so one shared network can be invoked for every intersection despite InTAS's TLS ranging from 4–19 lanes and 3–11 phases.
- **Baselines**: Fixed-time (InTAS's own default cycle) and Max-Pressure (queue-pressure differential).
- **RL Algorithm**: Proximal Policy Optimization (PPO) via Stable-Baselines3 for both architectures — IPPO trains the same SB3 `PPO` class on a PettingZoo `ParallelEnv` wrapped with SuperSuit so 98 independent agents share one set of weights.

---

## 📂 Repository Structure

```text
├── configs/                  # Environment, PPO, and IPPO hyperparameters
├── data/                     # InTAS scenario files, processed network GeoJSON
├── src/
│   ├── scenario/              # Scenario inspection, TLS metadata extraction
│   ├── environment/            # Centralized + decentralized Gymnasium/PettingZoo envs, safety, rewards, metrics
│   ├── controllers/            # Baselines (Fixed-Time, Max-Pressure)
│   ├── agents/                 # PPOAgent and IPPOAgent wrappers
│   └── evaluation/              # Paired evaluation, statistical comparison, XAI
├── api/                       # FastAPI backend serving metrics, replay, network geometry, XAI
├── frontend/                  # React + TypeScript Vite dashboard
├── artifacts/                 # Metrics, trained checkpoints, trajectories, reports
├── scripts/                   # CLI entrypoints (training, evaluation, visual demo, XAI, GIF export)
└── tests/                     # Pytest unit and integration test suites
```

---

## 🚀 Quickstart

### 1. Requirements
- macOS (Apple Silicon M-series recommended) or Linux
- Python 3.11+ (`uv` recommended)
- Eclipse SUMO 1.20+ with `SUMO_HOME` exported

### 2. Environment Setup
```bash
# Verify SUMO and clone the InTAS scenario
chmod +x scripts/setup_sumo.sh
./scripts/setup_sumo.sh

# Create the Python virtual environment and install dependencies
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 3. Workflow Progression
1. **Inspect the scenario**: `python -m src.scenario.inspect_intas`
2. **Train**:
   - Centralized PPO: `python scripts/run_training.py --ppo-config configs/ppo.yaml`
   - Decentralized IPPO: `python scripts/run_ippo_training.py --ippo-config configs/ippo.yaml`
3. **Evaluate & compare against baselines**:
   ```bash
   python scripts/run_evaluation.py \
     --sumo-cfg data/raw/intas/scenario/InTAS_buildings.sumocfg \
     --tls-json artifacts/scenario/tls_metadata.json \
     --ppo-model artifacts/experiments/ppo/<run_id>/model_final.zip \
     --ippo-model artifacts/experiments/ippo/<run_id>/model_final.zip \
     --output-dir artifacts/metrics --plot
   ```
4. **Watch it live** (SUMO-GUI, any of the 4 controllers, including side-by-side):
   `python scripts/run_visual_demo.py --controller ippo`
5. **Generate XAI explanations**: `python scripts/run_xai.py --model artifacts/experiments/ppo/<run_id>/model_final.zip --tls-json artifacts/scenario/tls_metadata.json`
6. **Launch the dashboard**:
   - Backend: `uvicorn api.main:app --reload --port 8000`
   - Frontend: `cd frontend && npm install && npm run dev`

### Further reading
- [`CLAUDE_AGENT_HANDOVER.md`](CLAUDE_AGENT_HANDOVER.md) — operational notes on the SUMO/TraCI performance work and the original centralized-PPO XAI engine (predates IPPO; kept as historical record of the perf investigation).
- [`PROJECT_RESEARCH_AND_VISION.md`](PROJECT_RESEARCH_AND_VISION.md) — the original research proposal (predates IPPO and the final scope; several ideas in it, like the full DQN/A2C experiment matrix across multiple demand levels, were superseded by the centralized-vs-decentralized PPO comparison actually run).
