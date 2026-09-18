# CityMind AI — Centralized Reinforcement Learning for Traffic Signal Control

CityMind AI is a reproducible research and engineering platform evaluating whether a **single centralized deep reinforcement learning (RL) policy** can optimize network-wide traffic signals across a real-world city grid (the Ingolstadt InTAS scenario in Eclipse SUMO) compared with conventional fixed-time and Max-Pressure controllers.

---

## 🎯 Core Research Question

> **Can a centralized deep-RL policy that observes the complete traffic network and jointly controls all traffic signals reduce congestion and travel delay compared to fixed-time and Max-Pressure control under identical simulation conditions?**

---

## 🏛️ Architecture & Formulation

- **Single Centralized Agent**: One policy jointly selects actions for all controlled intersections ($N$ TLS). Not multi-agent.
- **Microscopic Traffic Engine**: Eclipse SUMO + `libsumo` / TraCI.
- **Scenario**: InTAS (Ingolstadt Traffic Scenario) — a realistic, validated urban network with 20 traffic-light systems.
- **Action Space**: `MultiBinary(N)` where action at each TLS is `0` (maintain current phase) or `1` (advance to next phase).
- **Safety Layer**: Strictly guarantees legal transitions, minimum green ($5\text{s}$), maximum green ($60\text{s}$), and yellow/all-red clearance intervals.
- **Observation Space**: Concatenation of per-TLS lane queues, densities, speeds, phase states, and global network delay metrics.
- **Baselines**: Fixed-time (InTAS default cycle) and Max-Pressure (queue pressure differentials).
- **Primary RL**: Proximal Policy Optimization (PPO) via Stable-Baselines3.

---

## 📂 Repository Structure

```text
citymind-ai/
├── configs/                  # Experiment, environment, and PPO hyperparameters
│   ├── environment.yaml
│   ├── ppo.yaml
│   └── evaluation.yaml
├── data/
│   ├── raw/intas/            # Cloned InTAS scenario files (.net.xml, .sumocfg, routes)
│   ├── processed/            # Processed artifacts & GeoJSON network map
│   └── metadata/             # TLS metadata, network topology, demand specs
├── src/
│   ├── scenario/             # Scenario inspection, metadata extractors
│   ├── environment/          # Centralized Gymnasium environment, safety, rewards, metrics
│   ├── controllers/          # Baselines (Fixed-time, Max-Pressure)
│   ├── agents/               # RL agent wrappers (PPO)
│   ├── training/             # Training loop, callbacks, TensorBoard logging
│   ├── evaluation/           # Paired evaluation, statistical analysis, comparison
│   └── replay/               # Parquet trajectory recorder and network GeoJSON export
├── api/                      # FastAPI backend serving metrics, replay, and network geometry
├── frontend/                 # React + TypeScript Vite dashboard with Canvas network replay
├── experiments/              # Training checkpoints and TensorBoard runs
├── artifacts/                # Metrics, reports, and trajectory parquet files
├── scripts/                  # Shell utilities (setup_sumo.sh, etc.)
└── tests/                    # Pytest unit and integration test suites
```

---

## 🚀 Quickstart

### 1. Requirements
- macOS (Apple Silicon M-series recommended) or Linux
- Python 3.11 or 3.12 (`uv` recommended)
- Eclipse SUMO 1.20+ with `SUMO_HOME` exported

### 2. Environment Setup
```bash
# Verify SUMO and clone InTAS scenario
chmod +x scripts/setup_sumo.sh
./scripts/setup_sumo.sh

# Create Python virtual environment and install dependencies
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 3. Workflow Progression
1. **Inspect Scenario**: `python -m src.scenario.inspect_intas`
2. **Run Fixed-Time Baseline**: `python -m src.evaluation.evaluate --controller fixed_time`
3. **Train PPO**: `python -m src.training.trainer --config configs/ppo.yaml`
4. **Evaluate & Compare**: `python -m src.evaluation.compare`
5. **Launch Dashboard**:
   - Backend: `uvicorn api.main:app --reload --port 8000`
   - Frontend: `cd frontend && npm install && npm run dev`
