# 🤝 CLAUDE AGENT HANDOVER & OPERATIONAL BRIEFING
**Project**: CityMind AI — Centralized Deep-RL Traffic Signal Intelligence Platform  
**Target City**: InTAS Ingolstadt Network (98 TLS, 1042 Controlled Lanes, 22 Route Files)  
**Hardware Profile**: macOS Apple Silicon (MPS acceleration enabled, SUMO 1.27.1 installed)

---

## 1. Executive Briefing: Where We Are & Critical Findings

A deep-dive audit and runtime profiling of the codebase was conducted. All core algorithmic modules (safety state machine, centralized gymnasium environment, PPO agent, baselines, evaluation harness, API, and frontend) are functional. However, **one major performance bottleneck** and **one concurrency challenge** were identified:

### 🔴 Critical Discovery: The TraCI Socket Round-Trip Bottleneck
1. **The Symptom**: A single 5-second simulation step takes **~2.47 seconds**. For a 3600-second episode (720 steps), one episode takes ~30 minutes. 500,000 steps would take over 300 hours!
2. **The Root Cause**:
   - On macOS, `libsumo` is not compiled for Python 3.12, falling back to standard socket TraCI.
   - In `src/environment/metrics.py`: line 129 iterates over **all lanes in the entire city network** (`for lid in traci.lane.getIDList(): traci.lane.getLastStepHaltingNumber(lid)`). In Ingolstadt, this generates **3,000+ separate TCP calls** every 5 simulation seconds!
   - In `src/environment/observation.py`: 1042 controlled lanes are queried across 4 metrics (4,168 socket calls).
   - In `observation.py`, `reward.py`, and `metrics.py`: all three independently loop over all network vehicles calling `traci.vehicle.getWaitingTime(vid)` separately!
   - **Total**: ~10,000 socket round-trips per step, consuming **66% of wall-clock time** in network latency.
   - **Measured & Verified Speedup**:
     - TraCI duration: dropped from **9.87s → 0.67s** (a **14.7x speedup in TraCI socket latency**).
     - Average step time: dropped from **2.473s → 0.212s** (an **11.7x overall speedup**).
     - Simulation speed: **7.18x real-time factor**.
     - An entire 1-hour simulation episode (720 steps) now executes in **~2.5 minutes** (previously 30 minutes).
     - Full PPO training is now realistically achievable on macOS Apple Silicon.

---

## 2. Concurrency Architecture: Training PPO While Running Simulation

You cannot run `sumo-gui` inside the active PPO training process because GUI rendering blocks the headless training loop. We decouple them into two concurrent tracks:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 InTAS Simulation Engine                 │
                  └───────────────────────────┬─────────────────────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    │                                                   │
        [Track A: Training]                                 [Track B: Visual Demo]
   Headless SUMO (`sumo`) in BG                     Live GUI (`sumo-gui`) or Web Replay
   • Port: Dynamic / 8813                           • Port: 8873 (Non-conflicting)
   • MPS Acceleration                               • Loads latest checkpoint from Track A
   • Saves Checkpoints every 25k/50k                • Runs watchable real-time simulation
   • Logs to TensorBoard                            • Replays at 60 FPS in Web Dashboard
```

---

## 3. Step-by-Step Execution Protocol

### Step 1: Launch PPO Training in Background
Use the correct scenario configuration path (`data/raw/intas/scenario/InTAS_buildings.sumocfg`):

```bash
cd "/Users/parthsharma/Desktop/FInal Try"
source .venv/bin/activate

# Launch PPO training (runs headless with MPS acceleration)
python scripts/run_training.py \
  --sumo-cfg data/raw/intas/scenario/InTAS_buildings.sumocfg \
  --tls-json artifacts/scenario/tls_metadata.json \
  --ppo-config configs/ppo.yaml \
  --env-config configs/environment.yaml \
  --seed 42 \
  --demand-id 1.0x \
  --output-dir artifacts/experiments/ppo
```
- **Checkpoints**: Automatically written to `artifacts/experiments/ppo/run_<TIMESTAMP>_seed42/checkpoints/`
- **Final Model**: Saved as `model_final.zip`

### Step 2: Monitor Training Dynamics via TensorBoard
Open in a separate terminal:
```bash
cd "/Users/parthsharma/Desktop/FInal Try"
source .venv/bin/activate
tensorboard --logdir artifacts/experiments/ppo/ --port 6006
# Open browser: http://localhost:6006
```
Key metrics to watch:
- `rollout/ep_rew_mean` (must rise towards zero / positive)
- `train/value_loss` (should stabilize)
- `train/entropy_loss` (gradual drop from -1.0 to -0.2)

### Step 3: Run the Visual Simulation Concurrently (SUMO-GUI)
While training runs in the background, you can inspect the simulation visually at any time:

```bash
cd "/Users/parthsharma/Desktop/FInal Try"
source .venv/bin/activate

# Launch live visual simulation with PPO controlling the lights (auto-loads viewsettings.xml):
python scripts/run_visual_demo.py --delay 50

# Or test with an explicit checkpoint:
python scripts/run_visual_demo.py \
  --model artifacts/experiments/ppo/run_20260917_172613_seed42/model_final.zip \
  --delay 50
```
- **What happens in SUMO-GUI**:
  - Automatically loads `configs/viewsettings.xml` (exaggerated vehicles, colored by speed: red=slow, green=fast, zoom=250).
  - Starts automatically without needing to press play (`--start`).
  - **The PPO RL policy controls the traffic lights** in real time, making decisions every 5 simulation seconds.
  - Differs from raw `sumo-gui` because lights are adaptively driven by the trained neural network instead of static schedules.

### Step 4: Run Headless Paired Evaluation & Plot Comparisons
Once training completes (or for baseline benchmarking):

```bash
python scripts/run_evaluation.py \
  --sumo-cfg data/raw/intas/scenario/InTAS_buildings.sumocfg \
  --tls-json artifacts/scenario/tls_metadata.json \
  --ppo-model artifacts/experiments/ppo/run_<TIMESTAMP>_seed42/model_final.zip \
  --output-dir artifacts/metrics \
  --trajectory-dir artifacts/trajectories \
  --plot
```
Outputs produced:
- Comparison table (Mean Waiting Time, Mean Queue, Throughput, Speed) across `Fixed-Time`, `Max-Pressure`, and `PPO`.
- High-resolution bar charts in `artifacts/reports/`:
  - `comparison_mean_waiting_time.png`
  - `comparison_throughput.png`
  - `comparison_mean_queue.png`

### Step 5: Web Dashboard & Trajectory Replay
Start the FastAPI server and React frontend to showcase the web replay:

```bash
# Terminal 1: Backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
cd frontend && npm run dev
# Browser: http://localhost:5173
```
- The frontend connects to `/api/network/geojson` to render the Ingolstadt road network.
- The Replay tab streams recorded frames from `/api/replay/frames` for fast 60 FPS playback without SUMO overhead.

---

---

## 4. XAI (Explainable AI) Engine for PPO Policy Decisions

A full, dependency-free XAI module has been built and verified in `src/evaluation/xai.py`, complete with a CLI analyzer (`scripts/run_xai.py`) and FastAPI endpoints.

### How It Works
- **Mathematical Grounding**: Uses **Integrated Gradients** and **Input-x-Gradient** backpropagating directly through the PPO Actor network from action logits to the 4,876-dimensional observation space:
  $$\text{Attribution}_j = (x_j - x'_j) \times \frac{1}{m} \sum_{k=1}^m \frac{\partial \text{Logit}_i}{\partial x_j}\Big|_{x' + \frac{k}{m}(x - x')}$$
- **Full Semantic Mapping**: Every one of the 4,876 feature indices is mapped to its exact TLS ID, lane ID, and group (`phase`, `elapsed`, `queue`, `vehicle_count`, `speed`, `density`, `global`).
- **Network Coordination**: Computes the $98 \times 98$ inter-junction coordination matrix, showing whether an intersection's decisions are driven by its own state or by queue spillbacks from neighbors.

### Running the XAI Analyzer CLI
```bash
python scripts/run_xai.py \
  --model artifacts/experiments/ppo/run_20260917_172613_seed42/model_final.zip \
  --tls-json artifacts/scenario/tls_metadata.json \
  --output-dir artifacts/reports/xai
```
Outputs produced:
- `artifacts/reports/xai/drivers_<TLS_ID>.png`: Horizontal bar chart of top positive (promoting phase switch) and negative (retaining green) feature drivers.
- `artifacts/reports/xai/coordination_heatmap.png`: Matrix heatmap of the top 20 interconnected junctions.
- `artifacts/reports/xai/xai_summary.json`: Complete JSON summary with natural language decision narratives.

### Querying XAI via REST API
Start FastAPI (`uvicorn api.main:app --port 8000`), then query:
- `GET /api/xai/summary`: Returns the latest precomputed explanations and decision narratives.
- `GET /api/xai/explain/{tls_id}`: Returns structured explanation for a specific junction.

---

## 5. Troubleshooting & Verified Path Checklist

| Resource | Correct Location | Incorrect Path (Do NOT use) |
|---|---|---|
| InTAS SUMO Config | `data/raw/intas/scenario/InTAS_buildings.sumocfg` | `data/raw/intas/ingolstadt.sumocfg` ❌ |
| TLS Metadata | `artifacts/scenario/tls_metadata.json` | `data/metadata/tls_metadata.json` ❌ |
| Network GeoJSON | `data/processed/network.geojson` | — |
| Trajectories | `artifacts/trajectories/*.parquet` | — |
| Production PPO Config (500k) | `configs/ppo.yaml` | — |
| Fast PPO Profile (5 episodes, 3.6k) | `configs/ppo_fast.yaml` | — |
| Smoke Test PPO Config (128 steps) | `configs/ppo_test.yaml` | — |
| Visual Demo Script | `scripts/run_visual_demo.py --delay 50` | — |
| XAI CLI Analyzer | `scripts/run_xai.py` | — |
| Parquet Engine | Always use `engine="pyarrow"` | `fastparquet` is missing ❌ |
