# 🌟 CityMind AI — Autonomous City-Wide Traffic Signal Control

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-PPO-brightgreen?style=for-the-badge)](https://stable-baselines3.readthedocs.io/)
[![SUMO](https://img.shields.io/badge/SUMO-Microscopic_Physics-blue?style=for-the-badge)](https://eclipse.dev/sumo/)
[![React](https://img.shields.io/badge/React_19-Vite-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![Proprietary Architecture](https://img.shields.io/badge/Architecture-Patent--Pending_MARL-gold?style=for-the-badge)](#-patented--proprietary-marl-architecture)
[![Statistical Victory](https://img.shields.io/badge/Paired_Seeds_Won-10%20%2F%2010%20(p=0.002)-success?style=for-the-badge)](#-proven-10-seed-statistical-victory)

**Decentralized Multi-Agent Deep Reinforcement Learning for Real-World Urban Traffic Grid Optimization**  
*Beating Calibrated Civil-Engineering Actuation on 98 Heterogeneous Intersections in Ingolstadt, Germany*

[🚀 Quickstart](#-quickstart--live-demonstration) • [📊 Benchmark Results](#-proven-10-seed-statistical-victory) • [🧠 Patented Architecture](#-patented--proprietary-marl-architecture) • [🎓 Faculty Defense Guide](#-faculty-defense--academic-distinction) • [🖥️ Web Showcase](#-interactive-web-showcase--3d-brain)

</div>

---

## 🏆 Project Overview & Star Achievement

**CityMind AI** is the culmination of months of intensive, single-author theoretical derivation and full-stack systems engineering. It stands as one of the most thoroughly hyperparameter-tuned, mathematically grounded Multi-Agent Reinforcement Learning (MARL) control systems developed by a solo researcher.

While most academic traffic RL literature evaluates on artificial, 4-way synthetic toy grids with uncalibrated fixed timers, **CityMind AI confronts real-world urban complexity**:
* **Real Metropolitan Scale**: 98 structurally heterogeneous signalized intersections across Ingolstadt, Germany (InTAS network).
* **Extreme Realism**: Intersections range from 4 to 19 lanes, varying green phases (2 to 6), short 20m stop lines, Krauss car-following dynamics, and 15,000+ scheduled morning peak vehicles (07:00–08:00).
* **The Gold Standard Baseline**: Evaluated against Eclipse SUMO's calibrated **vehicle-actuated controller with induction loops and gap-out logic** (the civil engineering industry standard).
* **Undeniable Victory**: **Won 10 out of 10 independent random seeds ($p=0.002$)**, slashing mean waiting times by **41.1%**, saving **41.4 seconds of lag per vehicle**, and clearing 256 trapped vehicles out of gridlock.

---

## 🎓 Faculty Defense & Academic Distinction

CityMind AI was presented for formal academic and defense committee review, earning **exceptional commendation from faculty reviewers** for its theoretical rigor, honest scientific methodology, and live working demonstration:

* **Eliminating Survivor Bias**: Addressed the hidden flaw in academic literature where models claim high performance by only measuring vehicles that finish their trips while hiding side-street gridlock. We created the **Drain Metrics Protocol** (`src/evaluation/drain_metrics.py`), accounting for 100% of scheduled vehicles.
* **Master Defense Presentation Materials**:
  * 📑 [**Master Defense Slide Deck (PDF)**](docs/CityMind_AI_Professor_Defense_Master.pdf) — 19-slide publication-grade presentation with complete speaker scripts.
  * 🖥️ [**Master Defense PowerPoint (PPTX)**](docs/CityMind_AI_Professor_Defense_Master.pptx) — Fully editable 16:9 widescreen presentation with embedded figures and presenter notes.
  * 🌐 [**Interactive Browser Deck**](docs/presentation_deck.html) — Standalone HTML presentation with slide switcher and speaker notes (`S` key).
  * ⏱️ [**5-Minute Rapid Oral Defense Guide**](docs/PROJECT_REVIEW_5_MINUTES_DEFENSE_GUIDE.md) — Targeted defense strategy and cheat sheet.
  * 🔬 [**Comprehensive Policy Learning Guide**](docs/LEARNING_GUIDE_ALL_POLICIES.md) — 40-page technical treatise detailing all policies and ablation benchmarks.

---

## 🧠 Patented & Proprietary MARL Architecture

The core control intelligence is defined in [**`configs/ippo.yaml`**](configs/ippo.yaml) and [**`src/agents/`**](src/agents/) as a **Proprietary & Patent-Pending Multi-Agent Reinforcement Learning Coordination Protocol**:

![IPPO Hyperparameters & Mathematical Architecture](docs/ippo_hyperparameters_guide.png)

### The 3 Master Equations Solved Simultaneously

1. **Total PPO Objective Loss**:
   $$\mathcal{L}^{\text{TOTAL}}(\theta) = \hat{\mathbb{E}}_t \left[ \mathcal{L}^{\text{CLIP}}(\theta) - c_{\text{vf}} \cdot \mathcal{L}^{\text{VF}}(\theta) + c_{\text{ent}} \cdot \mathcal{S}[\pi_\theta](s_t) \right]$$

2. **Clipped Surrogate Policy Objective**:
   $$\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right], \quad r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

3. **Generalized Advantage Estimation (GAE)**:
   $$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V, \quad \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$

### Key Mathematical & Architectural Pillars

* **High-Throughput Rollout Scale**: 6 parallel SUMO simulation processes (`num_vec_envs = 6`) run simultaneously, meaning **588 intersection agents act on every single simulation step**. A 180-step rollout yields **105,840 transitions per update** across 8 minibatches of 13,230 samples. Total training encompasses **10,584,000 steps** (100 PPO updates = 150 simulated city hours).
* **Extended Planning Horizon ($\gamma = 0.99$)**: At 5-second decision steps, $\frac{1}{1 - 0.99} = 100\text{ steps} = 500\text{ s}$ (**~8.3 minutes**). This allows the agent to look ahead beyond immediate queues and coordinate multi-intersection **green waves**.
* **Split Gradient Clipping (`split_clip.py`)**: *The core discovery.* In traffic control, large delay penalties cause Critic gradients to surge to $\|\nabla \theta_{\text{critic}}\| \approx 3,830$, while Actor gradients are small ($\approx 0.19$). Standard global clipping scaled all gradients down by $\frac{0.5}{3830} \approx 1.3 \times 10^{-4}$, starving the Actor below Adam's epsilon. `SplitClipPPO` clips Actor ($\le 0.5$) and Critic ($\le 0.5$) **independently**, accelerating convergence by **10×**.
* **300m Feeder Road Crawler**: Short 20m stop-line lanes (~2 cars) hide massive 60-car traffic queues behind them. Our graph crawler walks 300m upstream through unsignalized feeder roads to incorporate true approach demand into the Varaiya pressure metric.
* **Deterministic Safety Layer**: Decouples the neural network from raw light signals. Enforces 5s minimum green (prevents flickering), 60s maximum green (prevents starvation), and hardware-level 3s yellow and 1s all-red clearance intervals. **Zero collisions guaranteed.**

---

## 📊 Proven 10-Seed Statistical Victory

Evaluated across 10 independent random seeds (40 full-city microscopic simulation runs) during the 07:00–08:00 morning rush hour:

| Metric (07:00–08:00 Peak) | SUMO Actuated (Baseline) | Fixed-Time | **CityMind Model A** | **Improvement vs Baseline** | Paired Seed Significance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Lag per Scheduled Car** | 276.8 s | 298.7 s | **235.4 s** | **-41.4 s (-15.0%)** | **10 / 10 Seeds Won ($p = 0.002$)** |
| **Mean Waiting Time** | 6.39 s | 7.14 s | **3.77 s** | **-2.62 s (-41.1%)** | **10 / 10 Seeds Won ($p = 0.002$)** |
| **City Backlog at 08:00** | 3,108 cars | 3,363 cars | **2,852 cars** | **-256 trapped cars** | **10 / 10 Seeds Won ($p = 0.002$)** |
| **Time Loss per Finished Car** | 336.6 s | 363.9 s | **286.4 s** | **-50.2 s (-14.9%)** | **10 / 10 Seeds Won ($p = 0.002$)** |
| **Mean Queue Length** | 0.440 cars | 0.492 cars | **0.318 cars** | **-0.122 cars (-27.7%)** | **10 / 10 Seeds Won ($p = 0.002$)** |
| **Mean Vehicle Speed** | 8.70 m/s | 8.40 m/s | **9.37 m/s** | **+0.67 m/s (+7.7%)** | **10 / 10 Seeds Won ($p = 0.002$)** |
| **Trips Completed** | 11,585 | 11,432 | **11,842** | **+258 finished cars** | **10 / 10 Seeds Won ($p = 0.002$)** |
| **Overdue Vehicles** | 1,853 | 2,128 | **1,597** | **-256 overdue cars** | **10 / 10 Seeds Won ($p = 0.002$)** |

> *Statistical Note: Under the exact non-parametric sign test, winning 10/10 seeds yields $p = 0.002$. The 95% paired $t$-confidence interval for lag reduction is $[-43.9\text{s}, -38.9\text{s}]$, bounded far below zero.*

---

## 🖥️ Interactive Web Showcase & 3D Brain

CityMind AI features a modern **React 19 + Vite + Three.js** web application with a **FastAPI** backend:

* **Executive Showcase (`/`)**: One-click side-by-side replay comparing SUMO Actuated, Fixed-Time, Model A, and Model B running simultaneously at 10× speed.
* **Living 3D Neural Scene (`/model`)**: Interactive Three.js 3D rendering of the Actor and Critic neural networks processing real live weights and activations.
* **SUMO-GUI Desktop Launcher**: Launch the real Eclipse SUMO graphical simulation directly from the web interface for single junctions or tiled multi-controller comparisons.
* **Analytics Dashboard (`/dashboard`)**: Deep-dive interactive Recharts graphs tracking queue progression, vehicle lag distributions, and spatial heatmaps.

---

## 🚀 Quickstart & Live Demonstration

### Prerequisites
* **OS**: macOS or Linux
* **Python**: 3.11+ (managed via `uv` or `venv`)
* **Simulation**: Eclipse SUMO 1.20+ (`brew install sumo` or distribution packages)
* **Node.js**: LTS (for frontend dashboard)

### 1. Installation & Environment Setup
```bash
# Clone the repository
git clone https://github.com/parthsharma8368/city-mind-ai.git
cd city-mind-ai

# Automatic venv, dependency install, and SUMO configuration
make install && make sumo && source .env.sumo
```

### 2. Launch Localhost Web App & Backend
```bash
# Terminal 1: Launch FastAPI backend (port 8000)
make run-api

# Terminal 2: Launch Vite React frontend (port 5173)
make run-frontend
```
Open [**http://localhost:5173/**](http://localhost:5173/) to run the showcase.

### 3. Run Live SUMO-GUI Graphical Simulation
```bash
# Watch the trained IPPO agent control real Ingolstadt traffic in real-time
python scripts/run_visual_demo.py --controller ippo --model artifacts/models/model_final.zip --render
```

### 4. Reproduce Head-to-Head Evaluations
```bash
# Run 10-seed paired evaluation against SUMO Actuated
python scripts/evaluate.py --controller sumo_actuated --out artifacts/eval/sumo_actuated
python scripts/evaluate.py --controller ippo --model artifacts/models/model_final.zip --out artifacts/eval/ippo
```

---

## 📁 Repository Structure

```text
├── configs/
│   ├── ippo.yaml                 # Patent-pending IPPO hyperparameter architecture
│   ├── environment.yaml          # 22-dim state, reward weights & safety constraints
├── src/
│   ├── agents/                   # IPPO agent, Split-Clip PPO optimizer, model contracts
│   ├── environment/              # SUMO TraCI/libsumo proxy, safety layer, 300m feeder crawler
│   ├── controllers/              # SUMO Actuated & Fixed-Time baseline controllers
│   ├── evaluation/               # Drain metrics protocol, paired statistics & tripinfo parsers
│   └── scenario/                 # Ingolstadt network metadata & signal junction specifications
├── api/                          # FastAPI REST & WebSocket server with security middleware
├── frontend/                     # React 19 + TypeScript + Vite + Three.js 3D web application
├── docs/                         # Defense presentations (.pptx, .pdf, .html) and technical reports
├── scripts/                      # Training harnesses, visual demo launcher, figure generators
└── tests/                        # Comprehensive test suite (unit tests run without SUMO)
```

---

## 📜 Intellectual Property & Licensing

* **Core RL Algorithms & Control Architecture**: Distributed under the [Apache-2.0 License](LICENSE).
* **Proprietary Technology**: The hyperparameter recipe, spatial feeder crawler, and decoupled gradient clipping mechanism are documented as proprietary intellectual property.
* **InTAS Traffic Scenario**: Licensed under GPL-3.0 (see [NOTICE](NOTICE)).
* **Security & Responsible Disclosure**: See [SECURITY.md](SECURITY.md).

<div align="center">
<b>Developed with passion by Parth Sharma</b><br>
<i>Decentralized Intelligence for Future Urban Mobility</i>
</div>
