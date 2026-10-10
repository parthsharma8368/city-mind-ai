> **Historical / superseded document.** Pre-implementation plan (20 intersections, centralized controller, "prove RL outperforms", Python 3.12, DQN). The real project uses 98 intersections, Python 3.11-3.12, PPO/IPPO/MaskablePPO, and the final result is that no RL controller beats calibrated actuated control. Kept for history only. See [README.md](../../README.md) and [docs/](../).

# CityMind AI: Centralized RL for Traffic Signal Control

**Executive Summary:** This project aims to build a **centralized deep-RL traffic signal controller** in the realistic Ingolstadt (InTAS) SUMO scenario, and demonstrate that it outperforms conventional fixed-time control under varying demand. InTAS is a validated real-city network with 20 signalized intersections. We will use SUMO (via `libsumo`/TraCI) as the simulation engine, and train one RL policy that observes the *entire network state* and outputs joint actions for all signals. The RL agent (e.g. PPO or DQN variants) will be compared against **baselines**: fixed-time schedules, an actuated controller, and a Max-Pressure adaptive controller. The plan covers data setup, environment design, model training, and rigorous evaluation (multiple traffic scenarios and random seeds). We will collect metrics (travel time, waiting time, queues, throughput, etc.) and analyze statistical significance of improvements. Finally, we will store results in Parquet and build a FastAPI+React dashboard to replay and compare signal control strategies.

## Motivation and Research Question  
Urban traffic congestion is a growing problem. Conventional fixed-time traffic lights cannot adapt to fluctuations in demand, leading to inefficiencies. Recent work suggests RL can adaptively optimize signals. Our research question is:  
*“Can a single **centralized** deep-RL agent, with full knowledge of the traffic network, jointly control all signals to **reduce congestion** (e.g. lower travel times and queues) compared to fixed-time or rule-based controllers?”*  

The project scope is one city-wide scenario (InTAS ~19–20 intersections) under multiple demand levels. Success criteria include: statistically significant reduction in average travel/waiting time across test scenarios and seeds, and robust performance (not overfitting to one demand). We will quantify success as percentage improvement over baselines in key metrics (e.g. ≥10–20% reduction in travel time).

## Dataset / Environment (InTAS)  
We use **InTAS** (Ingolstadt Traffic Scenario for SUMO) as our environment. InTAS is an open-source SUMO scenario (GitHub: *silaslobo/InTAS*) representing a real German city with **20 traffic lights** and validated against real measurements. The InTAS paper notes: “InTAS road topology accurately represents Ingolstadt’s real road map... Twenty traffic lights systems were simulated according to the real program deployed on the traffic lights”. The scenario includes 24h traffic demand, real bus routes, pedestrian flows, and detection loops. Key files in the InTAS scenario (from the `scenario/` directory) include:

- **`ingolstadt.net.xml`** – the road network (junctions, lanes, edge geometry) defining Ingolstadt’s roads and 20 intersections.  
- **Route files (`.rou.xml`)** – vehicle demand. In InTAS, these are split by mode: e.g. `routes/ped.rou.xml` (pedestrians), `routes/BusRoutes.flow.xml` (buses), and `routes/InTAS_00x.rou.xml` (cars). These describe vehicle types, departure times, and routes.  
- **SUMO config (`.sumocfg`)** – e.g. `InTAS_buildings.sumocfg` includes the net-file and all route-files. It also references **additional files** with detectors (`InTAS_E1.add.xml` contains induction-loop sensors) and bus-stop geometry.  
- **`*.add.xml`** – additional objects, such as detectors and bus stations. For example, `InTAS_E1.add.xml` defines all traffic detectors (E1 loops) that we can use to measure queue lengths.  

We will assume a subset of 19 intersections for control (as used in prior work), leaving one as a reference or ungated intersection. Unless otherwise noted, we plan to control all signalized junctions in the scenario.

**Links:** [InTAS GitHub](https://github.com/silaslobo/InTAS){target=_blank}; InTAS paper (SUMO Conf. 2020).

## Software Stack and Installation

The system uses Python and established ML/traffic tools. Essential components:

- **Python 3.10+** (we use **3.12**). [Stable-Baselines3](https://stable-baselines3.readthedocs.io) requires Python ≥3.10.
- **SUMO** (Simulation of Urban Mobility) as the traffic simulator. We can install via package manager or pip (`pip install eclipse-sumo` which includes sumo, sumo-tools, sumo-doc). Example (Ubuntu):  
  ```bash
  sudo add-apt-repository ppa:sumo/stable
  sudo apt update
  sudo apt install sumo sumo-tools sumo-doc
  ```  
  *Note:* After install, set `SUMO_HOME` (e.g. `export SUMO_HOME="/usr/share/sumo"`).  
- **TraCI / libsumo**: We will use **TraCI** API to control SUMO, and for speed we enable **libsumo** (in-memory SUMO). After installing SUMO, set:  
  ```bash
  export LIBSUMO_AS_TRACI=1
  ```  
  This can yield ~8× speedup (but disables `sumo-gui`).  
- **SUMO libraries** (via pip): `pip install sumolib traci libsumo`. This gives the Python modules to interface with SUMO.  
- **Stable-Baselines3**: for RL algorithms (PPO, DQN, etc.). Install via pip:  
  ```bash
  pip install 'stable-baselines3[extra]'
  ```  
  (This ensures optional deps like TensorBoard). Stable-Baselines3 (latest, ≥2.0) requires PyTorch ≥2.8.  
- **Other ML libs**: `pip install gymnasium sb3-contrib` (SB3-contrib adds extra algos like QR-DQN/A2C variants). Also `numpy`, `pandas`, `pyarrow` (for data), `matplotlib`, etc. Example:  
  ```bash
  pip install gymnasium sb3-contrib numpy pandas pyarrow matplotlib seaborn
  ```  
- **CityFlow (optional)**: A high-performance city-scale simulator. We mention CityFlow because it provides benchmark networks (Hangzhou, Manhattan, etc.) and is ~25× faster than SUMO on large networks. We can optionally `pip install cityflow` to validate ideas quickly on smaller grids (e.g. Hangzhou 4×4) before scaling to InTAS.

All software with version requirements:

| Software / Library    | Version / Notes                                             |
|----------------------|--------------------------------------------------------------|
| Python               | ≥3.10 (we use 3.12)                              |
| SUMO                 | Latest stable (InTAS was tested on SUMO 1.7–1.10) |
| TraCI / libsumo      | Comes with SUMO (enabled via `LIBSUMO_AS_TRACI=1`) |
| Gymnasium            | Latest stable (>=1.0)                                        |
| Stable-Baselines3    | ≥2.0 (requires PyTorch ≥2.8)                      |
| SB3-Contrib          | Latest for additional algorithms                              |
| PyTorch              | ≥2.8 (compatible with SB3)                      |
| NumPy, Pandas        | Latest stable                                               |
| pyarrow              | Latest (for Parquet)                                       |
| CityFlow (optional)  | Latest (see CityFlow repo)                    |

## Implementation Roadmap

We outline the development steps from data setup through evaluation:

1. **Set up the environment:** Clone the InTAS repo and install dependencies.  
   ```bash
   git clone https://github.com/silaslobo/InTAS.git
   cd InTAS
   ```
   Open the SUMO scenario in GUI to verify it runs (no RL yet):  
   ```bash
   sumo-gui -c scenario/InTAS_buildings.sumocfg
   ```  
   (This should display the Ingolstadt map with vehicles and 20 traffic lights.) Ensure no errors.

2. **Understand the InTAS scenario:** Inspect key files:  
   - `ingolstadt.net.xml` (junctions and TLS),  
   - routes in `scenario/routes/`,  
   - `InTAS_buildings.sumocfg` which lists net-file, route-files, and additional detectors.  
   Write a small Python script using `sumolib`/TraCI to list all TLS IDs and their phases. For example:  
   ```python
   import traci
   traci.start([sumoBinary, "-c", "scenario/InTAS_buildings.sumocfg"])
   tls_ids = traci.trafficlight.getIDList()
   print("TLS:", tls_ids)
   for tls in tls_ids:
       logic = traci.trafficlight.getAllProgramLogics(tls)[0]
       print(tls, "phases:", [p.state for p in logic.getPhases()])
   traci.close()
   ```  
   Record the number of intersections (should be 20) and their phase configurations.

3. **Build baseline controllers:** Before RL, implement and test conventional controllers.  
   - *Fixed-time controller:* Use the schedule given in InTAS (if known) or define a simple cycle (e.g. NS-green 30s, yellow 5s, EW-green 30s, yellow 5s). In SUMO you can let the default program run by not issuing TraCI changes. Alternatively, explicitly set each TLS to follow a repeating program via TraCI.  
   - *Actuated controller:* If detectors are available (InTAS has loop detectors), implement a simple actuated logic (e.g. extend green if queue present). SUMO allows changing phases in code.  
   - *Max-Pressure controller:* Implement or adapt a known algorithm that, at each decision interval, calculates pressure (difference in queue lengths) for each phase and chooses the max. This is a strong adaptive baseline often used. The CityFlow repo’s Hangzhou results use MaxPressure with good effect.  

   Evaluate each baseline in SUMO (headless or GUI) across a few runs (varying seed) to gather metrics. This establishes reference values (we expect RL should beat them).

4. **Design the centralized Gym environment:** We will create a custom Gymnasium environment class that wraps SUMO. Key elements:  
   - **Observation (state):** At each timestep (or decision step), the agent observes the *entire network*. For each controlled intersection, include e.g.: current phase (one-hot), elapsed time in phase, and aggregated traffic info on incoming lanes (vehicle count or density, queue length, average speed). Also include any global stats (total queued vehicles). For example, the state vector might be:  
     ```
     [TLS1_phase_onehot, TLS1_elapsed, TLS1_lane1_count, ... , TLS1_laneN_count,
      TLS2_phase_onehot, ... , TLS19_phase, TLS19_lane_counts, global_queue, ...]
     ```  
     In SUMO-RL, each agent’s state vector includes one-hot phase, lane densities and queues. We extend this to all intersections in one large vector.  
   - **Action space:** A joint action for all TLS. To keep it tractable, we use a *binary* decision per intersection: “*extend current phase* (0)” or “*switch to next phase* (1)”. This is effectively a **MultiBinary** action of length = number of controlled intersections. (We could also use MultiDiscrete if we allow more options, but binary is simpler.) We enforce safety: a phase change automatically triggers the required yellow and all-red sub-phases in SUMO, and we respect minimum green times. Note from SUMO-RL that after a change, a fixed yellow is applied.  
   - **Reward function:** A network-wide metric. We use (negative) cumulative waiting time or delay. For example, at step *t* let \(W_t\) be the total waiting time of all vehicles. One reward is \(- (W_t - W_{t-1})\), i.e. reduction in waiting time. We can also penalize switching (to avoid rapid oscillations). So a full reward might be:  
     \[
     R_t = \alpha (W_{t-1}-W_t) - \beta\,(\text{total queue length}) - \gamma\,(\text{switches})
     \]  
     This encourages reducing delay and queues, while discouraging needless phase changes. SUMO-RL’s default is change-in-delay, which we can adopt as a baseline.  
   - **Step logic:** In each `step()`, we send the agent’s joint action to SUMO (via TraCI: e.g. for each TLS, either call `trafficlight.setPhase()` to advance or do nothing to extend). Then we run the SUMO simulation forward for a fixed interval (e.g. `delta_time = 5s`) to let vehicles move. At the next step, we collect the new global state and compute reward. Continue until end of simulation.

   We will code this as a `gymnasium.Env` (or PettingZoo for consistency, but Gym single-agent mode is fine since it’s one agent controlling all). A minimal skeleton might look like:  
   ```python
   import gymnasium as gym

   class CentralizedTrafficEnv(gym.Env):
       def __init__(self, net_file, route_files, **kwargs):
           super().__init__()
           # Load SUMO config, TLS IDs, detectors, etc.
           # Define self.observation_space and self.action_space
       def reset(self):
           # Start SUMO (with libsumo) and return initial observation
       def step(self, action):
           # Apply action to all TLS via TraCI
           # Step SUMO for N seconds
           # Compute new observation, reward, done, info
           return obs, reward, done, False, info
   ```  
   We will thoroughly test the environment with random actions: ensure `reset()`, `step()` run without errors and outputs reasonable values.

5. **RL Training Regimen:** Using the environment above, we train RL agents. Key points:  
   - **Algorithms:** Focus on discrete-action algorithms. We plan to train **DQN variants** (DQN, Double DQN, Dueling DQN) and **PPO** (plus optionally A2C). Note: SAC/TD3 are for continuous actions and less natural here. SB3 has implementations for DQN and PPO.  
   - **Hyperparameters:** Start with standard values: e.g. MLP policy with two hidden layers (128-256 units each). Learning rate ~3e-4 (tune later). Use **5s** decision intervals initially (action every 5s).  
   - **Training steps:** Preliminary run ~500k steps to debug, then scale up (1–2M steps) until convergence (monitor reward curves in TensorBoard).  
   - **Random seeds:** Use at least 5 different seeds for each algorithm to estimate variability.  
   - **Traffic demand variants:** Avoid overfitting to one scenario. We will create *demand-scaling variants*. For example, multiply all vehicle departure rates by factors {0.75, 1.0, 1.25, 1.5}. Train on a mix of 0.75–1.3× flows, and test generalization on intermediate levels (e.g. 0.9, 1.1).  
   - **Training loop:** Pseudocode example for PPO:  
     ```python
     from stable_baselines3 import PPO
     env = CentralizedTrafficEnv(net_file="...", route_files=[...])
     model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
     model.learn(total_timesteps=1_000_000)
     ```  
     We will similarly train DQN variants. Use `stable-baselines3[extra]` for TensorBoard and monitoring.

6. **Evaluation Protocol:** After training, we evaluate all controllers under controlled conditions:  
   - **Controllers:** Fixed-time, Actuated, MaxPressure, and each RL model (DQN, DDQN, Dueling-DQN, PPO, A2C).  
   - **Traffic scenarios:** Test on multiple demand levels (e.g. 0.75x, 1.0x, 1.25x, 1.5x) that may differ from training.  
   - **Random seeds / episodes:** For each (controller, demand) combination, run **5** different random seeds/episodes. Use the same SUMO config except randomizing vehicle departure orders (or initial seeds).  
   - **Metrics to collect:** For each run, compute:  
     - Average travel time of all vehicles,  
     - Average waiting time, total waiting time,  
     - Average and maximum queue length (across detectors),  
     - Throughput (vehicles arrived),  
     - Number of stops (vehicles that had to stop), etc.  
   These can be extracted from SUMO’s outputs or using TraCI (vehicle speed/waiting info).  
   - **Statistical analysis:** For each metric and controller, compute the mean and standard deviation over seeds. Compare RL vs baseline with percentage improvement. If needed, perform t-tests to confirm significance of differences.  

7. **Results Storage:** Save raw simulation data and summary metrics. Use efficient formats: e.g. store time-series trajectories in Parquet files (timestamped vehicle counts, signal states, etc.). Store summary results (one row per run: controller, demand, seed, metrics) in CSV/Parquet for analysis. This enables quick plotting and web display later.

8. **Visualization and Dashboard:** We will build a demo dashboard. Approach: after evaluation, we have saved “replay” data for each run (e.g. signal states and vehicle positions at each second). A backend (FastAPI) can serve this data. A React frontend (CityMind AI app) will allow switching between controllers and replaying simulation progress (similar to a map with moving vehicles and signal lights). The dashboard will also show summary stats for the current run (avg wait, queue, etc.) and compare % differences (e.g. “Waiting time: –20% vs fixed”). This makes the results tangible.

9. **Timeline and Milestones:**  
   - *Weeks 1–2:* Setup SUMO/InTAS, run baselines, collect baseline metrics.  
   - *Weeks 3–4:* Develop and debug centralized Gym env (state, action, reward). Validate with random policy.  
   - *Weeks 5–6:* Train PPO (initial 500k, then up to 1–2M). Evaluate intermediate results.  
   - *Weeks 7–8:* Train additional agents (DQN, DDQN, A2C). Optimize hyperparameters.  
   - *Weeks 9–10:* Run full evaluation (all controllers × demands × seeds). Perform analysis and plotting.  
   - *Weeks 11–12:* Build visualization (FastAPI + React), finalize report.  

Below are the requested tables:

**(a) Experimental Matrix:** Controllers vs. demand scenarios (5 seeds each).  

| Controller        | 0.75× demand (5 runs) | 1.00× (5 runs) | 1.25× (5 runs) | 1.50× (5 runs) |
|-------------------|-----------------------|----------------|----------------|----------------|
| Fixed-time        | ✓                     | ✓              | ✓              | ✓              |
| Actuated          | ✓                     | ✓              | ✓              | ✓              |
| Max-Pressure      | ✓                     | ✓              | ✓              | ✓              |
| DQN (from scratch)| ✓                     | ✓              | ✓              | ✓              |
| Double DQN        | ✓                     | ✓              | ✓              | ✓              |
| Dueling DQN       | ✓                     | ✓              | ✓              | ✓              |
| PPO               | ✓                     | ✓              | ✓              | ✓              |
| A2C               | ✓                     | ✓              | ✓              | ✓              |

**(b) Software/Dependencies:** (Examples shown; versions may update)  

| Component        | Version/Notes                                           |
|------------------|----------------------------------------------------------|
| Python           | 3.12 (>=3.10 for SB3)                        |
| SUMO             | 1.12 (tested 1.7–1.10 by InTAS)             |
| SUMO tools       | sumolib, traci, libsumo (installed via pip) |
| Gymnasium        | Latest (>=1.0)                                          |
| PyTorch          | ≥2.8 (for SB3)                              |
| Stable-Baselines3| ≥2.0 ('`pip install stable-baselines3[extra]`')|
| SB3-Contrib      | Latest (for algorithms like D4PG etc.)                   |
| Numpy/Pandas     | Latest stable                                           |
| PyArrow          | Latest (for Parquet)                                    |
| CityFlow         | Latest (optional; Python interface for RL; 25× faster vs SUMO) |

**Key References:** 
- InTAS GitHub and documentation. 
- InTAS paper (SUMO Conf 2020). 
- CityFlow project (fast RL traffic sim). 
- SUMO-RL repo (observations/actions). 
- Stable-Baselines3 install guide. 
- Traffic-signal benchmark data (Hangzhou 4×4). 

All implementation will emphasize **clean, modular code** and thorough testing. By following this plan, we ensure a rigorous, reproducible study of centralized RL for traffic control using the InTAS dataset, with clear evidence (metrics and visualizations) of performance against standard baselines. Each choice (e.g. full-network state, binary actions, reward design) is grounded in traffic-RL literature and SUMO best practices.