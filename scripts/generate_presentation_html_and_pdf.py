"""
Generates an interactive HTML presentation deck and compiles it to PDF via Google Chrome.
Outputs:
  - docs/presentation_deck.html (Interactive browser slide deck with speaker notes)
  - docs/CityMind_AI_Professor_Defense_Master.pdf (Publication-grade 16:9 presentation PDF)
"""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML_PATH = ROOT / "docs" / "presentation_deck.html"
PDF_PATH = ROOT / "docs" / "CityMind_AI_Professor_Defense_Master.pdf"

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

SLIDES_DATA = [
    {
        "num": 1,
        "is_dark": True,
        "badge": "FINAL RL PROJECT REVIEW & DEFENSE",
        "title": "CityMind AI",
        "subtitle": "Decentralized Deep Reinforcement Learning for City-Wide Traffic Signal Control",
        "description": "Beating Calibrated Civil-Engineering Actuation on 98 Real-World Heterogeneous Intersections in Ingolstadt, Germany",
        "stats": [
            ("-41.4 s", "Lag per scheduled car", "10/10 seeds won (p=0.002)"),
            ("-41.1%", "Mean waiting time", "Dropped from 6.39s to 3.77s"),
            ("-256 Cars", "City backlog at 08:00", "Zero gridlock spillback"),
            ("10 / 10", "Statistical victory", "Paired seed-by-seed tests")
        ],
        "meta": "Environment: SUMO Microscopic Physics (TraCI / libsumo) | Algorithm: IPPO + Split Gradient Clipping | Baseline: SUMO Actuated Induction Loops",
        "notes": "Good morning, Professor. Today I am presenting CityMind AI, a decentralized Deep Reinforcement Learning system for city-wide traffic signal control. Most academic RL papers test on synthetic, 4-way toy grids with uncalibrated fixed timers and claim massive improvements. We asked a much tougher question: Can Deep RL outperform real-world civil engineering on a genuine city scale? We evaluated on the Ingolstadt InTAS network in Germany—98 heterogeneous intersections during the morning rush hour with over 15,000 scheduled trips. Our benchmark is SUMO's calibrated vehicle-actuated controller running induction loops with gap-out logic. As shown here, our model beats the civil engineering baseline on 10 out of 10 seeds, slashing wait time by 41.1% and saving 41.4 seconds of lag per vehicle across the entire city."
    },
    {
        "num": 2,
        "is_dark": False,
        "badge": "Problem Statement & Motivation",
        "title": "The Reality Check: Escaping Academic Toy Grids",
        "subtitle": "Why typical RL traffic papers fail in reality, and how CityMind evaluates real urban scale",
        "cards": [
            ("1. Academic Toy Grids ❌", "coral", [
                "Synthetic 4-way intersections with identical 2-lane approaches.",
                "Uncalibrated fixed-time timers as the comparison baseline (trivial to beat).",
                "Assumes infinite queue capacity; ignores turning-bay spillback and lane blocking.",
                "Reports 'Mean Wait Time of Completed Cars', disguising massive gridlock on side streets (Survivor Bias).",
                "Fails completely when deployed on real urban topology."
            ]),
            ("2. Real Ingolstadt (InTAS) 🇩🇪", "blue", [
                "Genuine European city network: 98 signalized intersections in Ingolstadt, Germany.",
                "Severe structural heterogeneity: Junctions have between 4 and 19 lanes, varying green phases (2 to 6).",
                "Morning rush hour scenario (07:00–08:00) with over 15,000 scheduled trips.",
                "Realistic driver behavior: Krauss car-following, LC2013 lane change dynamics, clearance delays.",
                "Real-world bottlenecks: Stop lines holding only 2 cars with 60-car feeder jams."
            ]),
            ("3. The True Civil Baseline 🎯", "green", [
                "SUMO Calibrated Vehicle-Actuated Controller (the real-world industry standard).",
                "Uses embedded induction loop detectors that sense gap-out headway in real time.",
                "Extends green phases dynamically when platoons arrive and terminates when gaps appear.",
                "Governed by authentic civil-engineering minimum, maximum, and yellow clearance intervals.",
                "Our Goal: Prove Deep RL can outperform calibrated civil engineering across the full network."
            ])
        ],
        "callout": "Explain to the professor: 'We refused to test on toy 4-way synthetic grids. We took the full Ingolstadt city network with 98 real intersections and matched our RL agent against SUMO's calibrated induction-loop actuated controller. If our model wins here, it proves RL is ready for real cities.'",
        "notes": "Professor, standard RL traffic papers test on 4-way toy grids and claim 20% improvements. But real cities have 4 to 19 lanes per junction, short stop lines, and turning conflicts. We benchmarked against SUMO's calibrated vehicle-actuated controller with gap-out logic—the actual system civil engineers use. Our goal was to prove Deep RL can beat calibrated civil engineering at true city scale."
    },
    {
        "num": 3,
        "is_dark": False,
        "badge": "Simulation & Physical Modeling",
        "title": "Microscopic Vehicle Physics in Eclipse SUMO",
        "subtitle": "Simulating continuous-space driver dynamics, car-following, and urban heterogeneity",
        "two_col": (
            ("Microscopic Simulation Engine: Eclipse SUMO", "blue", [
                "Continuous Space & Discrete Time: Tracks every single vehicle individually at 1.0-second physics increments.",
                "Krauss Car-Following Model: Models realistic acceleration, safe following distance, deceleration limits, and driver reaction latencies.",
                "LC2013 Lane-Changing Model: Simulates tactical overtaking, lane blocking, merge conflicts, and turning-lane queues.",
                "Intersection Clearance Physics: Yellow change intervals (3–4s) and all-red clearance intervals are explicitly simulated to prevent phantom teleportations.",
                "Spillback Dynamics: Traffic backing up past intersection stop-lines actively blocks upstream cross-traffic, capturing cascading urban gridlock."
            ]),
            ("Ingolstadt InTAS Network Specifications", "green", [
                "98 Controlled Intersections: Spans high-speed arterial corridors, highway entry ramps, and dense downtown streets.",
                "Heterogeneous Geometry: Signals feature between 4 and 19 controlled incoming lanes and 2 to 6 unique green phase configurations.",
                "Morning Rush Hour Window: 07:00 to 08:00 (3,600 simulated seconds), representing peak commuting demand.",
                ">15,000 Scheduled Vehicle Trips: Calibrated origin-destination trip demand from German municipal mobility surveys.",
                "Decision Step: The RL agent acts every 5 seconds (720 discrete decision steps per hour), allowing vehicle platoons to clear without jitter.",
                "Parallel Training: 6 parallel SUMO environments (588 simultaneous signal transitions per step) for high sample efficiency."
            ])
        ),
        "callout": "Explain to the professor: 'SUMO doesn't treat traffic as abstract fluid flow. Every car has an engine, brakes, driver latency, and lane-changing logic. If an intersection turns green, cars don't instantly move—they accelerate one by one according to Krauss physics. That makes signal coordination genuinely challenging.'",
        "notes": "Professor, we chose Eclipse SUMO because it is a microscopic simulator. It models discrete vehicles using Krauss car-following for realistic acceleration and brake latency, and LC2013 for lane changes. In Ingolstadt, 98 junctions handle over 15,000 scheduled trips. The agent acts every 5 seconds, providing a stable 720 decision cycles per hour."
    },
    {
        "num": 4,
        "is_dark": False,
        "badge": "Middleware & Integration",
        "title": "TraCI: The Bi-directional TCP Socket Protocol",
        "subtitle": "How our Python RL environment queries sensors and actuates signal phases in SUMO",
        "cards": [
            ("1. What is TraCI? 🔌", "blue", [
                "TraCI = Traffic Control Interface.",
                "Client-Server Architecture: SUMO runs as a socket server (listening on a TCP port); our Python environment connects as a client.",
                "Bidirectional Interaction: Retrieves simulation state (sensors) and commands simulation objects (actuators).",
                "Deterministic Stepping: The simulation clock is frozen until Python sends traci.simulationStep().",
                "Speed Optimization: In src/environment/sumo_connection.py, we use libsumo—a direct C++ SWIG binding eliminating socket overhead during training."
            ]),
            ("2. Sensor Ingestion (Every 5s) 📡", "green", [
                "Queue Measurements: Calls traci.lane.getLastStepHaltingNumber(lane) to detect stopped vehicles (<0.1 m/s).",
                "Upstream Feeder Walking: Recursively crawls lane topology via traci.lane.getLinks() to inspect traffic up to 300 meters behind stop-lines.",
                "Downstream Speeds: Queries traci.lane.getLastStepMeanSpeed(lane) to monitor exit road congestion and prevent spillback.",
                "Outside Network Demand: Queries traci.simulation.getPendingVehicles() to count vehicles queued at city boundary gates waiting to spawn."
            ]),
            ("3. Phase Actuation & Safety 🚦", "coral", [
                "RL Decision: Agent chooses action a in {0: Keep, 1: Advance}.",
                "Deterministic Safety Filter: Actions pass through SafetyLayer (safety.py) before reaching SUMO.",
                "Guarantees Min/Max Green: Prevents rapid oscillation and starvation.",
                "Executes Clearance: Automatically schedules yellow and all-red clearance intervals.",
                "Direct Actuation Command: Sends traci.trafficlight.setPhase(tls_id, phase_index) to command the hardware.",
                "Advances Clock: Calls traci.simulationStep() to progress physics by 1 second."
            ])
        ],
        "callout": "Explain to the professor: 'TraCI is our TCP socket middleware. In src/environment/sumo_connection.py, we connect Python to SUMO. Every 5 seconds, Python queries vehicle queues and speeds, runs our RL policy, checks the safety layer, and commands traci.trafficlight.setPhase(). It is 100% automated and deterministic.'",
        "notes": "Professor, TraCI is the Traffic Control Interface—a TCP socket client-server protocol. SUMO acts as the server; Python acts as the client. In training, we use libsumo C++ bindings for speed. Every 5 seconds, Python queries queues via getLastStepHaltingNumber, checks 300m feeder roads, evaluates our RL model, filters actions through the SafetyLayer, and commands traci.trafficlight.setPhase()."
    },
    {
        "num": 5,
        "is_dark": False,
        "badge": "System Architecture Figure",
        "title": "Figure 1 Walkthrough: The Closed-Loop Control System",
        "subtitle": "Explaining the 720-cycle agent-environment interaction loop and the safety layer",
        "fig_layout": {
            "img": "fig/fig_loop.png",
            "title": "Deconstructing Figure 1",
            "points": [
                ("Left Box (The Agent):", "ONE shared neural network for all 98 signals. It takes a 22-number state vector and outputs an action."),
                ("Right Box (The Environment):", "SUMO simulating 98 signals, ~15,000 cars, and microscopic road physics across Ingolstadt."),
                ("Top Arrow (Action):", "Sent every 5 seconds: 0 = Keep Green, 1 = Advance to Next Phase."),
                ("Middle Arrow (State):", "22 normalized features: current phase, elapsed green time, queue lengths, 300m feeder jams, and neighbor traffic."),
                ("Bottom Arrow (Reward):", "A direct cost penalty proportional to stopped queues, driver delays, and city boundary backlogs."),
                ("The Bottom Banner (Safety Layer):", "A deterministic filter ensuring minimum green, maximum green, and yellow clearance are never violated.")
            ]
        },
        "callout": "Explain to the professor: 'Point directly to Figure 1: On the left is our single shared neural network. On the right is SUMO. Every 5 seconds, 98 signals send 22 numbers, the policy chooses Keep or Advance, and the safety layer ensures no accidents happen. This loop runs 720 times per hour.'",
        "notes": "Professor, Figure 1 illustrates our closed-loop architecture. On the left is our single shared neural network; on the right is SUMO simulating 98 signals. Every 5 seconds, the agent receives 22 numbers describing traffic, picks an action (Keep or Advance), and receives a delay-cost reward. The safety layer sits between the policy and SUMO, guaranteeing clearance intervals."
    },
    {
        "num": 6,
        "is_dark": False,
        "badge": "System Architecture Evolution",
        "title": "Architectural Exploration: 3 Approaches Tested",
        "subtitle": "Directly reflecting our handwritten engineering journey from Centralized to IPPO",
        "cards": [
            ("Approach 1: Centralized Agent ❌", "coral", [
                "Concept: One single central controller observes the entire city state and selects joint actions for all 98 intersections.",
                "Observation Space: Combined state vector of thousands of raw lane queues.",
                "Action Space Explosion: With 98 binary signals, joint discrete action space is 2^98 ≈ 3.16 × 10^29 actions per step!",
                "Why It Failed: Mathematically intractable. Neural networks cannot sample or converge across 10^29 joint actions. Severe sample inefficiency."
            ]),
            ("Approach 2: 98 Separate Agents ❌", "amber", [
                "Concept: Every single intersection gets its own dedicated, specialized neural network policy.",
                "Theoretical Appeal: Each policy network can specialize in the unique geometry of its intersection.",
                "Massive Overhead: Training, checkpointing, and optimizing 98 distinct neural networks simultaneously.",
                "Why It Failed: Environment non-stationarity. Each agent changed the traffic environment for its neighbors, destabilizing learning. Zero knowledge transfer between junctions."
            ]),
            ("Approach 3: IPPO + Shared Network ✅", "green", [
                "Concept: Independent PPO (IPPO) with Parameter Sharing + Neighbor Observation Sharing.",
                "One Shared Brain: All 98 intersections share one identical Actor-Critic policy (22 -> 256 -> 256 -> 2).",
                "Local Observations: Each signal inputs its own 22-dim local view (including queues from immediate neighbor junctions).",
                "Why It Won: Massive sample efficiency! Every 5s step produces 98 experience transitions. Knowledge learned on a major avenue immediately generalizes across the city."
            ])
        ],
        "callout": "Explain to the professor: 'As written in our review notes, we tested all three architectures: A centralized agent suffers from 2^98 action space explosion; 98 separate networks suffer from non-stationarity and compute bloat; but IPPO with parameter sharing gives us 98x sample efficiency and spatial generalization.'",
        "notes": "Professor, as documented in our handwritten notes, we explored three architectures. Centralized control failed due to 2^98 action space explosion. 98 separate networks failed due to non-stationarity and high maintenance overhead. IPPO with parameter sharing won: one shared neural network receives 98 transitions per step, allowing knowledge learned on major corridors to instantly benefit neighbor intersections."
    },
    {
        "num": 7,
        "is_dark": False,
        "badge": "Theoretical Foundations",
        "title": "Why PPO? Grounded in Sudharsan Ravichandiran (2018)",
        "subtitle": "Comparing classical RL algorithms against PPO across stability, variance, and sample efficiency",
        "table": {
            "headers": ["Algorithm", "Book Reference", "Why It Fails on 98-Signal City Control", "Why PPO / IPPO Wins"],
            "rows": [
                ("Q-Learning / Tabular TD", "Ch 3 & 5 (pp. 53–120)", "State space is continuous & infinite (22 numbers per signal). Tabular Q cannot generalize.", "PPO uses deep neural network function approximation mapping continuous states to actions."),
                ("Deep Q-Networks (DQN)", "Ch 8 (pp. 165–188)", "Joint action space is 2^98. In multi-agent, off-policy non-stationarity invalidates replay buffer.", "PPO is on-policy; handles multi-agent decentralized transitions naturally without replay corruption."),
                ("REINFORCE (Vanilla PG)", "Ch 11 (pp. 226–230)", "Relies on full episode Monte Carlo returns. Massive variance, slow learning, no baseline.", "PPO uses Actor-Critic with GAE (gamma=0.99, lambda=0.95), drastically reducing return variance."),
                ("A2C / A3C (Actor-Critic)", "Ch 10 (pp. 208–224)", "Unconstrained policy updates. One bad gradient can push policy off a cliff into irrecoverable collapse.", "PPO clips the surrogate objective to [1-eps, 1+eps], guaranteeing safe, monotonic policy updates."),
                ("TRPO (Trust Region)", "Ch 11 (pp. 240–244)", "Requires calculating second-order Fisher Information matrix inversions with conjugate gradients.", "PPO achieves trust-region safety using first-order clipping optimized with standard Adam.")
            ]
        },
        "callout": "Explain to the professor: 'Grounded in Sudharsan Ravichandiran's book: DQN fails due to the 2^98 action space and replay buffer corruption; REINFORCE has high Monte Carlo variance; TRPO requires expensive second-order Hessian inversions. PPO gives us first-order trust-region clipping with GAE variance reduction.'",
        "notes": "Professor, we grounded our algorithm choice directly in Sudharsan Ravichandiran’s textbook. Tabular Q cannot handle continuous 22-dimensional states. DQN suffers from 2^98 joint actions and multi-agent replay buffer corruption. REINFORCE has high variance from full Monte Carlo rollouts. TRPO requires expensive second-order Fisher matrix inversions. PPO achieves trust-region safety with a first-order clipped surrogate loss and GAE variance reduction."
    },
    {
        "num": 8,
        "is_dark": False,
        "badge": "Mathematical Mechanics Figure",
        "title": "Figure 2 Walkthrough: The Clipped Surrogate Objective",
        "subtitle": "Understanding how PPO prevents policy collapse through probability ratio bounding",
        "fig_layout": {
            "img": "fig/fig_ppo_clip.png",
            "title": "Mathematical Intuition of Figure 2",
            "points": [
                ("The Probability Ratio r(θ):", "r_t(θ) = π_θ(a_t | s_t) / π_old(a_t | s_t). Measures how much the updated policy differs from the old policy."),
                ("Left Plot (Advantage > 0, Good Action):", "The dashed line shows unclipped gain. But PPO's blue curve FLATTENS at r = 1 + ε = 1.2. The model gets no extra credit for making an update larger than 20%."),
                ("Right Plot (Advantage < 0, Bad Action):", "If an action was bad, we want to decrease its probability. But the blue curve flattens at r = 1 - ε = 0.8. It prevents destructive over-penalization."),
                ("The Shaded Corridor [0.8, 1.2]:", "This is the Trust Region (ε = 0.2). Updates within this corridor are smooth; updates outside are clipped."),
                ("GAE (γ=0.99, λ=0.95):", "Generalized Advantage Estimation balances immediate reward (low variance) against future traffic delay (low bias).")
            ]
        },
        "callout": "Explain to the professor: 'Point to Figure 2: The x-axis is r (how much the policy changed). On the left, if the action was good, the blue line clips at 1.2 so the agent doesn't get greedy. On the right, if the action was bad, it clips at 0.8 so the policy doesn't collapse. That 20% corridor guarantees training stability.'",
        "notes": "Professor, Figure 2 shows PPO's clipped surrogate objective from Ravichandiran Chapter 11. On the left, when an action was good, the objective flattens at 1.2, preventing greedy over-updates. On the right, when an action was bad, it clips at 0.8, preventing policy collapse. The 20% corridor restricts policy updates to a safe trust region, ensuring monotonic improvement."
    },
    {
        "num": 9,
        "is_dark": False,
        "badge": "Key Breakthrough Figure",
        "title": "Figure 3 Walkthrough: Split Gradient Clipping Fix",
        "subtitle": "How decoupling Actor and Critic gradient norms solved the hidden 7,600x suppression bug",
        "fig_layout": {
            "img": "fig/fig_gradnorm.png",
            "title": "The 'Aha!' Engineering Breakthrough",
            "points": [
                ("The Initial Mystery:", "Early in the project, standard PPO failed against SUMO actuated. Policy entropy remained frozen at ~90%, and lights switched almost randomly."),
                ("The Discovery (Figure 3):", "We probed the raw gradients: Critic gradient norm was ~3,830 (due to large early value errors), while Actor gradient norm was only ~0.19!"),
                ("The Joint Clipping Bug:", "Standard Stable-Baselines3 clips all parameters jointly: Scale = 0.5 / sqrt(3830^2 + 0.19^2) ≈ 0.5 / 3830 ≈ 1.3 × 10^-4."),
                ("The Catastrophe:", "The actor's gradient (0.19) was multiplied by 1.3 × 10^-4, shrinking to 0.000025—below Adam optimizer's epsilon threshold (10^-8)! The actor was effectively turned off!"),
                ("Our Fix (split_clip.py):", "We decoupled gradient clipping: Actor clipped at 0.5; Critic clipped at 0.5 independently."),
                ("The Result:", "Policy learning accelerated by 10x immediately, and CityMind began decisively beating the civil engineering baseline.")
            ]
        },
        "callout": "Explain to the professor: 'Point to Figure 3: Look at the top bar—the critic gradient was 3,830. The actor gradient at the bottom was only 0.19. Standard libraries clip them together, which scaled down the actor by 7,600 times and froze the policy. By clipping them independently, we solved the bug and unlocked 10x faster learning.'",
        "notes": "Professor, Figure 3 represents our key algorithmic breakthrough: Split Gradient Clipping. Early on, policy entropy was frozen at 90%. Probing the gradients revealed that the Critic gradient norm was ~3,830 while the Actor was only ~0.19. Standard libraries clipped them jointly under 0.5, scaling the actor down by 7,600x—below Adam's epsilon. In split_clip.py, we clipped the actor and critic independently at 0.5. The actor immediately woke up, accelerating learning by 10x."
    },
    {
        "num": 10,
        "is_dark": False,
        "badge": "Mathematical & Hyperparameter Rigor",
        "title": "Figure Walkthrough: IPPO Hyperparameters & Architecture",
        "subtitle": "Rigorous parameter selection, rollout scale (588 agents/tick), and mathematical loss formulation",
        "fig_layout": {
            "img": "ippo_hyperparameters_guide.png",
            "title": "Mathematical & Physical Traffic Rationale",
            "points": [
                ("Master Objective L(θ):", "L^CLIP(θ) - 0.5·L^VF(θ) + 0.01·S[π_θ](s_t). Balances policy gain, value estimation, and exploration."),
                ("Rollout Scale (588 Agents):", "6 parallel SUMO envs × 98 signals = 588 agents acting per tick. 180 steps = 900s (15 min) rush hour = 105,840 samples/update."),
                ("Horizon γ = 0.99 & GAE λ = 0.95:", "Effective horizon = 500s (~8.3 min), enabling coordinated multi-intersection green waves instead of myopic 5s gains."),
                ("Trust Region ε = 0.2:", "Bounds policy ratio to [0.8, 1.2], preventing destructive policy shifts and citywide gridlock."),
                ("Split Gradient Clipping:", "Decouples Actor (0.5) and Critic (0.5) clipping, resolving the 7,600x gradient suppression bug and speeding learning by 10x."),
                ("MLP [256, 256] with Tanh:", "Zero-centered smooth gradients in [-1, 1], avoiding dying ReLUs and executing inference in <1ms on CPU.")
            ]
        },
        "callout": "Explain to the professor: 'Every hyperparameter is mathematically grounded: γ=0.99 ensures an 8.3-minute horizon for green waves; ε=0.2 bounds policy shift to 20%; 6 parallel environments collect 105,840 transitions per update; and Split Gradient Clipping isolates actor gradients from critic value error.'",
        "notes": "Professor, this slide details our best-performing IPPO hyperparameters. With 6 parallel environments, 588 agents act simultaneously, collecting 105,840 transitions every 15 minutes of simulated rush hour across 8 minibatches. Gamma=0.99 grants an 8.3-minute horizon for green wave formation, epsilon=0.2 bounds updates to a 20% trust region, and Split Gradient Clipping clips actor and critic independently at 0.5 to prevent gradient starvation."
    },
    {
        "num": 11,
        "is_dark": False,
        "badge": "Environment Specification",
        "title": "22-Dim Observation Space, Action Space & Safety Layer",
        "subtitle": "How 22 numbers describe any intersection geometry and how safety rules prevent crashes",
        "cards": [
            ("1. Observation Vector (22-Dim) 📊", "blue", [
                "Normalized to [-1, 1] for stable NN input.",
                "0–5: Active green phase (one-hot vector across up to 6 green phases).",
                "6: In-transition flag (1.0 during yellow or red clearance).",
                "7–8: Elapsed green duration relative to authored Min Green and Max Green.",
                "9–14: Varaiya pressure per green movement (Upstream + Feeders - Downstream).",
                "15: Max queue on worst approach (300m feeders).",
                "16: Pending outside-network vehicles waiting to enter.",
                "17: Number of valid green phases / 6.",
                "18–21: Neighbor queues (2 upstream, 2 downstream junctions)."
            ]),
            ("2. Action Space: Discrete(2) 🚦", "green", [
                "Action 0: KEEP current green phase.",
                "Action 1: ADVANCE to next sequential green phase.",
                "Why Discrete(2)? Avoids combinatorial phase selection bugs.",
                "Spatial Invariance: A 2-phase pedestrian crossing and a 6-phase major boulevard use the exact same action contract.",
                "Safety Decoupling: The neural network never directly commands raw light colors. It expresses intent ('keep' or 'advance'), which is evaluated by the SafetyLayer."
            ]),
            ("3. Deterministic Safety Layer 🛡️", "coral", [
                "Implemented in src/environment/safety.py as a deterministic finite-state machine.",
                "Minimum Green Enforcement: Even if agent chooses 'advance', phase cannot switch until minimum green (10s) expires. Prevents light flickering.",
                "Maximum Green Cap: Forces phase advance if maximum green (60s) is exceeded. Prevents lane starvation.",
                "Mandatory Clearance Phases: Yellow (3-4s) and all-red clearance intervals are strictly executed in hardware before green switches.",
                "Zero Collision Risk: The agent physically cannot cause an accident in the simulator."
            ])
        ],
        "callout": "Explain to the professor: 'Because junctions have different numbers of lanes (4 to 19), feeding raw lanes into a neural network is impossible. We engineered a compact 22-dimensional feature vector that works across all 98 signals, and backed it with a deterministic safety layer that enforces clearance intervals.'",
        "notes": "Professor, we engineered a fixed 22-dimensional observation vector that maps any junction geometry: features 0-5 encode active phase, 7-8 encode relative green duration, 9-14 encode Varaiya pressure, 15 tracks worst feeder queue, 16 tracks vehicles waiting outside, and 18-21 encode neighbor traffic. Actions are Discrete(2): Keep or Advance. The deterministic SafetyLayer enforces minimum green, maximum green, and yellow clearance, preventing collisions."
    },
    {
        "num": 12,
        "is_dark": False,
        "badge": "Civil Engineering Realism",
        "title": "Engineering Breakthroughs: 300m Feeders & Delay Cost",
        "subtitle": "Overcoming stop-line blindness and designing a non-gameable reward function",
        "two_col": (
            ("Breakthrough 2: 300-Meter Feeder Walker 🚗", "blue", [
                "The Stop-Line Blindness Trap: In European city centers, stop-lines can be as short as 15 to 20 meters, holding only 2 or 3 vehicles per lane.",
                "The Blindspot: If an RL agent only queries the stop-line lane, a massive 60-car traffic jam sitting 30 meters behind it is completely invisible!",
                "Our Solution (local_observation.py): We implemented a recursive upstream road crawler.",
                "300-Meter Depth: The algorithm walks backwards along incoming edges up to 300 meters, aggregating halted vehicles on all connected feeder lanes.",
                "Result: The agent can see platoons and traffic jams forming hundreds of meters away, giving it the foresight to extend green phases before spillback blocks cross-traffic."
            ]),
            ("Breakthrough 3: Delay-Level Cost Reward 🎯", "green", [
                "Why Change in Waiting Time (ΔW) Failed: Academic papers often reward 'decrease in waiting time'. Agents quickly game this reward by rapidly flipping lights—briefly reducing wait time while causing massive physical gridlock.",
                "Our Delay-Cost Function (local_reward.py): Formulated as a direct economic cost penalty:",
                "r_i(t) = -0.1 × min(Q_i + 2D_i + S_i + 0.5P_i + 0.5(P_total / 98), 200)",
                "Q_i = Halted vehicles on stop-line and 300m feeder approaches.",
                "D_i = Stopped vehicles experiencing severe delay (>30s).",
                "S_i = Phase switching penalty (discourages unnecessary phase changes).",
                "P_i & P_total = Vehicles queued outside the network waiting to enter.",
                "Result: Impossible to game. The agent maximizes reward only by flushing vehicles cleanly through the network."
            ])
        ),
        "callout": "Explain to the professor: 'Two practical civil engineering fixes: First, stop-lines only hold 2 cars, so we recursively crawl upstream roads 300 meters to see real queues. Second, we abandoned delta-wait-time (which causes light-flipping oscillation) and used a direct delay-cost penalty that includes outside entry queues.'",
        "notes": "Professor, here are two vital engineering breakthroughs: First, stop-lines in Ingolstadt only hold 2 cars. If the agent only checks the stop-line, a 60-car jam 30 meters back is invisible. We recursively crawl upstream roads up to 300 meters. Second, delta-waiting-time caused rapid light flipping. We replaced it with a direct delay-cost penalty on queues, delays, phase switches, and outside entry backlogs."
    },
    {
        "num": 13,
        "is_dark": False,
        "badge": "Scientific Evaluation Protocol",
        "title": "The Survivor Bias Trap & Drain Metrics Protocol",
        "subtitle": "Why standard academic metrics lie, and how our evaluation scores all scheduled demand",
        "two_col": (
            ("The Survivor Bias Trap in Standard RL ❌", "coral", [
                "What standard literature reports: 'Mean Waiting Time of Completed Trips'.",
                "The Fatal Flaw: In a failing, gridlocked simulation, vehicles attempting long journeys across the city get trapped in queues and NEVER reach their destinations.",
                "Who actually completes? Only lucky vehicles taking short, direct trips on uncongested perimeter roads.",
                "The Paradoxical Lie: The simulator computes average wait time ONLY across the completed cars. Because only the fast, uncongested cars finished, the metric reports a falsely low waiting time!",
                "Severe Disguise: An agent causing catastrophic citywide gridlock can appear to have 'low wait times' simply by trapping all difficult trips forever."
            ]),
            ("Our Solution: The Drain Metrics Protocol ✅", "green", [
                "Implemented in src/evaluation/drain_metrics.py to eliminate survivor bias.",
                "Scores ALL ~15,000 Scheduled Trips: Evaluates every vehicle that was supposed to travel during the rush hour.",
                "Free-Flow Ideal Baseline: Calculates the theoretical zero-delay completion time for each car: ideal_k = depart_k + free_flow_time_k.",
                "Lag per Scheduled Car: Lag_k = min(actual_arrival_k, T_end) - min(ideal_k, T_end).",
                "Trapped Car Penalty: If a vehicle gets stuck or never enters the city, its completion time is penalized at the simulation boundary (T_end = 08:00).",
                "Total Network Accountability: A controller cannot score well without draining the entire city cleanly."
            ])
        ),
        "callout": "Explain to the professor: 'Evaluating traffic by only completed cars is like evaluating a hospital by only counting patients who walked out alive! If an RL policy causes gridlock, cars get trapped and never finish. Our Drain Metrics score all 15,000 scheduled cars against ideal free-flow arrival, eliminating survivor bias.'",
        "notes": "Professor, standard literature reports 'Mean Wait Time of Completed Trips'. But if an agent causes gridlock, cars get trapped and never finish. Only lucky cars on short trips finish, making wait times look artificially low! Our Drain Metrics protocol evaluates all 15,000 scheduled trips against ideal zero-delay completion. Trapped cars are penalized at simulation end time."
    },
    {
        "num": 14,
        "is_dark": False,
        "badge": "Statistical Evidence Figure",
        "title": "Figure 4 Walkthrough: 10-Seed Head-to-Head Victory",
        "subtitle": "Every single gray line slopes downward: CityMind wins 10 out of 10 independent random seeds",
        "fig_layout": {
            "img": "fig/fig_seed_lag.png",
            "title": "Statistical Rigor of Figure 4",
            "points": [
                ("The Evaluation Protocol:", "40 full 1-hour simulation runs across 10 random seeds (7, 11, 19, 23, 31, 42, 57, 91, 123, 777)."),
                ("What the Gray Lines Represent:", "Each connecting gray line is ONE exact traffic realization tested head-to-head across SUMO actuated, Model A, and Model B."),
                ("The Unbroken Downward Slope:", "Notice that every single line slopes downward from baseline to our models. There is not a single seed where the baseline won!"),
                ("Baseline (SUMO Actuated):", "Mean lag of 276.8 seconds per scheduled vehicle."),
                ("CityMind Model A:", "Mean lag of 235.4 seconds (-41.4 seconds, -15.0%). Won 10/10 seeds (p = 0.002, exact sign test)."),
                ("CityMind Model B:", "Mean lag of 239.9 seconds (-36.9 seconds, -13.3%). Won 10/10 seeds (p = 0.002, exact sign test)."),
                ("Confidence Interval:", "95% t-interval for Model A difference is [-43.9 s, -38.9 s], safely excluding zero.")
            ]
        },
        "callout": "Explain to the professor: 'Point directly to Figure 4: The y-axis is lag per car. The left column is the civil engineering baseline; the middle is Model A. Each gray line is the exact same traffic seed. Every single line slopes downward! That proves our victory is not a random lucky seed—it is a 10-out-of-10 statistically verified result (p=0.002).'",
        "notes": "Professor, Figure 4 provides our definitive statistical proof across 10 independent seeds. Each gray line connects the exact same random traffic seed across controllers. Every single gray line slopes downward: on SUMO actuated, mean lag is 276.8s; on Model A, it drops to 235.4s—saving 41.4 seconds per vehicle. Under the exact sign test, winning 10/10 seeds yields p = 0.002."
    },
    {
        "num": 15,
        "is_dark": False,
        "badge": "Benchmark Comparison Figure",
        "title": "Figure 5 Walkthrough: Multi-Metric Citywide Gains",
        "subtitle": "Longer bars indicate greater percentage improvement over SUMO Actuated across 14 metrics",
        "fig_layout": {
            "img": "fig/fig_improvement.png",
            "title": "Key Metric Breakdown in Figure 5",
            "points": [
                ("How to Read the Chart:", "Horizontal bars show percentage improvement over SUMO Actuated (longer = better). Blue = Model A; Orange = Model B. Numbers show seeds won (out of 10)."),
                ("Mean Waiting Time (-41.1%):", "The longest bar at the top! Waiting time at intersections dropped from 6.39s to 3.77s (10/10 seeds won)."),
                ("Mean Queue Length (-27.9%):", "Average junction queue dropped from 0.44 to 0.32 halted cars (10/10 seeds won)."),
                ("City Backlog (-8.2%, -256 Cars):", "At 08:00, 256 fewer cars were left trapped inside the city network (10/10 seeds won)."),
                ("Trips Completed (+258 Cars):", "Finished journeys increased by +258 vehicles (from 11,585 to 11,842, 10/10 seeds won)."),
                ("Zero Degradation:", "No metric is worse. Fixed-time, in contrast, failed on 11 out of 14 metrics.")
            ]
        },
        "callout": "Explain to the professor: 'Point to Figure 5: This bar chart displays all 14 traffic metrics. Notice that every bar extends to the right—meaning positive improvement over SUMO Actuated. Notice the small numbers on the right of each bar: 10/10, 10/10, 10/10. We didn't sacrifice one metric to boost another; the entire network improved.'",
        "notes": "Professor, Figure 5 demonstrates holistic network improvement. Every bar extends to the right, showing percentage gain over SUMO Actuated. Waiting time dropped 41.1%, queue lengths dropped 27.9%, city backlog dropped by 256 trapped vehicles, and trips completed increased by +258 vehicles. All core metrics won 10 out of 10 seeds."
    },
    {
        "num": 16,
        "is_dark": False,
        "badge": "Numerical Performance Matrix",
        "title": "Comprehensive Head-to-Head Benchmark Table",
        "subtitle": "Statistical comparison across SUMO Actuated, Fixed-Time, Model A, and Model B",
        "table": {
            "headers": ["Metric Evaluated", "SUMO Actuated (Base)", "Fixed-Time", "CityMind Model A", "Difference (vs Base)", "Seeds Won", "Statistical Verdict"],
            "rows": [
                ("Lag per scheduled car (s)", "276.8 s", "298.7 s (+7.9%)", "235.4 s", "-41.4 s (-15.0%)", "10 / 10", "BEATEN (p=0.002)"),
                ("Time loss per finished car (s)", "336.6 s", "363.5 s (+8.0%)", "286.4 s", "-50.2 s (-14.9%)", "10 / 10", "BEATEN (p=0.002)"),
                ("Mean waiting time (s)", "6.39 s", "7.14 s (+11.7%)", "3.77 s", "-2.63 s (-41.1%)", "10 / 10", "BEATEN (p=0.002)"),
                ("Backlog at 08:00 (cars)", "3,108 cars", "3,297 cars", "2,852 cars", "-256 cars (-8.2%)", "10 / 10", "BEATEN (p=0.002)"),
                ("Mean queue length (cars)", "0.440", "0.492 (+11.8%)", "0.318", "-0.123 (-27.9%)", "10 / 10", "BEATEN (p=0.002)"),
                ("Mean vehicle speed (m/s)", "8.70 m/s", "8.40 m/s (-3.4%)", "9.37 m/s", "+0.68 m/s (+7.8%)", "10 / 10", "BEATEN (p=0.002)"),
                ("Trips completed (vehicles)", "11,585", "11,432 (-152)", "11,842", "+258 cars (+2.2%)", "10 / 10", "BEATEN (p=0.002)"),
                ("Depart delay (s)", "15.7 s", "15.3 s", "14.1 s", "-1.59 s (-10.2%)", "10 / 10", "BEATEN (p=0.002)")
            ]
        },
        "callout": "Explain to the professor: 'This table summarizes the 40 experimental runs. Look at Fixed-Time in column 3: it is worse than the baseline on almost every metric (+7.9% lag, +11.7% wait time). In contrast, CityMind Model A beats SUMO Actuated across the board with p=0.002 statistical significance.'",
        "notes": "Professor, this table aggregates our findings. Fixed-time timers performed significantly worse than SUMO Actuated (+7.9% lag, +11.7% wait time). In contrast, CityMind Model A reduced lag by 41.4s, reduced time loss by 50.2s, reduced waiting time by 41.1%, and increased completed trips by +258 vehicles, winning 10/10 seeds with p=0.002."
    },
    {
        "num": 17,
        "is_dark": False,
        "badge": "Engineering & Reproducibility",
        "title": "Modular Codebase Architecture & Live Execution",
        "subtitle": "Clean modular software design with verified reproducibility commands",
        "two_col": (
            ("Modular Codebase Architecture 📂", "blue", [
                "src/agents/ippo_agent.py: PettingZoo wrapper, GAE calculator, and training harness.",
                "src/agents/split_clip.py: Decoupled Actor/Critic gradient norm clipping optimizer.",
                "src/environment/sumo_connection.py: Lazy TraCI / libsumo socket proxy.",
                "src/environment/decentralized_sumo_env.py: 98-agent PettingZoo parallel traffic environment.",
                "src/environment/local_observation.py: 22-dim feature extractor and 300m feeder crawler.",
                "src/environment/local_reward.py: Delay-cost reward function with boundary backlog tracking.",
                "src/environment/safety.py: Deterministic clearance and min/max green state machine.",
                "src/evaluation/drain_metrics.py: Survivor-bias-free scheduled demand evaluation."
            ]),
            ("Live Verification & CLI Commands 💻", "green", [
                "100% Runnable & Self-Contained: Managed with uv and virtual environment.",
                "Evaluate 10 Seeds Head-to-Head: uv run python scripts/evaluate.py --help",
                "Launch Live SUMO GUI Demo: uv run python scripts/run_visual_demo.py --render",
                "Train IPPO from Scratch: uv run python scripts/run_ippo_training.py",
                "Re-generate Figures & Report: uv run python docs/professor_report/make_figures.py",
                "Automated CI & Validation: Passes pytest, ruff formatting, and model contracts."
            ])
        ),
        "callout": "Explain to the professor: 'Everything in this project is modular, fully typed, unit-tested, and reproducible with simple uv run CLI commands. If you would like to see a live demonstration right now, we can run the visual GUI demo or inspection script immediately.'",
        "notes": "Professor, our software is fully modular and reproducible. src/agents contains IPPO and split-clipping; src/environment isolates TraCI and the safety layer; src/evaluation contains drain metrics. The entire project is runnable with uv. We can run the live SUMO visual demo right now if you wish."
    },
    {
        "num": 18,
        "is_dark": True,
        "badge": "Summary & Contributions",
        "title": "Conclusion: Why CityMind AI Succeeds",
        "subtitle": "Summary of key contributions and theoretical breakthroughs in city-wide traffic control",
        "pillars": [
            ("1. Real-World Scale 🇩🇪", "Evaluated on Ingolstadt InTAS: 98 heterogeneous intersections, >15k vehicles, Krauss car-following physics, beating calibrated induction loops."),
            ("2. Solid RL Theory 📚", "Formulated as Dec-POMDP with parameter-shared IPPO, GAE variance reduction, and clipped surrogate loss grounded in Ravichandiran (2018)."),
            ("3. Engineering Breakthroughs 🛠️", "Invented Split Gradient Clipping (fixing 7,600x actor suppression), 300m feeder walker, delay-cost reward, and deterministic safety layer."),
            ("4. Decisive 10-Seed Victory 🏆", "Slashing lag by -41.4s (-15%) and wait times by -41.1%, clearing 256 trapped vehicles, winning 10/10 seeds with p=0.002.")
        ],
        "notes": "In conclusion, Professor: CityMind AI demonstrates that decentralized Deep RL can outperform calibrated civil engineering across 98 real intersections in Ingolstadt. By resolving the 7,600x actor gradient suppression bug, seeing 300m feeder queues, and enforcing deterministic safety, we achieved a statistically bulletproof 10/10 seed victory with a 41.1% wait-time reduction. Thank you, and I look forward to your questions."
    },
    {
        "num": 19,
        "is_dark": False,
        "badge": "Appendix & Oral Defense Reference",
        "title": "Professor Q&A Cheat Sheet: Rapid-Fire Answers",
        "subtitle": "Targeted technical answers for anticipated professor questions",
        "qa_grid": [
            ("Q1: Why not Centralized DQN?", "Centralized action space is 2^98 ≈ 3.16 × 10^29. Finding argmax Q is NP-hard. In multi-agent, off-policy replay buffers become invalid due to environment non-stationarity."),
            ("Q2: What if TraCI socket drops or lags?", "Wrapped in a resilient proxy (sumo_connection.py). For training, we use libsumo—a direct C++ SWIG binding that runs in-process, eliminating sockets and network latency entirely."),
            ("Q3: Why Tanh activations in MLP?", "Tanh produces zero-centered, smooth gradients bounded in [-1, 1]. In continuous-control PPO, ReLU often leads to dead neurons and aggressive policy shifts; Tanh ensures stable updates."),
            ("Q4: How does Safety Layer prevent crashes?", "The RL agent never sets light colors directly. SafetyLayer implements a deterministic state machine: when advancing, it strictly inserts yellow (3-4s) and all-red clearance intervals."),
            ("Q5: Why GAE gamma=0.99 and lambda=0.95?", "Grounded in Ravichandiran Ch. 10: gamma=0.99 ensures the policy values clearing future downstream queues. lambda=0.95 balances Temporal Difference bias against Monte Carlo variance."),
            ("Q6: Why are teleports noisy in results?", "Teleports occur only when extreme gridlock persists for 300+ seconds. In a 1-hour run, teleports are rare (7.5 vs 13.3). They dropped 43%, but due to high variance, sign test is p=0.508 (directional).")
        ],
        "callout": "Tip for the student: 'If the professor asks any technical question, keep calm, refer to this slide, and connect your answer to: 1. Ravichandiran's book principles, 2. The physical realities of traffic in Ingolstadt, or 3. Our exact code implementation in src/.'",
        "notes": "Keep this slide ready as backup during the Q&A session. If the professor asks why Tanh was used instead of ReLU, point to smooth bounded gradient propagation in control. If asked how collisions are prevented, point to the deterministic SafetyLayer state machine."
    }
]


def render_html():
    css = """
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --navy-dark: #0b192c;
        --slate-dark: #0f172a;
        --bg-light: #f8fafc;
        --card-bg: #ffffff;
        --border-color: #e2e8f0;
        --blue-accent: #2a78d6;
        --coral-accent: #eb6834;
        --green-accent: #10b981;
        --text-main: #0f172a;
        --text-muted: #64748b;
        --text-light: #f1f5f9;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background: #0f172a;
        color: var(--text-main);
        padding: 24px;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 32px;
    }

    .slide-container {
        width: 1280px;
        height: 720px;
        background: var(--bg-light);
        border-radius: 12px;
        overflow: hidden;
        position: relative;
        padding: 44px 56px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        page-break-after: always;
        break-after: page;
    }

    .slide-container.dark {
        background: var(--navy-dark);
        color: var(--text-light);
    }

    /* Header */
    .header { display: flex; flex-direction: column; gap: 6px; }
    .header-top { display: flex; justify-content: space-between; align-items: center; }
    .badge {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        background: #eff6ff;
        color: #1d4ed8;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
    }
    .slide-container.dark .badge {
        background: #1e293b;
        color: #93c5fd;
        border: 1px solid var(--blue-accent);
    }
    .slide-num {
        font-size: 13px;
        font-weight: 600;
        color: var(--text-muted);
    }
    .slide-title {
        font-size: 26px;
        font-weight: 800;
        color: var(--text-main);
        line-height: 1.2;
    }
    .slide-container.dark .slide-title { color: #ffffff; }
    .slide-subtitle {
        font-size: 14px;
        color: var(--text-muted);
        font-weight: 500;
    }
    .slide-container.dark .slide-subtitle { color: #94a3b8; }

    /* Content Area */
    .content-area {
        flex: 1;
        margin: 20px 0;
        display: flex;
        gap: 20px;
        overflow: hidden;
    }

    .card {
        background: var(--card-bg);
        border: 1.5px solid var(--border-color);
        border-radius: 10px;
        padding: 20px;
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    .slide-container.dark .card {
        background: #182235;
        border-color: #334155;
    }
    .card-title {
        font-size: 15px;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-title.blue { color: var(--blue-accent); }
    .card-title.coral { color: var(--coral-accent); }
    .card-title.green { color: var(--green-accent); }
    .card-title.amber { color: #d97706; }

    .bullet-list {
        list-style: none;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .bullet-list li {
        font-size: 12.5px;
        line-height: 1.45;
        color: #334155;
        position: relative;
        padding-left: 14px;
    }
    .slide-container.dark .bullet-list li { color: #cbd5e1; }
    .bullet-list li::before {
        content: "•";
        position: absolute;
        left: 0;
        color: var(--blue-accent);
        font-weight: bold;
    }

    /* Table */
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
        border-radius: 8px;
        overflow: hidden;
    }
    .data-table th {
        background: var(--slate-dark);
        color: #ffffff;
        padding: 10px 12px;
        text-align: left;
        font-weight: 700;
    }
    .data-table td {
        padding: 9px 12px;
        border-bottom: 1px solid var(--border-color);
        background: #ffffff;
    }
    .data-table tr:nth-child(even) td { background: #f8fafc; }
    .data-table tr:hover td { background: #eff6ff; }
    .highlight-green { color: var(--green-accent); font-weight: 700; }

    /* Figure Layout */
    .fig-container {
        flex: 1.4;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #ffffff;
        border: 1.5px solid var(--border-color);
        border-radius: 10px;
        padding: 12px;
    }
    .fig-container img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
        border-radius: 6px;
    }
    .fig-explanation {
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    /* Callout */
    .callout {
        background: #f0fdf4;
        border: 1.5px solid #bbf7d0;
        border-radius: 8px;
        padding: 12px 18px;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .callout-title {
        font-size: 11px;
        font-weight: 800;
        color: #166534;
        letter-spacing: 0.5px;
    }
    .callout-body {
        font-size: 12.5px;
        color: var(--text-main);
        line-height: 1.4;
    }

    /* Dark Slide 1 Specifics */
    .hero-stats {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 28px 0;
    }
    .hero-stat-card {
        background: #182235;
        border: 1.5px solid #334155;
        border-radius: 10px;
        padding: 18px;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .hero-stat-val {
        font-size: 28px;
        font-weight: 800;
        color: var(--green-accent);
    }
    .hero-stat-val.blue { color: var(--blue-accent); }
    .hero-stat-lbl {
        font-size: 13px;
        font-weight: 700;
        color: #f1f5f9;
    }
    .hero-stat-sub {
        font-size: 11px;
        color: #94a3b8;
    }

    /* Speaker Notes Drawer */
    .notes-drawer {
        background: #1e293b;
        color: #e2e8f0;
        font-size: 12px;
        line-height: 1.5;
        padding: 12px 16px;
        border-radius: 8px;
        margin-top: 8px;
        width: 1280px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .notes-drawer strong { color: #38bdf8; }

    @media print {
        body { background: transparent; padding: 0; gap: 0; }
        .slide-container {
            width: 16in;
            height: 9in;
            border-radius: 0;
            box-shadow: none;
            padding: 0.5in 0.6in;
        }
        .notes-drawer { display: none !important; }
        @page { size: 16in 9in landscape; margin: 0; }
    }
    """

    slides_html = []
    for s in SLIDES_DATA:
        dark_class = " dark" if s.get("is_dark") else ""
        content_inner = ""

        # Slide 1: Hero
        if s["num"] == 1:
            stats_html = "".join(f"""
            <div class="hero-stat-card">
                <div class="hero-stat-val {'blue' if '%' not in val and 'won' not in sub else ''}">{val}</div>
                <div class="hero-stat-lbl">{lbl}</div>
                <div class="hero-stat-sub">{sub}</div>
            </div>
            """ for val, lbl, sub in s["stats"])

            content_inner = f"""
            <div style="flex: 1; display: flex; flex-direction: column; justify-content: center; gap: 16px;">
                <h1 style="font-size: 52px; font-weight: 800; color: #ffffff; letter-spacing: -1px;">{s['title']}</h1>
                <p style="font-size: 24px; font-weight: 600; color: #cbd5e1;">{s['subtitle']}</p>
                <p style="font-size: 16px; color: #38bdf8; font-weight: 500;">{s['description']}</p>
                <div class="hero-stats">{stats_html}</div>
                <p style="font-size: 12px; color: #94a3b8; border-top: 1px solid #334155; padding-top: 12px;">{s['meta']}</p>
            </div>
            """
        # Pillars Layout
        elif "pillars" in s:
            pillars_html = "".join(f"""
            <div class="card" style="background: #182235; border-color: #334155;">
                <div class="card-title {'green' if 'Victory' in title else 'blue'}" style="font-size: 16px;">{title}</div>
                <p style="font-size: 13px; color: #cbd5e1; line-height: 1.5;">{desc}</p>
            </div>
            """ for title, desc in s["pillars"])

            content_inner = f"""
            <div style="flex: 1; display: flex; flex-direction: column; justify-content: space-between;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 16px;">
                    {pillars_html}
                </div>
                <div style="background: #1e293b; border: 1.5px solid var(--blue-accent); border-radius: 8px; padding: 14px; text-align: center; color: #ffffff; font-weight: 700; font-size: 14px; margin-top: 20px;">
                    🎯 THANK YOU, PROFESSOR. READY FOR YOUR QUESTIONS & DISCUSSION.
                </div>
            </div>
            """
        # 3 Cards Layout
        elif "cards" in s:
            cards_html = "".join(f"""
            <div class="card">
                <div class="card-title {color}">{title}</div>
                <ul class="bullet-list">
                    {''.join(f'<li>{b}</li>' for b in bullets)}
                </ul>
            </div>
            """ for title, color, bullets in s["cards"])
            content_inner = f"""
            <div class="content-area">{cards_html}</div>
            <div class="callout">
                <div class="callout-title">💡 HOW TO EXPLAIN TO THE PROFESSOR IN 20 SECONDS</div>
                <div class="callout-body">{s['callout']}</div>
            </div>
            """
        # Two Column Layout
        elif "two_col" in s:
            col1, col2 = s["two_col"]
            content_inner = f"""
            <div class="content-area">
                <div class="card">
                    <div class="card-title {col1[1]}">{col1[0]}</div>
                    <ul class="bullet-list">
                        {''.join(f'<li>{b}</li>' for b in col1[2])}
                    </ul>
                </div>
                <div class="card">
                    <div class="card-title {col2[1]}">{col2[0]}</div>
                    <ul class="bullet-list">
                        {''.join(f'<li>{b}</li>' for b in col2[2])}
                    </ul>
                </div>
            </div>
            <div class="callout">
                <div class="callout-title">💡 HOW TO EXPLAIN TO THE PROFESSOR IN 20 SECONDS</div>
                <div class="callout-body">{s['callout']}</div>
            </div>
            """
        # Figure Layout
        elif "fig_layout" in s:
            fig = s["fig_layout"]
            points_html = "".join(f"""
            <div style="font-size: 12.5px; line-height: 1.45; color: #334155;">
                <strong style="color: var(--blue-accent);">{title}</strong> {desc}
            </div>
            """ for title, desc in fig["points"])

            content_inner = f"""
            <div class="content-area">
                <div class="fig-container">
                    <img src="{fig['img']}" alt="Figure">
                </div>
                <div class="card fig-explanation">
                    <div class="card-title blue">{fig['title']}</div>
                    <div style="display: flex; flex-direction: column; gap: 10px; overflow-y: auto;">
                        {points_html}
                    </div>
                </div>
            </div>
            <div class="callout">
                <div class="callout-title">💡 HOW TO EXPLAIN TO THE PROFESSOR IN 20 SECONDS</div>
                <div class="callout-body">{s['callout']}</div>
            </div>
            """
        # Table Layout
        elif "table" in s:
            t = s["table"]
            th_html = "".join(f"<th>{h}</th>" for h in t["headers"])
            tr_rows = []
            for row in t["rows"]:
                cells = []
                for val in row:
                    cls_name = "highlight-green" if any(k in str(val) for k in ["BEATEN", "-41.1%", "-15.0%"]) else ""
                    cells.append(f'<td class="{cls_name}">{val}</td>')
                tr_rows.append(f"<tr>{''.join(cells)}</tr>")
            tr_html = "".join(tr_rows)

            content_inner = f"""
            <div class="content-area" style="flex-direction: column;">
                <table class="data-table">
                    <thead><tr>{th_html}</tr></thead>
                    <tbody>{tr_html}</tbody>
                </table>
            </div>
            <div class="callout">
                <div class="callout-title">💡 HOW TO EXPLAIN TO THE PROFESSOR IN 20 SECONDS</div>
                <div class="callout-body">{s['callout']}</div>
            </div>
            """
        # Q&A Grid (Slide 18)
        elif "qa_grid" in s:
            grid_html = "".join(f"""
            <div class="card" style="padding: 14px;">
                <div style="font-size: 13px; font-weight: 800; color: var(--blue-accent); margin-bottom: 4px;">{q}</div>
                <p style="font-size: 11.5px; color: var(--text-main); line-height: 1.4;">{a}</p>
            </div>
            """ for q, a in s["qa_grid"])

            content_inner = f"""
            <div class="content-area" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;">
                {grid_html}
            </div>
            <div class="callout">
                <div class="callout-title">💡 HOW TO EXPLAIN TO THE PROFESSOR IN 20 SECONDS</div>
                <div class="callout-body">{s['callout']}</div>
            </div>
            """

        slide_markup = f"""
        <div class="slide-container{dark_class}" id="slide-{s['num']}">
            <div class="header">
                <div class="header-top">
                    <span class="badge">{s['badge']}</span>
                    <span class="slide-num">SLIDE {s['num']:02d} / 18</span>
                </div>
                <div class="slide-title">{s['title']}</div>
                <div class="slide-subtitle">{s['subtitle']}</div>
            </div>
            {content_inner}
        </div>
        <div class="notes-drawer">
            <strong>🎤 SPEAKER NOTES (WHAT TO SAY TO PROFESSOR):</strong><br>
            {s['notes']}
        </div>
        """
        slides_html.append(slide_markup)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CityMind AI — Professor Defense Master Presentation Deck</title>
    <style>{css}</style>
</head>
<body>
    {''.join(slides_html)}
</body>
</html>
"""
    HTML_PATH.write_text(full_html, encoding="utf-8")
    print(f"Interactive HTML presentation saved to: {HTML_PATH}")


def compile_pdf():
    print(f"Compiling PDF via Headless Chrome: {CHROME_BIN}")
    cmd = [
        CHROME_BIN,
        "--headless",
        "--disable-gpu",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={PDF_PATH}",
        str(HTML_PATH)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"Presentation PDF successfully compiled to: {PDF_PATH}")
    else:
        print(f"PDF compilation error: {res.stderr}")


if __name__ == "__main__":
    render_html()
    compile_pdf()
