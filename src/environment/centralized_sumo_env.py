"""
centralized_sumo_env.py
=======================
CentralizedSumoEnv — the core Gymnasium environment for CityMind AI.

Design constraints (from PROJECT_RESEARCH_AND_VISION.md):
  - ONE policy for the entire controlled network
  - Global observation (all TLS + global network state)
  - Joint MultiBinary action (one bit per TLS)
  - Network-level reward (delta total waiting time)
  - Safety layer enforces all signal timing constraints
  - SUMO/libsumo is the simulation engine (not reimplemented)
"""
from __future__ import annotations

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from src.environment.sumo_connection import traci, USING_LIBSUMO

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402

from src.scenario.tls_metadata import TLSRuntimeState, TLSSpec, load_tls_specs
from src.environment.observation import ObservationBuilder, ObservationConfig
from src.environment.action import ActionDecoder
from src.environment.reward import RewardCalculator, RewardConfig
from src.environment.safety import SafetyLayer, SafetyConfig
from src.environment.metrics import MetricsCollector


class CentralizedSumoEnv(gym.Env):
    """
    Centralized single-agent Gymnasium environment wrapping Eclipse SUMO.

    Observation: flat numpy array (global network state — all TLS + global)
    Action:      MultiBinary(num_tls) — 0=keep, 1=advance to next green phase
    Reward:      -(delta total network waiting time)

    Parameters
    ----------
    sumo_cfg : str
        Path to the .sumocfg file for the scenario.
    tls_metadata_json : str
        Path to artifacts/scenario/tls_metadata.json (output of inspect_intas.py).
    sumo_binary : str
        "sumo" for headless training, "sumo-gui" for visual debugging.
    decision_interval : int
        Number of SUMO simulation seconds between RL decisions.
    step_length : float
        SUMO simulation step length in seconds.
    obs_config : ObservationConfig, optional
    reward_config : RewardConfig, optional
    safety_config : SafetyConfig, optional
    controller : str
        "rl" | "fixed_time" | "max_pressure" (affects metrics labeling).
    seed : int
        Random seed for this episode.
    demand_id : str
        Demand condition identifier (e.g., "1.0x") for metric logging.
    record_trajectories : bool
        Whether to record per-second trajectory data for replay.
    trajectory_output_dir : str | None
        Where to save trajectory Parquet files.
    sumo_extra_args : list[str]
        Extra arguments passed directly to SUMO.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        sumo_cfg: str,
        tls_metadata_json: str,
        sumo_binary: str = "sumo",
        decision_interval: int = 5,
        step_length: float = 1.0,
        obs_config: ObservationConfig | None = None,
        reward_config: RewardConfig | None = None,
        safety_config: SafetyConfig | None = None,
        controller: str = "rl",
        seed: int = 42,
        demand_id: str = "1.0x",
        record_trajectories: bool = False,
        trajectory_output_dir: str | None = None,
        episode_duration: float = 3600.0,
        sumo_extra_args: list[str] | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.sumo_cfg = str(Path(sumo_cfg).resolve())
        self.tls_metadata_json = str(Path(tls_metadata_json).resolve())
        self.sumo_binary = sumo_binary
        self.decision_interval = decision_interval
        self.step_length = step_length
        self.controller_name = controller
        self._seed = seed
        self.demand_id = demand_id
        self.record_trajectories = record_trajectories
        self.trajectory_output_dir = trajectory_output_dir
        self.episode_duration = float(episode_duration)
        self.sumo_extra_args = sumo_extra_args or []
        self.render_mode = render_mode

        # ── Load TLS specifications ──────────────────────────────────────────
        self.tls_specs: list[TLSSpec] = load_tls_specs(Path(self.tls_metadata_json))
        self.num_tls = len(self.tls_specs)
        print(f"[CentralizedSumoEnv] Loaded {self.num_tls} TLS specs.")
        print(f"[CentralizedSumoEnv] Using {'libsumo' if USING_LIBSUMO else 'TraCI (socket)'}")

        # ── Sub-modules ──────────────────────────────────────────────────────
        self.obs_builder = ObservationBuilder(self.tls_specs, obs_config)
        self.action_decoder = ActionDecoder(self.tls_specs)
        self.reward_calc = RewardCalculator(reward_config)
        self.safety = SafetyLayer(self.tls_specs, safety_config)
        self.metrics = MetricsCollector()

        # ── Gymnasium spaces ─────────────────────────────────────────────────
        obs_size = self.obs_builder.compute_obs_size()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.MultiBinary(self.num_tls)

        # ── Episode state ────────────────────────────────────────────────────
        self._sim_running = False
        self._step_count = 0
        self._switch_count_episode = 0
        self._runtime_states: dict[str, TLSRuntimeState] = {}
        self._trajectory: list[dict] = []

        # Register runtime states dict with safety layer
        self.safety.register_states(self._runtime_states)

    # ──────────────────────────────────────────────────────────────────────────
    # Gymnasium interface: reset
    # ──────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Start a new episode."""
        if seed is not None:
            self._seed = seed
        super().reset(seed=self._seed)

        # Close previous simulation if running
        if self._sim_running:
            self._close_sumo()

        # Initialize runtime states (one per TLS)
        self._runtime_states.clear()
        for spec in self.tls_specs:
            self._runtime_states[spec.tls_id] = TLSRuntimeState(tls_id=spec.tls_id)

        # Start SUMO
        self._start_sumo()

        # Initialize TLS phases from SUMO's current state
        self._sync_tls_phases_from_sumo()

        # Reset sub-modules
        self.reward_calc.reset()
        self.metrics.reset(
            controller=self.controller_name,
            seed=self._seed,
            demand_id=self.demand_id,
        )

        self._step_count = 0
        self._switch_count_episode = 0
        self._trajectory = []

        obs = self.obs_builder.build(self._runtime_states)
        info = {"sim_time": 0.0, "controller": self.controller_name}
        return obs, info

    # ──────────────────────────────────────────────────────────────────────────
    # Gymnasium interface: step
    # ──────────────────────────────────────────────────────────────────────────

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply one RL decision: advance SUMO by decision_interval seconds.

        Returns
        -------
        (observation, reward, terminated, truncated, info)
        """
        if not self._sim_running:
            raise RuntimeError("step() called before reset()")

        # 1. Decode action
        action_dict = self.action_decoder.decode(action)

        # 2. Advance simulation by decision_interval steps
        step_switches = 0
        for sub_step in range(self.decision_interval):
            sim_time = traci.simulation.getTime()

            # Apply safety layer (updates SUMO signal states)
            prev_phases = {tid: s.current_phase for tid, s in self._runtime_states.items()}
            self.safety.step(sim_time, action_dict, self.step_length)
            # Count switches
            for tid, state in self._runtime_states.items():
                if state.current_phase != prev_phases[tid]:
                    step_switches += 1

            # Advance simulation by one step
            traci.simulationStep()

            # Record trajectory if needed
            if self.record_trajectories:
                self._record_trajectory_frame(sim_time)

        self._step_count += 1
        self._switch_count_episode += step_switches

        # 3. Build new observation (also refreshes lane state in runtime_states)
        obs = self.obs_builder.build(self._runtime_states)
        cache = getattr(self.obs_builder, "last_step_cache", None)

        # 4. Compute reward
        self.reward_calc.set_switch_count(step_switches)
        reward, reward_info = self.reward_calc.compute(
            self._runtime_states,
            cached_waiting_time=cache.get("total_waiting") if cache else None,
            cached_vehicle_count=int(cache.get("total_vehicles", 0)) if cache else None,
        )

        # 5. Record metrics
        sim_time = traci.simulation.getTime()
        self.metrics.record_step(
            step=self._step_count,
            sim_time=sim_time,
            reward=reward,
            switch_count=step_switches,
            cached_metrics=cache,
            states=self._runtime_states,
        )

        # 6. Check termination
        terminated = self._check_terminated(sim_time)
        truncated = sim_time >= self.episode_duration

        # 7. Build info dict
        info = {
            "sim_time": sim_time,
            "step": self._step_count,
            "switch_count": step_switches,
            **reward_info,
        }

        return obs, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────────
    # Gymnasium interface: close
    # ──────────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._sim_running:
            self._close_sumo()

    # ──────────────────────────────────────────────────────────────────────────
    # SUMO lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def _start_sumo(self) -> None:
        """Launch SUMO subprocess and connect via TraCI/libsumo."""
        cmd = [
            self.sumo_binary,
            "--configuration-file", self.sumo_cfg,
            "--step-length", str(self.step_length),
            "--no-warnings",
            "--no-step-log",
            "--seed", str(self._seed),
        ] + self.sumo_extra_args

        if USING_LIBSUMO:
            traci.start(cmd)
        else:
            traci.start(cmd, label=f"citymind_{self._seed}_{int(time.time())}")

        self._sim_running = True

    def _close_sumo(self) -> None:
        """Safely close TraCI/libsumo connection."""
        try:
            traci.close()
        except Exception:
            pass
        self._sim_running = False

    def _sync_tls_phases_from_sumo(self) -> None:
        """Read initial phase from SUMO for each TLS."""
        for spec in self.tls_specs:
            tid = spec.tls_id
            try:
                phase = traci.trafficlight.getPhase(tid)
            except traci.exceptions.TraCIException:
                phase = 0
            state = self._runtime_states[tid]
            state.current_phase = phase
            state.phase_elapsed = 0.0

    def _check_terminated(self, sim_time: float) -> bool:
        """Episode ends when the simulation time reaches the configured end or minExpected==0."""
        if sim_time >= self.episode_duration:
            return True
        try:
            min_expected = traci.simulation.getMinExpectedNumber()
            return min_expected == 0
        except Exception:
            return True

    # ──────────────────────────────────────────────────────────────────────────
    # Trajectory recording
    # ──────────────────────────────────────────────────────────────────────────

    def _record_trajectory_frame(self, sim_time: float) -> None:
        """Record one frame for replay visualization."""
        try:
            vehicles = []
            for vid in traci.vehicle.getIDList():
                x, y = traci.vehicle.getPosition(vid)
                spd = traci.vehicle.getSpeed(vid)
                vehicles.append({"id": vid, "x": x, "y": y, "speed": spd})

            tls_states = []
            for spec in self.tls_specs:
                tid = spec.tls_id
                state = self._runtime_states[tid]
                tls_states.append({
                    "id": tid,
                    "phase": state.current_phase,
                    "state": spec.phases[state.current_phase].state
                    if state.current_phase < len(spec.phases) else "",
                })

            self._trajectory.append({
                "time": sim_time,
                "vehicles": vehicles,
                "tls": tls_states,
            })
        except Exception:
            pass

    def save_trajectory(self, path: str) -> None:
        """
        Save recorded trajectory to a Parquet file.

        Long format: vehicle rows and TLS-phase rows share one table (vehicle
        columns are null on TLS rows and vice versa), so the replay API can
        serve the RL policy's *actual* per-frame signal state to the frontend
        instead of a client-side approximation.
        """
        if not self._trajectory:
            return
        import pandas as pd
        rows = []
        for frame in self._trajectory:
            t = frame["time"]
            for v in frame["vehicles"]:
                rows.append({
                    "time": t, "vehicle_id": v["id"], "x": v["x"], "y": v["y"], "speed": v["speed"],
                    "tls_id": None, "phase": None, "tls_state": None,
                })
            for tls in frame["tls"]:
                rows.append({
                    "time": t, "vehicle_id": None, "x": None, "y": None, "speed": None,
                    "tls_id": tls["id"], "phase": tls["phase"], "tls_state": tls["state"],
                })
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        try:
            df.to_parquet(path, index=False, engine="pyarrow")
        except Exception:
            try:
                df.to_parquet(path, index=False)
            except Exception:
                df.to_csv(str(path).replace(".parquet", ".csv"), index=False)
        print(f"  Trajectory saved: {path} ({len(rows)} rows)")

    # ──────────────────────────────────────────────────────────────────────────
    # Info
    # ──────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"CentralizedSumoEnv("
            f"num_tls={self.num_tls}, "
            f"obs_size={self.observation_space.shape[0]}, "
            f"backend={'libsumo' if USING_LIBSUMO else 'traci'})"
        )
