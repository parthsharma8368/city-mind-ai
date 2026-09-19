"""
decentralized_sumo_env.py
==========================
DecentralizedSumoEnv — PettingZoo ParallelEnv for parameter-shared IPPO.

One agent per TLS (98 for InTAS), each with the *same* fixed-size local
observation (local_observation.py) and the same binary action (0=keep,
1=advance — decoded through the identical SafetyLayer state machine the
centralized environment and both baselines use, so this is mechanically
comparable to everything else in the project). Each agent gets its own
local reward (local_reward.py) instead of one network-wide scalar shared
by all 98 joint-action dimensions.

Training wraps this with SuperSuit (pettingzoo_env_to_vec_env_v1 +
concat_vec_envs_v1) so a single shared-weight SB3 PPO model trains on all
98 agents' experience every decision step — see src/agents/ippo_agent.py.
For evaluation, call step()/reset() directly in a plain loop (no SuperSuit
needed) exactly like scripts/run_evaluation.py's other controllers.
"""
from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from src.environment.sumo_connection import traci, USING_LIBSUMO
from src.environment.local_observation import (
    LOCAL_OBS_SIZE,
    LocalObservationConfig,
    build_local_observation,
)
from src.environment.local_reward import LocalRewardCalculator, LocalRewardConfig
from src.environment.metrics import MetricsCollector
from src.environment.observation import ObservationBuilder, ObservationConfig
from src.environment.safety import SafetyConfig, SafetyLayer
from src.scenario.tls_metadata import TLSRuntimeState, load_tls_specs


class DecentralizedSumoEnv(ParallelEnv):
    """One agent per TLS, parameter-shared local observation/reward/action."""

    metadata = {"name": "citymind_decentralized_v0", "is_parallelizable": True}
    render_mode = None

    def __init__(
        self,
        sumo_cfg: str,
        tls_metadata_json: str,
        sumo_binary: str = "sumo",
        decision_interval: int = 5,
        step_length: float = 1.0,
        obs_config: LocalObservationConfig | None = None,
        reward_config: LocalRewardConfig | None = None,
        safety_config: SafetyConfig | None = None,
        seed: int = 42,
        demand_id: str = "1.0x",
        episode_duration: float = 3600.0,
        sumo_extra_args: list[str] | None = None,
    ) -> None:
        self.sumo_cfg = str(Path(sumo_cfg).resolve())
        self.tls_metadata_json = str(Path(tls_metadata_json).resolve())
        self.sumo_binary = sumo_binary
        self.decision_interval = decision_interval
        self.step_length = step_length
        self._seed = seed
        self.demand_id = demand_id
        self.episode_duration = float(episode_duration)
        self.sumo_extra_args = sumo_extra_args or []

        self.tls_specs = load_tls_specs(Path(self.tls_metadata_json))
        self.specs_by_id = {s.tls_id: s for s in self.tls_specs}
        self.possible_agents = sorted(self.specs_by_id.keys())
        self.agents: list[str] = list(self.possible_agents)

        self.obs_cfg = obs_config or LocalObservationConfig()
        self.obs_builder = ObservationBuilder(self.tls_specs, ObservationConfig())
        self.local_reward = LocalRewardCalculator(reward_config)
        self.safety = SafetyLayer(self.tls_specs, safety_config or SafetyConfig())
        self.metrics = MetricsCollector()

        self._runtime_states: dict[str, TLSRuntimeState] = {}
        self.safety.register_states(self._runtime_states)
        self._downstream_cache: dict[str, list[str]] = {}
        self._sim_running = False
        self._step_count = 0

    # ──────────────────────────────────────────────────────────────────────
    # PettingZoo spaces (identical for every agent -> parameter sharing works)
    # ──────────────────────────────────────────────────────────────────────

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return spaces.Box(low=0.0, high=1.0, shape=(LOCAL_OBS_SIZE,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return spaces.Discrete(2)

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        if seed is not None:
            self._seed = seed
        if self._sim_running:
            self._close_sumo()

        self.agents = list(self.possible_agents)
        self._runtime_states.clear()
        for spec in self.tls_specs:
            self._runtime_states[spec.tls_id] = TLSRuntimeState(tls_id=spec.tls_id)

        self._start_sumo()
        for spec in self.tls_specs:
            tid = spec.tls_id
            try:
                phase = traci.trafficlight.getPhase(tid)
            except traci.exceptions.TraCIException:
                phase = 0
            self._runtime_states[tid].current_phase = phase
            self._runtime_states[tid].phase_elapsed = 0.0

        self.local_reward.reset()
        self.metrics.reset(controller="ippo", seed=self._seed, demand_id=self.demand_id)
        self._step_count = 0

        self.obs_builder.build(self._runtime_states)
        self.local_reward.refresh_lane_waiting()
        cache = self.obs_builder.last_step_cache

        observations = {
            tid: build_local_observation(
                self.specs_by_id[tid], self._runtime_states[tid], self.obs_cfg,
                self._downstream_cache, cache["total_vehicles"], cache["total_waiting"],
            )
            for tid in self.agents
        }
        infos = {tid: {} for tid in self.agents}
        return observations, infos

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]
    ]:
        if not self._sim_running:
            raise RuntimeError("step() called before reset()")

        switched_this_step = {tid: False for tid in self.agents}
        step_switches = 0
        for _ in range(self.decision_interval):
            sim_time = traci.simulation.getTime()
            prev_phases = {tid: s.current_phase for tid, s in self._runtime_states.items()}
            self.safety.step(sim_time, actions, self.step_length)
            for tid, state in self._runtime_states.items():
                if state.current_phase != prev_phases[tid]:
                    switched_this_step[tid] = True
                    step_switches += 1
            traci.simulationStep()

        self._step_count += 1
        self.obs_builder.build(self._runtime_states)
        self.local_reward.refresh_lane_waiting()
        cache = self.obs_builder.last_step_cache

        sim_time = traci.simulation.getTime()
        terminated_all = self._check_terminated(sim_time)
        truncated_all = sim_time >= self.episode_duration

        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        for tid in self.agents:
            spec = self.specs_by_id[tid]
            state = self._runtime_states[tid]
            observations[tid] = build_local_observation(
                spec, state, self.obs_cfg, self._downstream_cache,
                cache["total_vehicles"], cache["total_waiting"],
            )
            queue_sum = float(sum(state.lane_queue)) if state.lane_queue else 0.0
            reward, info = self.local_reward.compute(tid, spec, queue_sum, switched_this_step[tid])
            rewards[tid] = reward
            terminations[tid] = terminated_all
            truncations[tid] = truncated_all
            infos[tid] = info

        self.metrics.record_step(
            step=self._step_count,
            sim_time=sim_time,
            reward=sum(rewards.values()),
            switch_count=step_switches,
            cached_metrics=cache,
            states=self._runtime_states,
        )

        if terminated_all or truncated_all:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        if self._sim_running:
            self._close_sumo()

    # ──────────────────────────────────────────────────────────────────────
    # SUMO lifecycle (mirrors CentralizedSumoEnv)
    # ──────────────────────────────────────────────────────────────────────

    def _start_sumo(self) -> None:
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
            traci.start(cmd, label=f"citymind_ippo_{self._seed}_{int(time.time())}")
        self._sim_running = True

    def _close_sumo(self) -> None:
        try:
            traci.close()
        except Exception:
            pass
        self._sim_running = False

    def _check_terminated(self, sim_time: float) -> bool:
        if sim_time >= self.episode_duration:
            return True
        try:
            return traci.simulation.getMinExpectedNumber() == 0
        except Exception:
            return True
