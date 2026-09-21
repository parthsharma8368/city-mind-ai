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
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from src.environment.sumo_connection import traci, USING_LIBSUMO
from src.environment.local_observation import (
    LOCAL_OBS_SIZE,
    N_NEIGHBORS,
    LocalObservationConfig,
    build_local_observation,
    compute_downstream_queue_sum,
    compute_intersection_pressure,
    compute_mean_queue_norm,
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
        # Built lazily on first reset() (needs a live SUMO connection for
        # traci.junction.getPosition); positions are static, so this is
        # computed once and cached for the env instance's lifetime.
        self._neighbor_map: dict[str, list[str]] | None = None

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
        if self._neighbor_map is None:
            self._neighbor_map = self._compute_neighbor_map()
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
        own_queue_norms, own_downstream_sums, _own_pressures = self._compute_per_tls_aux()

        observations = {
            tid: build_local_observation(
                self.specs_by_id[tid], self._runtime_states[tid], self.obs_cfg,
                self._downstream_cache, cache["total_vehicles"], cache["total_waiting"],
                downstream_queue_sum=own_downstream_sums[tid],
                neighbor_queue_norms=[own_queue_norms.get(n, 0.0) for n in self._neighbor_map.get(tid, [])],
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
        own_queue_norms, own_downstream_sums, own_pressures = self._compute_per_tls_aux()

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
                downstream_queue_sum=own_downstream_sums[tid],
                neighbor_queue_norms=[own_queue_norms.get(n, 0.0) for n in self._neighbor_map.get(tid, [])],
            )
            queue_sum = float(sum(state.lane_queue)) if state.lane_queue else 0.0
            reward, info = self.local_reward.compute(
                tid, spec, queue_sum, switched_this_step[tid],
                downstream_queue_sum=own_downstream_sums[tid],
                pressure=own_pressures[tid],
            )
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
    # Neighbor coordination support (2026-09-20)
    # ──────────────────────────────────────────────────────────────────────

    def _compute_per_tls_aux(self) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Every agent's own mean-queue-norm, raw downstream-queue sum, and
        Varaiya pressure for this step, computed once and shared:
        build_local_observation needs the first two for this TLS's own
        features AND to answer neighbor lookups for whichever *other* TLS
        have it as one of their 2 nearest neighbors; local_reward.compute()
        reuses the downstream sum for the spillback penalty and the
        pressure value for the pressure-reward ablation arm (2026-09-21),
        instead of re-querying TraCI for either."""
        own_queue_norms: dict[str, float] = {}
        own_downstream_sums: dict[str, float] = {}
        own_pressures: dict[str, float] = {}
        for tid in self.agents:
            state = self._runtime_states[tid]
            spec = self.specs_by_id[tid]
            own_queue_norms[tid] = compute_mean_queue_norm(state, self.obs_cfg)
            own_downstream_sums[tid] = compute_downstream_queue_sum(spec, self._downstream_cache)
            own_pressures[tid] = compute_intersection_pressure(spec, state, self._downstream_cache)
        return own_queue_norms, own_downstream_sums, own_pressures

    def _compute_neighbor_map(self, k: int = N_NEIGHBORS) -> dict[str, list[str]]:
        """Each TLS's k geographically-nearest other TLS, by straight-line
        distance between junction positions (a proximity proxy, not true
        road-topology adjacency — same approach and same caveat as the XAI
        empirical spillback-correlation heatmap). Requires an active SUMO
        connection; computed once and cached since positions are static."""
        positions: dict[str, tuple[float, float]] = {}
        for tid in self.possible_agents:
            try:
                positions[tid] = traci.junction.getPosition(tid)
                continue
            except traci.exceptions.TraCIException:
                pass
            # ~8% of InTAS's tls_ids (auto-generated "gneJ..." nodes absorbed
            # into a joined-TLS cluster) aren't recognized by the Junction
            # API at all — verified against the live network. Every TLS's
            # own controlled lanes definitely exist, so fall back to the
            # average of their stop-line endpoints as an approximate
            # position instead of leaving this TLS with zero neighbors.
            spec = self.specs_by_id[tid]
            endpoints = []
            for lane_id in spec.controlled_lanes:
                try:
                    endpoints.append(traci.lane.getShape(lane_id)[-1])
                except traci.exceptions.TraCIException:
                    pass
            if endpoints:
                positions[tid] = (
                    sum(p[0] for p in endpoints) / len(endpoints),
                    sum(p[1] for p in endpoints) / len(endpoints),
                )

        neighbor_map: dict[str, list[str]] = {}
        for tid in self.possible_agents:
            if tid not in positions:
                neighbor_map[tid] = []
                continue
            px, py = positions[tid]
            distances = sorted(
                (
                    (math.hypot(px - positions[other][0], py - positions[other][1]), other)
                    for other in self.possible_agents
                    if other != tid and other in positions
                ),
                key=lambda d: d[0],
            )
            neighbor_map[tid] = [other for _, other in distances[:k]]
        return neighbor_map

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
