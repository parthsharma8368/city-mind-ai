"""
decentralized_sumo_env.py
==========================
DecentralizedSumoEnv - a PettingZoo ParallelEnv with ONE agent per traffic signal (98 on InTAS) and a shared policy.

  observation  22 numbers per signal (local_observation.py)
  action       Discrete(2): 0 = keep the current green, 1 = advance to the next green phase of the signal's own program
               (the SafetyLayer enforces min / max green and the program's own yellow / all-red clearance)
  reward       per signal, a level of delay (local_reward.py)
  decision     every ``decision_interval`` (5) simulated seconds

Training wraps it with SuperSuit so one SB3 PPO model learns from all 98 agents at once (src/agents/ippo_agent.py);
evaluation steps it directly.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from src.environment.lane_stats import LaneStatsReader
from src.environment.local_observation import (
    N_ROAD_NEIGHBORS,
    OBS_SIZE,
    LocalObservationConfig,
    build_observation,
    compute_downstream_queue_sum,
    compute_feeder_lanes,
    compute_per_phase_pressure,
    get_downstream_lanes,
)
from src.environment.local_reward import LocalRewardCalculator, LocalRewardConfig
from src.environment.metrics import MetricsCollector
from src.environment.safety import SafetyConfig, SafetyLayer
from src.environment.sumo_connection import traci
from src.environment.sumo_session import SumoSessionMixin, mark_time_limit
from src.scenario.tls_metadata import TLSRuntimeState, load_tls_specs


class DecentralizedSumoEnv(SumoSessionMixin, ParallelEnv):
    """One agent per signal, parameter-shared observation / reward / action."""

    metadata = {"name": "citymind_decentralized_v1", "is_parallelizable": True}
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
        seed_mode: str = "fixed",
        warm_start_seconds: float = 0.0,
        freeze_native_program: bool = True,
        route_steps: int | None = None,
        teleport_time: float | None = None,
    ) -> None:
        """seed_mode "fixed" (evaluation: the constructor seed every episode) | "per_episode" (training: seed + episode
        index); warm_start_seconds / freeze_native_program / route_steps / teleport_time: see sumo_session.py."""
        self.sumo_cfg = str(Path(sumo_cfg).resolve())
        self.tls_metadata_json = str(Path(tls_metadata_json).resolve())
        self.sumo_binary = sumo_binary
        self.decision_interval = decision_interval
        self.step_length = step_length
        self._seed = seed
        self.demand_id = demand_id
        self.episode_duration = float(episode_duration)
        self.sumo_extra_args = sumo_extra_args or []
        self._init_session(
            seed_mode=seed_mode, warm_start_seconds=warm_start_seconds,
            freeze_native_program=freeze_native_program, route_steps=route_steps,
            teleport_time=teleport_time, label_prefix="citymind_ippo",
        )

        self.tls_specs = load_tls_specs(Path(self.tls_metadata_json))
        self.specs_by_id = {s.tls_id: s for s in self.tls_specs}
        self.possible_agents = sorted(self.specs_by_id.keys())
        self.agents: list[str] = list(self.possible_agents)

        self.obs_cfg = obs_config or LocalObservationConfig()
        self.lane_reader = LaneStatsReader(self.tls_specs)
        self.local_reward = LocalRewardCalculator(reward_config)
        self.safety = SafetyLayer(self.tls_specs, safety_config or SafetyConfig())
        self.metrics = MetricsCollector()

        self._runtime_states: dict[str, TLSRuntimeState] = {}
        self.safety.register_states(self._runtime_states)
        self._downstream_cache: dict[str, list[str]] = {}
        self._sim_running = False
        self._step_count = 0

        # Static topology, derived from the network (no ids are written by hand).
        # signalised edge -> first signal controlling it / its stop-line lanes; signal -> indices of its UNIQUE lanes
        # (``controlled_lanes`` has one entry per LINK, so a lane serving several links repeats).
        self._edge_to_tls: dict[str, str] = {}
        self._edge_lanes: dict[str, list[str]] = {}
        self._unique_lane_idx: dict[str, list[int]] = {}
        for spec in self.tls_specs:
            seen: set[str] = set()
            self._unique_lane_idx[spec.tls_id] = []
            for i, lane in enumerate(spec.controlled_lanes):
                edge = lane.rsplit("_", 1)[0]
                self._edge_to_tls.setdefault(edge, spec.tls_id)
                if lane not in seen:
                    seen.add(lane)
                    self._unique_lane_idx[spec.tls_id].append(i)
                    self._edge_lanes.setdefault(edge, []).append(lane)
        # Built at the first reset (need a live SUMO connection), then cached for the instance's lifetime.
        self._road_up: dict[str, list[str]] | None = None
        self._road_down: dict[str, list[str]] | None = None
        self._feeders: dict[str, list[str]] | None = None
        # Per-step caches.
        self._halt_cache: dict[str, float] = {}
        self._queue_level: dict[str, float] = {}
        self._pending_lane_extra: dict[str, float] = {}

    # ── PettingZoo spaces (identical for every agent -> parameter sharing works) ──
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return spaces.Discrete(2)

    # ── lifecycle ──
    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        self._resolve_episode_seed(seed)
        if self._sim_running:
            self._close_sumo()

        self.agents = list(self.possible_agents)
        self._runtime_states.clear()
        for spec in self.tls_specs:
            self._runtime_states[spec.tls_id] = TLSRuntimeState(tls_id=spec.tls_id)

        # Shared command builder + native warm start + takeover (freezes SUMO's own program and adopts its phase as the
        # believed phase). If anything below fails the SUMO child is closed, never leaked.
        self._start_sumo()
        try:
            if self._road_up is None:
                self._road_up, self._road_down = self._compute_road_neighbors()
            if self.obs_cfg.queue_reach_m > 0 and self._feeders is None:
                self._feeders = self._compute_feeders()
            if not self.freeze_native_program:
                for spec in self.tls_specs:
                    try:
                        phase = traci.trafficlight.getPhase(spec.tls_id)
                    except traci.exceptions.TraCIException:
                        phase = 0
                    self._runtime_states[spec.tls_id].current_phase = phase
                    self._runtime_states[spec.tls_id].phase_elapsed = 0.0
            return self._finish_reset()
        except BaseException:
            self._close_sumo()
            raise

    def _refresh_step_state(self) -> tuple[dict, dict, dict]:
        """Everything that is read from SUMO once per decision step, in the order the observation depends on."""
        self.lane_reader.read(self._runtime_states)
        self._apply_queue_reach()
        self.local_reward.refresh_pending(self._edge_to_tls)
        self._update_pending_lanes()
        downstream_sums, phase_pressures = self._compute_per_tls_aux()
        return self.lane_reader.last_step_cache, downstream_sums, phase_pressures

    def _finish_reset(self) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        self.local_reward.reset()
        self.metrics.reset(
            controller="ippo", seed=self._seed, demand_id=self.demand_id, sumo_command=self.last_sumo_command,
        )
        self._step_count = 0
        _, _, phase_pressures = self._refresh_step_state()
        observations = {tid: self._obs_for(tid, phase_pressures[tid]) for tid in self.agents}
        infos = {tid: {"seed": self._seed, "sumo_command": self.last_sumo_command} for tid in self.agents}
        return observations, infos

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]
    ]:
        if not self._sim_running:
            raise RuntimeError("step() called before reset()")

        switched = {tid: False for tid in self.agents}
        step_switches = 0
        for _ in range(self.decision_interval):
            sim_time = traci.simulation.getTime()
            prev_phases = {tid: s.current_phase for tid, s in self._runtime_states.items()}
            self.safety.step(sim_time, actions, self.step_length)
            for tid, state in self._runtime_states.items():
                if state.current_phase != prev_phases[tid]:
                    switched[tid] = True
                    step_switches += 1
            traci.simulationStep()
            self.metrics.tick()

        self._step_count += 1
        cache, downstream_sums, phase_pressures = self._refresh_step_state()

        sim_time = traci.simulation.getTime()
        # Time limit => truncated only (terminated False), so GAE bootstraps from the last state's value.
        terminated_all, truncated_all = self._episode_end_flags(sim_time)

        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        for tid in self.agents:
            observations[tid] = self._obs_for(tid, phase_pressures[tid])
            reward, info = self.local_reward.compute(
                tid, switched[tid], queue_level=self._queue_level.get(tid, 0.0),
                downstream_level=downstream_sums[tid], n_agents=len(self.possible_agents),
            )
            mark_time_limit(info, terminated_all, truncated_all)
            rewards[tid] = reward
            terminations[tid] = terminated_all
            truncations[tid] = truncated_all
            infos[tid] = info

        self.metrics.record_step(
            step=self._step_count, sim_time=sim_time, reward=sum(rewards.values()),
            switch_count=step_switches, cached_metrics=cache,
        )
        if terminated_all or truncated_all:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    # ── per-step helpers ──
    def _compute_per_tls_aux(self) -> tuple[dict[str, float], dict[str, list[float]]]:
        """Per signal: mean halted cars per downstream lane (reward spillback term) and the demand of each of its green
        phases (observation), computed once and shared."""
        downstream_sums: dict[str, float] = {}
        phase_pressures: dict[str, list[float]] = {}
        for tid in self.agents:
            spec = self.specs_by_id[tid]
            downstream_sums[tid] = compute_downstream_queue_sum(spec, self._downstream_cache)
            phase_pressures[tid] = compute_per_phase_pressure(
                spec, self._downstream_cache, feeders=self._feeders,
                halting=self._halting if self._feeders is not None else None,
                extra_upstream=self._pending_lane_extra or None,
            )
        return downstream_sums, phase_pressures

    def _obs_for(self, tid: str, phase_pressures: list[float]) -> np.ndarray:
        spec, state = self.specs_by_id[tid], self._runtime_states[tid]

        def worst(tids: list[str]) -> list[float]:
            return [max(self._runtime_states[t].lane_queue or [0.0]) for t in tids if t in self._runtime_states]

        min_g, max_g = self._timing_bounds(spec, state)
        return build_observation(
            spec, state, self.obs_cfg, phase_pressures=phase_pressures, min_green=min_g, max_green=max_g,
            pending_count=self.local_reward.pending_for(tid),
            upstream_max_queues=worst(self._road_up.get(tid, [])),
            downstream_max_queues=worst(self._road_down.get(tid, [])),
        )

    def _update_pending_lanes(self) -> None:
        """pending_in_demand: spread each signalised edge's waiting-outside cars over that edge's stop-line lanes."""
        self._pending_lane_extra = {}
        if not self.obs_cfg.pending_in_demand:
            return
        for edge, n in self.local_reward.pending_by_edge().items():
            lanes = self._edge_lanes.get(edge, [])
            for lane in lanes:
                self._pending_lane_extra[lane] = n / len(lanes)

    def _timing_bounds(self, spec, state) -> tuple[float, float]:
        """(min_green, max_green) behind the v2/v3 elapsed ratios; see LocalObservationConfig.timing_source."""
        phase = spec.phases[state.current_phase] if 0 <= state.current_phase < len(spec.phases) else None
        if phase is None:
            return self.safety.cfg.min_green, self.safety.cfg.max_green
        if self.obs_cfg.timing_source == "program":
            lo = phase.min_dur if phase.min_dur > 0 else self.safety.cfg.min_green
            hi = max(phase.max_dur, phase.duration)
            return lo, (hi if hi > 0 else self.safety.cfg.max_green)
        return self.safety._green_bounds(phase)

    def _halting(self, lane: str) -> float:
        """Halting count of a lane, read once per decision step."""
        v = self._halt_cache.get(lane)
        if v is None:
            try:
                v = float(traci.lane.getLastStepHaltingNumber(lane))
            except traci.exceptions.TraCIException:
                v = 0.0
            self._halt_cache[lane] = v
        return v

    def _apply_queue_reach(self) -> None:
        """Per-TLS queue_level for the reward and, when queue_reach_m > 0, add the halted cars on the unsignalised
        feeder lanes behind each stop-line lane to state.lane_queue (read by max_queue, the demand inputs' siblings
        and the queue reward). Runs right after ObservationBuilder.build() refreshed the raw per-lane values."""
        self._halt_cache = {}
        for tid in self.agents:
            state = self._runtime_states[tid]
            own = state.lane_queue or []
            idx = self._unique_lane_idx[tid]
            level = float(sum(own[i] for i in idx))
            if self._feeders is not None and own:
                lanes = self.specs_by_id[tid].controlled_lanes
                extended, union = [], set()
                for lane, q in zip(lanes, own):
                    fs = self._feeders.get(lane, ())
                    extended.append(q + sum(self._halting(f) for f in fs))
                    union.update(fs)
                state.lane_queue = extended
                level += sum(self._halting(f) for f in union)
            self._queue_level[tid] = level

    def _compute_feeders(self) -> dict[str, list[str]]:
        lengths: dict[str, float] = {}

        def lane_length(lane: str) -> float:
            if lane not in lengths:
                try:
                    lengths[lane] = float(traci.lane.getLength(lane))
                except traci.exceptions.TraCIException:
                    lengths[lane] = 0.0
            return lengths[lane]

        incoming = [ln for s in self.tls_specs for ln in s.controlled_lanes]
        return compute_feeder_lanes(
            incoming, self._downstream_cache, list(traci.lane.getIDList()),
            set(self._edge_to_tls), lane_length, self.obs_cfg.queue_reach_m,
        )

    def _compute_road_neighbors(
        self, max_hops: int = 6, max_distance_m: float = 500.0
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Road-connected neighbours: for every TLS, the signals its traffic reaches next (downstream) and the
        signals that feed it (upstream), found by walking the road graph from each outgoing lane until the next
        signalised edge (at most ``max_hops`` edges / ``max_distance_m`` metres, through unsignalised junctions).
        Each (lane -> lane) connection found adds 1 to the pair's weight; the ``N_ROAD_NEIGHBORS`` strongest per
        direction are kept. Requires a live SUMO connection; static, so computed once."""
        lanes_by_edge: dict[str, list[str]] = {}
        for ln in traci.lane.getIDList():
            if not ln.startswith(":"):
                lanes_by_edge.setdefault(ln.rsplit("_", 1)[0], []).append(ln)

        def successors(edge: str) -> set[str]:
            out: set[str] = set()
            for ln in lanes_by_edge.get(edge, []):
                for dl in get_downstream_lanes(ln, self._downstream_cache):
                    if not dl.startswith(":"):
                        out.add(dl.rsplit("_", 1)[0])
            return out

        def length(edge: str) -> float:
            try:
                return float(traci.lane.getLength(lanes_by_edge[edge][0]))
            except (KeyError, IndexError, traci.exceptions.TraCIException):
                return 0.0

        down_w: dict[str, dict[str, int]] = {t: {} for t in self.possible_agents}
        up_w: dict[str, dict[str, int]] = {t: {} for t in self.possible_agents}
        for tid in self.possible_agents:
            spec = self.specs_by_id[tid]
            for lane in dict.fromkeys(spec.controlled_lanes):
                for dl in get_downstream_lanes(lane, self._downstream_cache):
                    if dl.startswith(":"):
                        continue
                    frontier = [(dl.rsplit("_", 1)[0], 0.0, 0)]
                    seen: set[str] = set()
                    while frontier:
                        edge, dist, hops = frontier.pop()
                        if edge in seen:
                            continue
                        seen.add(edge)
                        owner = self._edge_to_tls.get(edge)
                        if owner is not None:
                            if owner != tid:
                                down_w[tid][owner] = down_w[tid].get(owner, 0) + 1
                                up_w[owner][tid] = up_w[owner].get(tid, 0) + 1
                            continue  # the next signal ends this branch (also when it is ourselves)
                        dist += length(edge)
                        if hops + 1 > max_hops or dist > max_distance_m:
                            continue
                        frontier.extend((n, dist, hops + 1) for n in successors(edge))

        def top(weights: dict[str, int]) -> list[str]:
            return [t for t, _ in sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))[:N_ROAD_NEIGHBORS]]

        return {t: top(w) for t, w in up_w.items()}, {t: top(w) for t, w in down_w.items()}
