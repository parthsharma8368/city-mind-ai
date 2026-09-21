"""
max_pressure.py
===============
Max-Pressure (MP) controller — a strong non-RL baseline.

Reference: Varaiya (2013), "Max Pressure Control of a Network of Signalized Intersections"

Decision rule per TLS per decision step:
  1. Compute pressure = upstream queue - downstream queue for the current
     green phase and for the next green phase in the fixed cycle. Downstream
     lanes are resolved live via traci.lane.getLinks(), so this is true
     Varaiya pressure (not upstream-only).
  2. Request an advance (same binary action the RL policy uses) only if the
     next phase has strictly higher pressure than the current one.
  3. The request is applied through the same SafetyLayer state machine used
     by the RL environment — full min/max green, yellow, and all-red
     clearance. Max-Pressure gets no unsafe shortcut the RL policy doesn't
     also get, so the comparison is apples-to-apples.

Max-Pressure is important because:
  - It is a well-known, provably throughput-optimal policy (under certain assumptions)
  - Beating fixed-time alone is insufficient to claim RL superiority
  - MP is widely used as a baseline in traffic-signal-control research
"""
from __future__ import annotations

from pathlib import Path

from src.environment.sumo_connection import traci, USING_LIBSUMO

from src.controllers.base import BaseController, ControllerResult
from src.environment.metrics import MetricsCollector
from src.environment.safety import SafetyLayer, SafetyConfig
from src.scenario.tls_metadata import TLSSpec, TLSRuntimeState, PhaseSpec, load_tls_specs


class MaxPressureController(BaseController):
    """
    Max-Pressure controller operating on the same SUMO simulation and the
    same SafetyLayer state machine as the RL environment.

    Each intersection independently selects, every decision step, whether to
    hold its current phase or advance to the next green phase in its cycle —
    based on which side of that choice has higher Varaiya pressure. This is
    still a *decentralized* decision rule, unlike the centralized RL policy,
    but it is now mechanically constrained the same way (same legal
    transitions, same metrics units) so the comparison is fair.
    """

    def __init__(
        self,
        min_green: float = 5.0,
        max_green: float = 60.0,
        yellow_duration: float = 3.0,
        all_red_duration: float = 1.0,
        decision_interval: int = 5,
        step_length: float = 1.0,
        episode_duration: int = 3600,
        pressure_margin: float = 0.0,
        sumo_binary: str = "sumo",
        sumo_extra_args: list[str] | None = None,
    ) -> None:
        self.decision_interval = decision_interval
        self.step_length = step_length
        self.episode_duration = episode_duration
        self.sumo_binary = sumo_binary
        self.sumo_extra_args = sumo_extra_args or []
        # Textbook Varaiya rule: switch whenever the next phase has strictly
        # higher pressure (margin=0). A pressure_margin > 0 (require the next
        # phase to clear the current one by more than this before switching)
        # is exposed for experimentation but is NOT a validated improvement
        # here: tested at margin=3.0 on this network under the SafetyLayer's
        # mandatory yellow/all-red clearance, mean waiting time got *worse*
        # (37.4s -> 46.7s) with only a modest drop in switch count (5096 ->
        # 4084) — the default stays at the unmodified rule rather than a
        # speculative "fix" that measurement didn't support.
        self.pressure_margin = pressure_margin
        self.safety_cfg = SafetyConfig(
            min_green=min_green,
            max_green=max_green,
            yellow_duration=yellow_duration,
            all_red_duration=all_red_duration,
        )

    @property
    def name(self) -> str:
        return "max_pressure"

    def run_episode(
        self,
        sumo_cfg: str,
        tls_metadata_json: str,
        seed: int = 42,
        demand_id: str = "1.0x",
        trajectory_path: str | None = None,
    ) -> ControllerResult:
        tls_specs = load_tls_specs(Path(tls_metadata_json))
        specs_by_id: dict[str, TLSSpec] = {s.tls_id: s for s in tls_specs}
        runtime_states: dict[str, TLSRuntimeState] = {
            s.tls_id: TLSRuntimeState(tls_id=s.tls_id) for s in tls_specs
        }
        safety = SafetyLayer(tls_specs, self.safety_cfg)
        safety.register_states(runtime_states)
        downstream_cache: dict[str, list[str]] = {}

        cmd = [
            self.sumo_binary,
            "--configuration-file", sumo_cfg,
            "--step-length", str(self.step_length),
            "--no-warnings",
            "--no-step-log",
            "--seed", str(seed),
        ] + self.sumo_extra_args

        if USING_LIBSUMO:
            traci.start(cmd)
        else:
            traci.start(cmd, label=f"max_pressure_{seed}_{demand_id}")

        # Sync initial phases from SUMO (same as CentralizedSumoEnv.reset()).
        for tid, state in runtime_states.items():
            try:
                state.current_phase = traci.trafficlight.getPhase(tid)
            except traci.exceptions.TraCIException:
                state.current_phase = 0
            state.phase_elapsed = 0.0

        collector = MetricsCollector()
        collector.reset("max_pressure", seed, demand_id)

        trajectory_rows = []
        step = 0
        action_dict: dict[str, int] = {tid: 0 for tid in runtime_states}
        total_switches_prev = {tid: s.current_phase for tid, s in runtime_states.items()}
        total_switches = 0

        try:
            while traci.simulation.getMinExpectedNumber() > 0:
                sim_time = traci.simulation.getTime()

                if step % self.decision_interval == 0:
                    action_dict = {
                        tid: self._decide_action(specs_by_id[tid], state, downstream_cache)
                        for tid, state in runtime_states.items()
                    }

                prev_phases = {tid: s.current_phase for tid, s in runtime_states.items()}
                safety.step(sim_time, action_dict, self.step_length)
                for tid, state in runtime_states.items():
                    if state.current_phase != prev_phases[tid]:
                        total_switches += 1

                traci.simulationStep()

                if step % self.decision_interval == 0:
                    # No states= here: this keeps max-pressure on the same raw,
                    # all-network-lane TraCI query fixed_time.py uses, so mean_queue
                    # / mean_speed stay on identical units across every controller.
                    collector.record_step(step, sim_time, 0.0, total_switches)
                    if trajectory_path:
                        for vid in traci.vehicle.getIDList():
                            x, y = traci.vehicle.getPosition(vid)
                            spd = traci.vehicle.getSpeed(vid)
                            trajectory_rows.append({
                                "time": sim_time, "vehicle_id": vid, "x": x, "y": y, "speed": spd,
                                "tls_id": None, "phase": None, "tls_state": None,
                            })
                        # The safety layer drives signals via raw state strings, so our
                        # own tracked phase (not traci's program cursor) is authoritative.
                        for tid, state in runtime_states.items():
                            spec = specs_by_id[tid]
                            phase_idx = state.current_phase
                            state_str = (
                                spec.phases[phase_idx].state if phase_idx < len(spec.phases) else ""
                            )
                            trajectory_rows.append({
                                "time": sim_time, "vehicle_id": None, "x": None, "y": None, "speed": None,
                                "tls_id": tid, "phase": phase_idx, "tls_state": state_str,
                            })

                step += 1
                if sim_time >= self.episode_duration:
                    break
        finally:
            traci.close()

        if trajectory_path and trajectory_rows:
            import pandas as pd
            Path(trajectory_path).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(trajectory_rows).to_parquet(trajectory_path, index=False)

        summary = collector.summarize()
        return ControllerResult(
            controller="max_pressure",
            seed=seed,
            demand_id=demand_id,
            total_steps=summary.episode_steps,
            episode_duration=summary.episode_duration,
            mean_waiting_time=summary.mean_waiting_time,
            total_waiting_time=summary.total_waiting_time_sum,
            mean_queue=summary.mean_queue,
            max_queue=summary.max_queue,
            throughput=summary.throughput,
            total_arrived=summary.total_arrived,
            total_departed=summary.total_departed,
            mean_speed=summary.mean_speed,
            total_switches=total_switches,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Pressure computation
    # ──────────────────────────────────────────────────────────────────────────

    def _get_downstream_lanes(self, lane_id: str) -> list[str]:
        try:
            links = traci.lane.getLinks(lane_id)
        except traci.exceptions.TraCIException:
            return []
        return [link[0] for link in links if link and link[0]]

    def _phase_pressure(
        self,
        spec: TLSSpec,
        phase: PhaseSpec,
        downstream_cache: dict[str, list[str]],
    ) -> float:
        """Varaiya pressure: sum over green movements of (upstream - downstream queue)."""
        pressure = 0.0
        for i, sig in enumerate(phase.state):
            if sig not in ("G", "g") or i >= len(spec.controlled_lanes):
                continue
            lane_id = spec.controlled_lanes[i]
            try:
                upstream = traci.lane.getLastStepHaltingNumber(lane_id)
            except traci.exceptions.TraCIException:
                upstream = 0.0

            downstream_lanes = downstream_cache.get(lane_id)
            if downstream_lanes is None:
                downstream_lanes = self._get_downstream_lanes(lane_id)
                downstream_cache[lane_id] = downstream_lanes

            downstream = 0.0
            for dl in downstream_lanes:
                try:
                    downstream += traci.lane.getLastStepHaltingNumber(dl)
                except traci.exceptions.TraCIException:
                    pass

            pressure += upstream - downstream
        return pressure

    def _decide_action(
        self,
        spec: TLSSpec,
        state: TLSRuntimeState,
        downstream_cache: dict[str, list[str]],
    ) -> int:
        """Binary keep(0)/advance(1) decision, applied through the SafetyLayer."""
        if not spec.phases or state.yellow_pending:
            return 0
        current_phase = spec.phases[state.current_phase]
        if not current_phase.is_green:
            return 0

        next_idx = spec.next_green_phase(state.current_phase)
        if next_idx == state.current_phase:
            return 0

        current_pressure = self._phase_pressure(spec, current_phase, downstream_cache)
        next_pressure = self._phase_pressure(spec, spec.phases[next_idx], downstream_cache)
        return 1 if next_pressure > current_pressure + self.pressure_margin else 0
