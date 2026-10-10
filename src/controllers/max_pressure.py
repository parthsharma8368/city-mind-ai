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

from src.controllers.base import (
    BaseController,
    ControllerResult,
    append_tls_row,
    append_vehicle_rows,
    is_sample_step,
    result_from_summary,
)
from src.environment.metrics import MetricsCollector
from src.environment.safety import SafetyConfig, SafetyLayer
from src.environment.sumo_args import build_sumo_command
from src.environment.sumo_connection import USING_LIBSUMO, traci
from src.evaluation.tripinfo import parse_tripinfo
from src.scenario.tls_metadata import PhaseSpec, TLSRuntimeState, TLSSpec, load_tls_specs


class MaxPressureController(BaseController):
    """
    Max-Pressure controller operating on the same SUMO simulation and the
    same SafetyLayer state machine as the RL environment.

    Each intersection independently selects, every decision step, whether to
    hold its current phase or advance to the next green phase in its cycle —
    based on which side of that choice has higher Varaiya pressure. This is
    still a *decentralized* decision rule, unlike the centralized RL policy,
    but it is now mechanically constrained the same way (same legal
    transitions, same metrics definition, same takeover-at-t0 and sampling
    grid) so the comparison is fair.
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
        route_steps: int | None = None,
        teleport_time: float | None = None,
    ) -> None:
        self.route_steps = route_steps
        self.teleport_time = teleport_time
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

    def build_command(self, sumo_cfg: str, seed: int, tripinfo_path: str | None = None) -> list[str]:
        """The exact SUMO argv (shared builder -- see environment.sumo_args)."""
        return build_sumo_command(
            self.sumo_binary, sumo_cfg, step_length=self.step_length, seed=seed,
            route_steps=self.route_steps, teleport_time=self.teleport_time,
            tripinfo_output=tripinfo_path, extra_args=self.sumo_extra_args,
        )

    def run_episode(
        self,
        sumo_cfg: str,
        tls_metadata_json: str,
        seed: int = 42,
        demand_id: str = "1.0x",
        trajectory_path: str | None = None,
        tripinfo_path: str | None = None,
    ) -> ControllerResult:
        tls_specs = load_tls_specs(Path(tls_metadata_json))
        specs_by_id: dict[str, TLSSpec] = {s.tls_id: s for s in tls_specs}
        runtime_states: dict[str, TLSRuntimeState] = {
            s.tls_id: TLSRuntimeState(tls_id=s.tls_id) for s in tls_specs
        }
        safety = SafetyLayer(tls_specs, self.safety_cfg)
        safety.register_states(runtime_states)
        downstream_cache: dict[str, list[str]] = {}

        if tripinfo_path:
            Path(tripinfo_path).parent.mkdir(parents=True, exist_ok=True)

        cmd = self.build_command(sumo_cfg, seed, tripinfo_path)
        if USING_LIBSUMO:
            traci.start(cmd)
        else:
            traci.start(cmd, label=f"max_pressure_{seed}_{demand_id}")

        collector = MetricsCollector()
        trajectory_rows: list[dict] = []
        total_switches = 0
        try:
            # Freeze SUMO's native program at t0 and own the signals from the
            # first step (rl-core-003) -- identical to what the RL envs do.
            safety.take_over()

            collector.reset("max_pressure", seed, demand_id, sumo_command=cmd)

            step = 0
            interval_switches = 0
            action_dict: dict[str, int] = {tid: 0 for tid in runtime_states}
            while traci.simulation.getMinExpectedNumber() > 0:
                t_before = traci.simulation.getTime()

                if step % self.decision_interval == 0:
                    action_dict = {
                        tid: self._decide_action(specs_by_id[tid], state, downstream_cache)
                        for tid, state in runtime_states.items()
                    }

                prev_phases = {tid: s.current_phase for tid, s in runtime_states.items()}
                safety.step(t_before, action_dict, self.step_length)
                for tid, state in runtime_states.items():
                    if state.current_phase != prev_phases[tid]:
                        total_switches += 1
                        interval_switches += 1

                traci.simulationStep()
                collector.tick()
                # Post-step time: same sampling grid as every other controller
                # (eval-scripts-14: this loop used to read the time BEFORE
                # stepping and ran one extra second).
                sim_time = traci.simulation.getTime()

                if is_sample_step(step, self.decision_interval):
                    collector.record_step(step, sim_time, 0.0, interval_switches)
                    interval_switches = 0
                    if trajectory_path:
                        append_vehicle_rows(trajectory_rows, sim_time)
                        # The signal string SUMO is actually displaying (yellow /
                        # all-red clearance included), not the frozen green.
                        for tid, state in runtime_states.items():
                            append_tls_row(
                                trajectory_rows, sim_time, tid, state.current_phase,
                                safety.displayed_state(tid),
                            )

                step += 1
                if sim_time >= self.episode_duration:
                    break
            collector.record_backlog()
        finally:
            traci.close()

        if trajectory_path and trajectory_rows:
            import pandas as pd
            Path(trajectory_path).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(trajectory_rows).to_parquet(trajectory_path, index=False)

        summary = collector.summarize()
        trip_summary = parse_tripinfo(tripinfo_path) if tripinfo_path else None
        return result_from_summary(
            "max_pressure", summary, seed=seed, demand_id=demand_id,
            total_switches=total_switches, trip_summary=trip_summary, sumo_command=cmd,
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
            if sig not in ("G", "g"):
                continue
            # rl-core-007: signal index i -> its own incoming lane(s) via
            # spec.signal_lanes (controlled_lanes is flattened per link and is
            # NOT indexable by signal for 5 TLS).
            for lane_id in spec.incoming_lanes_for_signal(i):
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
