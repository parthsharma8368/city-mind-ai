"""
max_pressure.py
===============
Max-Pressure (MP) controller — a strong non-RL baseline.

Reference: Varaiya (2013), "Max Pressure Control of a Network of Signalized Intersections"

Algorithm per TLS per decision step:
  1. For each phase, compute pressure = sum of upstream queue - sum of downstream queue
     over all movements served by that phase.
  2. Select the phase with the highest total pressure.
  3. Apply via TraCI, subject to minimum green constraint.

Max-Pressure is important because:
  - It is a well-known, provably throughput-optimal policy (under certain assumptions)
  - Beating fixed-time alone is insufficient to claim RL superiority
  - MP is widely used as a baseline in traffic-signal-control research
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from src.environment.sumo_connection import traci, USING_LIBSUMO

import numpy as np
from dataclasses import dataclass, field

from src.controllers.base import BaseController, ControllerResult
from src.environment.metrics import MetricsCollector
from src.scenario.tls_metadata import TLSSpec, TLSRuntimeState, load_tls_specs


@dataclass
class MPState:
    """Runtime state for one TLS under Max-Pressure control."""
    tls_id: str
    current_phase: int = 0
    phase_elapsed: float = 0.0  # seconds


class MaxPressureController(BaseController):
    """
    Decentralized Max-Pressure controller operating on the same SUMO simulation.

    Each intersection independently selects the phase with maximum pressure.
    Min green constraint is enforced before switching.

    Note: MP is a *decentralized* controller — each intersection acts independently.
    This is contrasted with the centralized RL policy which observes/acts globally.
    """

    def __init__(
        self,
        min_green: float = 10.0,
        decision_interval: int = 5,
        step_length: float = 1.0,
        episode_duration: int = 3600,
    ) -> None:
        self.min_green = min_green
        self.decision_interval = decision_interval
        self.step_length = step_length
        self.episode_duration = episode_duration

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
        mp_states: dict[str, MPState] = {
            s.tls_id: MPState(tls_id=s.tls_id) for s in tls_specs
        }
        specs_by_id: dict[str, TLSSpec] = {s.tls_id: s for s in tls_specs}

        cmd = [
            "sumo",
            "--configuration-file", sumo_cfg,
            "--step-length", str(self.step_length),
            "--no-warnings",
            "--no-step-log",
            "--seed", str(seed),
        ]

        if USING_LIBSUMO:
            traci.start(cmd)
        else:
            traci.start(cmd, label=f"max_pressure_{seed}")

        collector = MetricsCollector()
        collector.reset("max_pressure", seed, demand_id)

        trajectory_rows = []
        step = 0
        total_switches = 0
        try:
            while traci.simulation.getMinExpectedNumber() > 0:
                sim_time = traci.simulation.getTime()

                # Make MP decisions every decision_interval steps
                if step % self.decision_interval == 0:
                    switches = self._apply_mp_decisions(specs_by_id, mp_states, sim_time)
                    total_switches += switches

                traci.simulationStep()

                # Update elapsed time for each TLS
                for tid, state in mp_states.items():
                    state.phase_elapsed += self.step_length

                if step % 5 == 0:
                    collector.record_step(step, sim_time, 0.0, total_switches)
                    if trajectory_path:
                        for vid in traci.vehicle.getIDList():
                            x, y = traci.vehicle.getPosition(vid)
                            spd = traci.vehicle.getSpeed(vid)
                            trajectory_rows.append({
                                "time": sim_time,
                                "vehicle_id": vid,
                                "x": x,
                                "y": y,
                                "speed": spd,
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

    def _apply_mp_decisions(
        self,
        specs: dict[str, TLSSpec],
        states: dict[str, MPState],
        sim_time: float,
    ) -> int:
        """Apply Max-Pressure phase selection to all TLS. Returns switch count."""
        switches = 0
        for tid, state in states.items():
            spec = specs[tid]
            # Only switch if minimum green has elapsed
            if state.phase_elapsed < self.min_green:
                continue

            best_phase = self._select_max_pressure_phase(spec, state)
            if best_phase != state.current_phase:
                # Insert yellow (simplified: direct phase switch via SUMO)
                # For production, integrate the safety layer here too
                try:
                    traci.trafficlight.setPhase(tid, best_phase)
                    state.current_phase = best_phase
                    state.phase_elapsed = 0.0
                    switches += 1
                except traci.exceptions.TraCIException:
                    pass
        return switches

    def _select_max_pressure_phase(
        self, spec: TLSSpec, state: MPState
    ) -> int:
        """
        Select the green phase with maximum pressure.

        Pressure for a phase = sum of upstream queue lengths for lanes that
        receive green in this phase minus sum of downstream queue lengths
        for the downstream lanes.

        Simplified implementation: uses only upstream halting vehicles
        (downstream estimation via downstream lane queues requires lane link
        topology which we approximate here).
        """
        if not spec.phases:
            return state.current_phase

        best_phase = state.current_phase
        best_pressure = -float("inf")

        for phase in spec.phases:
            if not phase.is_green:
                continue

            # Identify which lanes get green in this phase
            # SUMO signal state: 'G'/'g' = green for that lane position
            pressure = 0.0
            for i, sig in enumerate(phase.state):
                if sig in ("G", "g") and i < len(spec.controlled_lanes):
                    lane_id = spec.controlled_lanes[i]
                    try:
                        upstream_queue = traci.lane.getLastStepHaltingNumber(lane_id)
                        pressure += upstream_queue
                    except traci.exceptions.TraCIException:
                        pass

            if pressure > best_pressure:
                best_pressure = pressure
                best_phase = phase.index

        return best_phase
