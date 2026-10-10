"""
sumo_actuated.py
================
SUMO-actuated baseline controller.

This controller lets SUMO run its default/original traffic-light programs
without any external intervention. The RL policy does NOT act — SUMO's
built-in signal programs execute as-is from the InTAS configuration.

Named `sumo_actuated` (not "fixed_time") because 76 of InTAS's 98 tlLogic
programs are `type="actuated"` — vehicle-actuated, gap-based green
extension between minDur/maxDur — not fixed-duration. Verified directly on
2026-09-20/21: observed phase durations at TLS 1863241632 during the 07:00
peak were 7-10s against a programmed 6-35s, i.e. the controller was cutting
every green to its actuated minimum as soon as its detectors saw a gap.
This is what the project's own results tables previously called
"fixed_time" (artifacts/metrics/fixed_time_seed42_1.0x.json, pre-2026-09-21)
— that comparison was really "RL vs a calibrated adaptive controller", a
materially higher bar than a true fixed timer. See fixed_time.py for the
genuine static baseline added alongside this one.
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
from src.environment.metrics import GreenSwitchCounter, MetricsCollector
from src.environment.sumo_args import build_sumo_command
from src.environment.sumo_connection import USING_LIBSUMO, traci
from src.evaluation.tripinfo import parse_tripinfo


class SumoActuatedController(BaseController):
    """
    Runs the InTAS scenario with unmodified SUMO signal programs (a mix of
    vehicle-actuated and static tlLogic entries, per the network file).

    No TraCI signal commands are issued beyond the initial connection.
    SUMO executes its own programs from the .sumocfg / additional files.
    total_switches counts logical green-to-green changes of the displayed
    state (metrics.GreenSwitchCounter) -- it used to be hard-coded to 0.
    """

    def __init__(
        self,
        step_length: float = 1.0,
        episode_duration: int = 3600,
        sumo_binary: str = "sumo",
        sumo_extra_args: list[str] | None = None,
        decision_interval: int = 5,
        route_steps: int | None = None,
        teleport_time: float | None = None,
    ) -> None:
        self.step_length = step_length
        self.episode_duration = episode_duration
        self.sumo_binary = sumo_binary
        self.sumo_extra_args = sumo_extra_args or []
        self.decision_interval = decision_interval
        self.route_steps = route_steps
        self.teleport_time = teleport_time

    @property
    def name(self) -> str:
        return "sumo_actuated"

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
        if tripinfo_path:
            Path(tripinfo_path).parent.mkdir(parents=True, exist_ok=True)

        cmd = self.build_command(sumo_cfg, seed, tripinfo_path)
        if USING_LIBSUMO:
            traci.start(cmd)
        else:
            traci.start(cmd, label=f"sumo_actuated_{seed}")

        collector = MetricsCollector()
        switches = GreenSwitchCounter()
        trajectory_rows: list[dict] = []
        try:
            collector.reset("sumo_actuated", seed, demand_id, sumo_command=cmd)
            tls_ids = list(traci.trafficlight.getIDList())
            step = 0
            interval_switches = 0
            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                collector.tick()
                sim_time = traci.simulation.getTime()
                for tid in tls_ids:
                    if switches.update(tid, traci.trafficlight.getRedYellowGreenState(tid)):
                        interval_switches += 1
                if is_sample_step(step, self.decision_interval):
                    collector.record_step(step, sim_time, 0.0, interval_switches)
                    interval_switches = 0
                    if trajectory_path:
                        append_vehicle_rows(trajectory_rows, sim_time)
                        # SUMO owns these signal programs directly, so
                        # traci's own phase/state is authoritative here.
                        for tid in tls_ids:
                            try:
                                phase = traci.trafficlight.getPhase(tid)
                                state_str = traci.trafficlight.getRedYellowGreenState(tid)
                            except traci.exceptions.TraCIException:
                                continue
                            append_tls_row(trajectory_rows, sim_time, tid, phase, state_str)
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
            "sumo_actuated", summary, seed=seed, demand_id=demand_id,
            total_switches=switches.total, trip_summary=trip_summary, sumo_command=cmd,
        )
