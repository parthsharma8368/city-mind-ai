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

from src.environment.sumo_connection import traci, USING_LIBSUMO

from src.controllers.base import BaseController, ControllerResult
from src.environment.metrics import MetricsCollector


class SumoActuatedController(BaseController):
    """
    Runs the InTAS scenario with unmodified SUMO signal programs (a mix of
    vehicle-actuated and static tlLogic entries, per the network file).

    No TraCI signal commands are issued beyond the initial connection.
    SUMO executes its own programs from the .sumocfg / additional files.
    """

    def __init__(
        self,
        step_length: float = 1.0,
        episode_duration: int = 3600,
        sumo_binary: str = "sumo",
        sumo_extra_args: list[str] | None = None,
    ) -> None:
        self.step_length = step_length
        self.episode_duration = episode_duration
        self.sumo_binary = sumo_binary
        self.sumo_extra_args = sumo_extra_args or []

    @property
    def name(self) -> str:
        return "sumo_actuated"

    def run_episode(
        self,
        sumo_cfg: str,
        tls_metadata_json: str,
        seed: int = 42,
        demand_id: str = "1.0x",
        trajectory_path: str | None = None,
    ) -> ControllerResult:
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
            traci.start(cmd, label=f"sumo_actuated_{seed}")

        collector = MetricsCollector()
        collector.reset("sumo_actuated", seed, demand_id)
        tls_ids = list(traci.trafficlight.getIDList())

        trajectory_rows = []
        step = 0
        try:
            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                sim_time = traci.simulation.getTime()
                # Record metrics every 5 seconds (decision_interval)
                if step % 5 == 0:
                    collector.record_step(step, sim_time, 0.0, 0)
                    if trajectory_path:
                        for vid in traci.vehicle.getIDList():
                            x, y = traci.vehicle.getPosition(vid)
                            spd = traci.vehicle.getSpeed(vid)
                            trajectory_rows.append({
                                "time": sim_time, "vehicle_id": vid, "x": x, "y": y, "speed": spd,
                                "tls_id": None, "phase": None, "tls_state": None,
                            })
                        # SUMO owns these signal programs directly, so
                        # traci's own phase/state is authoritative here.
                        for tid in tls_ids:
                            try:
                                phase = traci.trafficlight.getPhase(tid)
                                state_str = traci.trafficlight.getRedYellowGreenState(tid)
                            except traci.exceptions.TraCIException:
                                continue
                            trajectory_rows.append({
                                "time": sim_time, "vehicle_id": None, "x": None, "y": None, "speed": None,
                                "tls_id": tid, "phase": phase, "tls_state": state_str,
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
            controller="sumo_actuated",
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
            total_switches=0,
        )
