"""
fixed_time.py
=============
Fixed-time baseline controller.

This controller lets SUMO run its default/original traffic-light programs
without any external intervention. The RL policy does NOT act — SUMO's
built-in static programs execute as-is from the InTAS configuration.

This is the primary comparison baseline.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from src.environment.sumo_connection import traci, USING_LIBSUMO

import numpy as np

from src.controllers.base import BaseController, ControllerResult
from src.environment.metrics import MetricsCollector


class FixedTimeController(BaseController):
    """
    Runs the InTAS scenario with unmodified SUMO signal programs.

    No TraCI signal commands are issued. SUMO executes the static
    green-time programs from the .sumocfg / additional files.
    """

    def __init__(self, step_length: float = 1.0, episode_duration: int = 3600) -> None:
        self.step_length = step_length
        self.episode_duration = episode_duration

    @property
    def name(self) -> str:
        return "fixed_time"

    def run_episode(
        self,
        sumo_cfg: str,
        tls_metadata_json: str,
        seed: int = 42,
        demand_id: str = "1.0x",
        trajectory_path: str | None = None,
    ) -> ControllerResult:
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
            traci.start(cmd, label=f"fixed_time_{seed}")

        collector = MetricsCollector()
        collector.reset("fixed_time", seed, demand_id)

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
            controller="fixed_time",
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
