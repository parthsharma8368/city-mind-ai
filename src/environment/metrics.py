"""
metrics.py
==========
Episode-level metrics collector.

Collects per-step data and computes episode summaries.
All metrics use the same SUMO TraCI values regardless of which controller is active
(fair comparison protocol).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

from src.environment.sumo_connection import traci


@dataclass
class StepRecord:
    """One row of per-step metrics."""
    step: int
    sim_time: float
    total_vehicles: int
    departed_vehicles: int
    arrived_vehicles: int
    total_waiting_time: float
    average_waiting_time: float
    total_queue: float
    average_queue: float
    max_queue: float
    average_speed: float
    signal_switch_count: int
    reward: float


@dataclass
class EpisodeSummary:
    """Aggregated episode-level summary."""
    controller: str
    seed: int
    demand_id: str
    episode_steps: int
    episode_duration: float   # simulated seconds

    # Throughput
    total_departed: int = 0
    total_arrived: int = 0
    throughput: float = 0.0    # arrived / departed

    # Waiting
    mean_waiting_time: float = 0.0
    total_waiting_time_sum: float = 0.0

    # Queue
    mean_queue: float = 0.0
    max_queue: float = 0.0

    # Speed
    mean_speed: float = 0.0

    # Signal
    total_switches: int = 0

    # Reward
    cumulative_reward: float = 0.0

    # Per-step records (not exported to summary, used for Parquet)
    step_records: list[StepRecord] = field(default_factory=list, repr=False)


class MetricsCollector:
    """
    Collects per-step metrics during an episode and emits a summary.

    Usage:
      collector.reset(controller, seed, demand_id)
      for each step:
          collector.record_step(step, sim_time, reward, switch_count)
      summary = collector.summarize()
    """

    def __init__(self) -> None:
        self._records: list[StepRecord] = []
        self._controller: str = ""
        self._seed: int = 0
        self._demand_id: str = ""
        self._departed_start: int = 0
        self._tracked_lanes: list[str] | None = None

    def reset(self, controller: str, seed: int, demand_id: str) -> None:
        self._records = []
        self._controller = controller
        self._seed = seed
        self._demand_id = demand_id
        self._tracked_lanes = None
        try:
            self._departed_start = traci.simulation.getDepartedNumber()
        except Exception:
            self._departed_start = 0

    def record_step(
        self,
        step: int,
        sim_time: float,
        reward: float,
        switch_count: int,
        cached_metrics: dict | None = None,
        states: dict | None = None,
    ) -> StepRecord:
        """Read TraCI values (or reuse cached values) and create a step record."""
        try:
            departed = traci.simulation.getDepartedNumber()
            arrived = traci.simulation.getArrivedNumber()

            if cached_metrics is not None:
                n_vehicles = int(cached_metrics.get("total_vehicles", 0))
                total_wait = float(cached_metrics.get("total_waiting", 0.0))
                avg_wait = float(cached_metrics.get("avg_waiting", 0.0))
            else:
                vehicle_ids = traci.vehicle.getIDList()
                n_vehicles = len(vehicle_ids)
                waiting_times = [traci.vehicle.getWaitingTime(v) for v in vehicle_ids]
                total_wait = sum(waiting_times)
                avg_wait = float(np.mean(waiting_times)) if waiting_times else 0.0

            if states is not None:
                all_queues = [q for s in states.values() for q in s.lane_queue]
                total_q = float(sum(all_queues))
                avg_q = float(np.mean(all_queues)) if all_queues else 0.0
                max_q = float(max(all_queues)) if all_queues else 0.0

                all_speeds = [spd for s in states.values() for spd in s.lane_speed if spd >= 0]
                avg_spd = float(np.mean(all_speeds)) if all_speeds else 0.0
            else:
                lane_queues = []
                if self._tracked_lanes is None:
                    try:
                        lanes = set()
                        for tid in traci.trafficlight.getIDList():
                            lanes.update(traci.trafficlight.getControlledLanes(tid))
                        self._tracked_lanes = list(lanes) if lanes else list(traci.lane.getIDList())
                    except Exception:
                        self._tracked_lanes = []

                for lid in self._tracked_lanes:
                    try:
                        lane_queues.append(traci.lane.getLastStepHaltingNumber(lid))
                    except Exception:
                        pass
                total_q = float(sum(lane_queues))
                avg_q = float(np.mean(lane_queues)) if lane_queues else 0.0
                max_q = float(max(lane_queues)) if lane_queues else 0.0

                vehicle_ids = (
                    cached_metrics.get("vehicle_ids", [])
                    if cached_metrics and "vehicle_ids" in cached_metrics
                    else traci.vehicle.getIDList()
                )
                speeds = [traci.vehicle.getSpeed(v) for v in vehicle_ids]
                avg_spd = float(np.mean(speeds)) if speeds else 0.0

        except traci.exceptions.TraCIException:
            n_vehicles, departed, arrived = 0, 0, 0
            total_wait, avg_wait = 0.0, 0.0
            total_q, avg_q, max_q = 0.0, 0.0, 0.0
            avg_spd = 0.0

        rec = StepRecord(
            step=step,
            sim_time=sim_time,
            total_vehicles=n_vehicles,
            departed_vehicles=departed,
            arrived_vehicles=arrived,
            total_waiting_time=total_wait,
            average_waiting_time=avg_wait,
            total_queue=total_q,
            average_queue=avg_q,
            max_queue=max_q,
            average_speed=avg_spd,
            signal_switch_count=switch_count,
            reward=reward,
        )
        self._records.append(rec)
        return rec

    def summarize(self) -> EpisodeSummary:
        """Aggregate per-step records into an EpisodeSummary."""
        if not self._records:
            return EpisodeSummary(
                controller=self._controller,
                seed=self._seed,
                demand_id=self._demand_id,
                episode_steps=0,
                episode_duration=0.0,
            )

        recs = self._records
        n = len(recs)

        # Throughput
        total_dep = sum(r.departed_vehicles for r in recs)
        total_arr = sum(r.arrived_vehicles for r in recs)
        throughput = total_arr / max(total_dep, 1)

        summary = EpisodeSummary(
            controller=self._controller,
            seed=self._seed,
            demand_id=self._demand_id,
            episode_steps=n,
            episode_duration=recs[-1].sim_time - recs[0].sim_time if n > 1 else 0.0,
            total_departed=total_dep,
            total_arrived=total_arr,
            throughput=throughput,
            mean_waiting_time=float(np.mean([r.average_waiting_time for r in recs])),
            total_waiting_time_sum=float(sum(r.total_waiting_time for r in recs)),
            mean_queue=float(np.mean([r.average_queue for r in recs])),
            max_queue=float(max(r.max_queue for r in recs)),
            mean_speed=float(np.mean([r.average_speed for r in recs])),
            total_switches=sum(r.signal_switch_count for r in recs),
            cumulative_reward=sum(r.reward for r in recs),
            step_records=recs,
        )
        return summary

    def to_dataframe(self):
        """Convert step records to a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([vars(r) for r in self._records])
