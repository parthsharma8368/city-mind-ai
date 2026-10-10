"""
metrics.py
==========
Episode-level metrics collector.

Collects per-step data and computes episode summaries.
All metrics use the same SUMO TraCI values regardless of which controller is active
(fair comparison protocol).

ONE DEFINITION (metrics_schema_version 2 -- see src/environment/schema.py):
queue/speed are computed by `compute_network_queue_speed()` for EVERY
controller and environment. Before v2 the RL environments averaged per-lane
mean speeds (empty lanes report free-flow speed, inflating it ~19%) over a
per-link-duplicated lane list while the baselines used per-vehicle speeds and
unique lanes, so RL-vs-baseline mean_speed/mean_queue were not comparable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from src.environment.schema import METRICS_SCHEMA_VERSION
from src.environment.sumo_connection import traci

log = logging.getLogger(__name__)


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
    total_teleports: int = 0
    backlog: int = 0           # vehicles still in the network when the episode ended

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

    # Provenance: bump-able definition version + the exact SUMO argv used.
    metrics_schema_version: int = METRICS_SCHEMA_VERSION
    sumo_command: list[str] = field(default_factory=list)

    # Per-step records (not exported to summary, used for Parquet)
    step_records: list[StepRecord] = field(default_factory=list, repr=False)


def unique_controlled_lanes() -> list[str]:
    """De-duplicated, sorted incoming lanes of every traffic light -- the ONE
    lane set queue metrics are defined over (falls back to all lanes when the
    network has no TLS)."""
    lanes: set[str] = set()
    for tid in traci.trafficlight.getIDList():
        lanes.update(traci.trafficlight.getControlledLanes(tid))
    if not lanes:
        lanes.update(traci.lane.getIDList())
    return sorted(lanes)


def compute_network_queue_speed(
    tracked_lanes: list[str], vehicle_ids=None
) -> tuple[float, float, float, float]:
    """THE shared queue/speed definition for every controller and env.

    Returns (total_queue, mean_queue, max_queue, mean_speed):
      * queue values are traci.lane.getLastStepHaltingNumber over the UNIQUE
        tracked lanes (total = sum, mean/max over lanes);
      * mean_speed is the mean of traci.vehicle.getSpeed over every vehicle in
        the network (per-vehicle, so empty lanes cannot inflate it).
    """
    lane_queues: list[float] = []
    for lid in tracked_lanes:
        try:
            lane_queues.append(float(traci.lane.getLastStepHaltingNumber(lid)))
        except traci.exceptions.TraCIException:
            pass  # lane vanished / not loaded: skip it, like the old baseline path
    total_q = float(sum(lane_queues))
    avg_q = float(np.mean(lane_queues)) if lane_queues else 0.0
    max_q = float(max(lane_queues)) if lane_queues else 0.0

    ids = vehicle_ids if vehicle_ids is not None else traci.vehicle.getIDList()
    speeds = [traci.vehicle.getSpeed(v) for v in ids]
    avg_spd = float(np.mean(speeds)) if speeds else 0.0
    return total_q, avg_q, max_q, avg_spd


def is_green_state(state_str: str) -> bool:
    """Same rule as PhaseSpec.is_green, on a raw RYG state string."""
    low = state_str.lower()
    return "g" in low and "y" not in low


class GreenSwitchCounter:
    """Counts LOGICAL switches (the displayed stable green state changing to a
    different green state) from raw per-step RYG strings, so SUMO-native
    baselines get a switch count on the same footing as the safety-layer
    controllers (rl-core-010; they used to hard-code 0)."""

    def __init__(self) -> None:
        self._last_green: dict[str, str] = {}
        self.total = 0

    def update(self, tls_id: str, state_str: str) -> bool:
        if not is_green_state(state_str):
            return False
        prev = self._last_green.get(tls_id)
        self._last_green[tls_id] = state_str
        if prev is not None and prev != state_str:
            self.total += 1
            return True
        return False


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
        self._pending_departed: int = 0
        self._pending_arrived: int = 0
        self._pending_teleports: int = 0
        self._backlog: int = 0
        self._sumo_command: list[str] = []

    def reset(
        self, controller: str, seed: int, demand_id: str, sumo_command: list[str] | None = None
    ) -> None:
        self._sumo_command = list(sumo_command or [])
        self._records = []
        self._controller = controller
        self._seed = seed
        self._demand_id = demand_id
        self._tracked_lanes = None
        self._pending_departed = 0
        self._pending_arrived = 0
        self._pending_teleports = 0
        self._backlog = 0
        try:
            self._departed_start = traci.simulation.getDepartedNumber()
        except Exception:
            self._departed_start = 0

    def tick(self) -> None:
        """Call once immediately after every traci.simulationStep().

        getDepartedNumber()/getArrivedNumber()/getStartingTeleportNumber()
        report counts SINCE THE LAST CALL, not cumulative totals. Callers
        that only invoke record_step() every N raw steps (the decision
        interval) but call simulationStep() every raw step were dropping
        (N-1)/N of all departures and arrivals — this accumulator makes
        counting correct regardless of how often record_step() is called.

        Only the expected TraCIException is swallowed (and logged); a dead
        SUMO (FatalTraCIError) now propagates instead of silently zeroing
        counts (rl-core-013b).
        """
        try:
            dep = traci.simulation.getDepartedNumber()
            arr = traci.simulation.getArrivedNumber()
            tel = traci.simulation.getStartingTeleportNumber()
        except traci.exceptions.TraCIException as exc:
            log.warning("MetricsCollector.tick(): TraCI read failed, counts for this step lost: %r", exc)
            return
        self._pending_departed += dep
        self._pending_arrived += arr
        self._pending_teleports += tel

    def record_backlog(self) -> None:
        """Call once, right before closing SUMO. backlog = vehicles in the
        network + vehicles waiting for insertion (pending). Independent of the
        config's --route-steps look-ahead (the old getMinExpectedNumber counted
        every vehicle already LOADED for the future, which depends on it)."""
        try:
            backlog = len(traci.vehicle.getIDList())
            try:
                backlog += len(traci.simulation.getPendingVehicles())
            except traci.exceptions.TraCIException:
                pass
            self._backlog = backlog
        except traci.exceptions.TraCIException as exc:
            log.warning("MetricsCollector.record_backlog(): %r", exc)
            self._backlog = 0

    def flush(self) -> None:
        """Fold counters accumulated after the last record_step() (a final
        partial interval) into the last record so nothing is dropped."""
        if self._records and (self._pending_departed or self._pending_arrived):
            last = self._records[-1]
            last.departed_vehicles += self._pending_departed
            last.arrived_vehicles += self._pending_arrived
            self._pending_departed = 0
            self._pending_arrived = 0

    def record_step(
        self,
        step: int,
        sim_time: float,
        reward: float,
        switch_count: int,
        cached_metrics: dict | None = None,
        states: dict | None = None,
    ) -> StepRecord:
        """Read TraCI values (or reuse cached values) and create a step record.

        ``states`` is accepted for backwards compatibility and IGNORED: queue
        and speed always come from compute_network_queue_speed() so every
        controller reports the same definition (eval-scripts-2).
        """
        try:
            if cached_metrics is not None:
                n_vehicles = int(cached_metrics.get("total_vehicles", 0))
                total_wait = float(cached_metrics.get("total_waiting", 0.0))
                avg_wait = float(cached_metrics.get("avg_waiting", 0.0))
                vehicle_ids = cached_metrics.get("vehicle_ids") if "vehicle_ids" in cached_metrics else None
            else:
                vehicle_ids = traci.vehicle.getIDList()
                n_vehicles = len(vehicle_ids)
                waiting_times = [traci.vehicle.getWaitingTime(v) for v in vehicle_ids]
                total_wait = sum(waiting_times)
                avg_wait = float(np.mean(waiting_times)) if waiting_times else 0.0

            if self._tracked_lanes is None:
                self._tracked_lanes = unique_controlled_lanes()
            total_q, avg_q, max_q, avg_spd = compute_network_queue_speed(
                self._tracked_lanes, vehicle_ids
            )
            # Drain the tick() accumulator only AFTER every read succeeded, so
            # a TraCIException here cannot zero the pending counts.
            departed, arrived = self._pending_departed, self._pending_arrived
            self._pending_departed = 0
            self._pending_arrived = 0
        except traci.exceptions.TraCIException as exc:
            log.warning("MetricsCollector.record_step(): TraCI read failed at t=%s: %r", sim_time, exc)
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
        self.flush()
        if not self._records:
            return EpisodeSummary(
                controller=self._controller,
                seed=self._seed,
                demand_id=self._demand_id,
                episode_steps=0,
                episode_duration=0.0,
                sumo_command=list(self._sumo_command),
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
            total_teleports=self._pending_teleports,
            backlog=self._backlog,
            mean_waiting_time=float(np.mean([r.average_waiting_time for r in recs])),
            total_waiting_time_sum=float(sum(r.total_waiting_time for r in recs)),
            mean_queue=float(np.mean([r.average_queue for r in recs])),
            max_queue=float(max(r.max_queue for r in recs)),
            mean_speed=float(np.mean([r.average_speed for r in recs])),
            total_switches=sum(r.signal_switch_count for r in recs),
            cumulative_reward=sum(r.reward for r in recs),
            sumo_command=list(self._sumo_command),
            step_records=recs,
        )
        return summary

    def to_dataframe(self):
        """Convert step records to a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([vars(r) for r in self._records])
