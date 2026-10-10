"""
base.py
=======
Abstract base class for all controllers (fixed_time, max_pressure, RL).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.environment.schema import METRICS_SCHEMA_VERSION
from src.environment.sumo_connection import traci


@dataclass
class ControllerResult:
    """Outcome of running one episode with a controller."""
    controller: str
    seed: int
    demand_id: str
    total_steps: int
    episode_duration: float  # simulated seconds
    # Metrics
    mean_waiting_time: float = 0.0
    total_waiting_time: float = 0.0
    mean_queue: float = 0.0
    max_queue: float = 0.0
    throughput: float = 0.0
    total_arrived: int = 0
    total_departed: int = 0
    mean_speed: float = 0.0
    total_switches: int = 0
    cumulative_reward: float = 0.0
    source_run_id: str = ""  # which training run's checkpoint produced this (RL controllers only)
    # Robustness / trip-level metrics (added alongside the per-step
    # undercounting fix — see metrics.py's tick()/record_backlog()).
    total_teleports: int = 0
    backlog: int = 0             # vehicles still in-network when the episode ended
    trips_completed: int = 0     # from --tripinfo-output; independent of in-network sampling bias
    mean_trip_duration: float = 0.0
    mean_trip_timeloss: float = 0.0
    # Provenance (rl-core-004 / eval-scripts-2): definition version of the
    # metrics above (results written before this field are version 1 and NOT
    # comparable for mean_speed/mean_queue/max_queue/backlog/total_switches)
    # and the exact SUMO argv that produced the numbers.
    metrics_schema_version: int = METRICS_SCHEMA_VERSION
    sumo_command: list[str] = field(default_factory=list)


def result_from_summary(
    controller: str,
    summary,
    *,
    seed: int,
    demand_id: str,
    total_switches: int,
    trip_summary=None,
    sumo_command: list[str] | None = None,
) -> ControllerResult:
    """Build a ControllerResult from a MetricsCollector EpisodeSummary -- the
    single mapping every baseline uses."""
    return ControllerResult(
        controller=controller,
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
        cumulative_reward=summary.cumulative_reward,
        total_teleports=summary.total_teleports,
        backlog=summary.backlog,
        trips_completed=trip_summary.trips_completed if trip_summary else 0,
        mean_trip_duration=trip_summary.mean_trip_duration if trip_summary else 0.0,
        mean_trip_timeloss=trip_summary.mean_trip_timeloss if trip_summary else 0.0,
        metrics_schema_version=summary.metrics_schema_version,
        sumo_command=list(sumo_command if sumo_command is not None else summary.sumo_command),
    )


class BaseController(ABC):
    """Interface for all traffic signal controllers."""

    @abstractmethod
    def run_episode(
        self,
        sumo_cfg: str,
        tls_metadata_json: str,
        seed: int = 42,
        demand_id: str = "1.0x",
    ) -> ControllerResult:
        """Run one full episode and return metrics."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Controller identifier string."""
        ...


def is_sample_step(step: int, interval: int) -> bool:
    """True after every ``interval``-th simulationStep (0-based ``step`` is
    the index of the step just executed). ALL controllers and the RL envs use
    this grid -- records at t = begin+interval, begin+2*interval, ... -- so
    their records are sampled at identical simulation times (eval-scripts-14:
    max_pressure used to be offset by one step, the others by four)."""
    return (step + 1) % interval == 0


def append_vehicle_rows(rows: list[dict], sim_time: float) -> None:
    """Append one trajectory row per vehicle currently in the network."""
    for vid in traci.vehicle.getIDList():
        x, y = traci.vehicle.getPosition(vid)
        rows.append({
            "time": sim_time, "vehicle_id": vid, "x": x, "y": y, "speed": traci.vehicle.getSpeed(vid),
            "tls_id": None, "phase": None, "tls_state": None,
        })


def append_tls_row(rows: list[dict], sim_time: float, tls_id: str, phase: int, state_str: str) -> None:
    rows.append({
        "time": sim_time, "vehicle_id": None, "x": None, "y": None, "speed": None,
        "tls_id": tls_id, "phase": phase, "tls_state": state_str,
    })
