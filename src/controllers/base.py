"""
base.py
=======
Abstract base class for all controllers (fixed_time, max_pressure, RL).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


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
