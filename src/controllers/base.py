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
