"""
reward.py
=========
Network-level reward computation.

Primary formulation (delta waiting time):
  R_t = -(W_t - W_{t-1})

where W_t is total network waiting time at step t.
Positive reward when waiting time decreases; negative when it grows.

Extended formulation (opt-in via config):
  R_t = -alpha * (W_t - W_{t-1})
       - beta  * Q_t
       - gamma * S_t

where:
  W_t = total network waiting time (sum over all vehicles)
  Q_t = total network queue (sum of halting vehicles)
  S_t = number of phase switches this step (switching penalty)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from src.environment.sumo_connection import traci

from src.scenario.tls_metadata import TLSRuntimeState


@dataclass
class RewardConfig:
    waiting_delta_weight: float = 1.0
    queue_weight: float = 0.0
    switching_penalty: float = 0.0
    # Clipping to prevent exploding rewards during initial training
    reward_clip: float | None = None   # e.g. -100.0 to 100.0


class RewardCalculator:
    """
    Stateful reward calculator — tracks previous step metrics to compute deltas.
    Must be reset at the start of each episode.
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.cfg = config or RewardConfig()
        self._prev_waiting_time: float = 0.0
        self._prev_queue: float = 0.0
        self._switch_count: int = 0   # set externally by safety layer each step

    def reset(self) -> None:
        self._prev_waiting_time = 0.0
        self._prev_queue = 0.0
        self._switch_count = 0

    def set_switch_count(self, count: int) -> None:
        """Called by environment after safety layer step."""
        self._switch_count = count

    def compute(
        self,
        states: dict[str, TLSRuntimeState],
        cached_waiting_time: float | None = None,
        cached_vehicle_count: int | None = None,
    ) -> tuple[float, dict]:
        """
        Compute reward and return diagnostic info.

        Returns
        -------
        (reward, info_dict)
        """
        # ── Collect total waiting time from SUMO or cache ───────────────────
        if cached_waiting_time is not None:
            current_waiting = cached_waiting_time
            num_vehicles = cached_vehicle_count if cached_vehicle_count is not None else 0
        else:
            try:
                vehicle_ids = traci.vehicle.getIDList()
                num_vehicles = len(vehicle_ids)
                current_waiting = sum(
                    traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids
                )
            except traci.exceptions.TraCIException:
                current_waiting = self._prev_waiting_time
                num_vehicles = 0

        # ── Collect total queue (halting vehicles) ───────────────────────────
        current_queue = 0.0
        for state in states.values():
            current_queue += sum(state.lane_queue)  # already raw counts

        # ── Compute reward components ────────────────────────────────────────
        delta_waiting = current_waiting - self._prev_waiting_time
        r_waiting = -self.cfg.waiting_delta_weight * delta_waiting
        r_queue = -self.cfg.queue_weight * current_queue
        r_switch = -self.cfg.switching_penalty * self._switch_count

        reward = r_waiting + r_queue + r_switch

        if self.cfg.reward_clip is not None:
            reward = max(-abs(self.cfg.reward_clip), min(abs(self.cfg.reward_clip), reward))

        # ── Update state for next step ───────────────────────────────────────
        self._prev_waiting_time = current_waiting
        self._prev_queue = current_queue

        info = {
            "reward/total": reward,
            "reward/waiting_component": r_waiting,
            "reward/queue_component": r_queue,
            "reward/switch_component": r_switch,
            "metrics/total_waiting_time": current_waiting,
            "metrics/total_queue": current_queue,
            "metrics/switch_count": self._switch_count,
            "metrics/vehicle_count": num_vehicles,
        }
        return float(reward), info
