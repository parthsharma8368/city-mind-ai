"""
local_reward.py
================
Per-TLS local reward for parameter-shared decentralized training.

The centralized environment's reward.py hands every one of the 98 joint
action dimensions the *same* network-wide diff-waiting-time signal — the
credit-assignment problem diagnosed from the 2026-09-18 training run
(rollout/ep_rew_mean flat for 500k steps, entropy never leaving ~max). This
gives each intersection a reward computed only from its own controlled
lanes, so a policy update actually reflects that intersection's own decision.

refresh_lane_waiting() does one pass over every vehicle in the network to
build a lane -> summed-waiting-time map; call it once per decision step
(shared across all 98 agents) rather than querying per agent, which would be
O(agents x vehicles) instead of O(vehicles).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.environment.sumo_connection import traci
from src.scenario.tls_metadata import TLSSpec


@dataclass
class LocalRewardConfig:
    waiting_delta_weight: float = 1.0
    queue_weight: float = 0.0
    switching_penalty: float = 0.3
    reward_clip: float | None = 50.0  # per-agent rewards are much smaller-scale than the network-wide sum


class LocalRewardCalculator:
    def __init__(self, config: LocalRewardConfig | None = None) -> None:
        self.cfg = config or LocalRewardConfig()
        self._lane_waiting: dict[str, float] = {}
        self._prev_tls_waiting: dict[str, float] = {}

    def reset(self) -> None:
        self._lane_waiting = {}
        self._prev_tls_waiting = {}

    def refresh_lane_waiting(self) -> None:
        """One pass over all vehicles, building lane_id -> summed waiting time."""
        lane_waiting: dict[str, float] = {}
        try:
            for vid in traci.vehicle.getIDList():
                lane_id = traci.vehicle.getLaneID(vid)
                wait = traci.vehicle.getWaitingTime(vid)
                lane_waiting[lane_id] = lane_waiting.get(lane_id, 0.0) + wait
        except traci.exceptions.TraCIException:
            pass
        self._lane_waiting = lane_waiting

    def compute(
        self,
        tls_id: str,
        spec: TLSSpec,
        queue_sum: float,
        switched: bool,
    ) -> tuple[float, dict]:
        current_waiting = sum(self._lane_waiting.get(lane, 0.0) for lane in spec.controlled_lanes)
        prev_waiting = self._prev_tls_waiting.get(tls_id, current_waiting)

        delta_waiting = current_waiting - prev_waiting
        r_waiting = -self.cfg.waiting_delta_weight * delta_waiting
        r_queue = -self.cfg.queue_weight * queue_sum
        r_switch = -self.cfg.switching_penalty * (1.0 if switched else 0.0)
        reward = r_waiting + r_queue + r_switch

        if self.cfg.reward_clip is not None:
            reward = max(-abs(self.cfg.reward_clip), min(abs(self.cfg.reward_clip), reward))

        self._prev_tls_waiting[tls_id] = current_waiting

        info = {
            "reward/waiting_component": r_waiting,
            "reward/queue_component": r_queue,
            "reward/switch_component": r_switch,
            "metrics/local_waiting": current_waiting,
        }
        return float(reward), info
