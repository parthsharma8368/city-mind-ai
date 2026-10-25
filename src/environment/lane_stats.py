"""
lane_stats.py
=============
Once per decision step: read the halted-car count of every incoming lane of every signal (into
``TLSRuntimeState.lane_queue``, aligned with ``TLSSpec.controlled_lanes``) and one network-wide pass over the vehicles
(``last_step_cache``: ids, waiting times, totals) that the metrics collector reuses instead of querying again.
"""
from __future__ import annotations

import numpy as np

from src.environment.sumo_connection import traci
from src.scenario.tls_metadata import TLSRuntimeState, TLSSpec


class LaneStatsReader:
    def __init__(self, tls_specs: list[TLSSpec]) -> None:
        self.specs = sorted(tls_specs, key=lambda s: s.tls_id)
        self.last_step_cache: dict = {}

    def read(self, states: dict[str, TLSRuntimeState]) -> None:
        halted: dict[str, float] = {}          # a lane that serves several links is read once
        for spec in self.specs:
            queue = []
            for lane in spec.controlled_lanes:
                if lane not in halted:
                    try:
                        halted[lane] = float(traci.lane.getLastStepHaltingNumber(lane))
                    except traci.exceptions.TraCIException:
                        halted[lane] = 0.0
                queue.append(halted[lane])
            states[spec.tls_id].lane_queue = queue

        vehicle_ids, waiting = [], []
        try:
            vehicle_ids = traci.vehicle.getIDList()
            waiting = [traci.vehicle.getWaitingTime(v) for v in vehicle_ids]
        except traci.exceptions.TraCIException:
            pass
        self.last_step_cache = {
            "vehicle_ids": vehicle_ids,
            "waiting_times": waiting,
            "total_vehicles": float(len(vehicle_ids)),
            "total_waiting": float(sum(waiting)),
            "avg_waiting": float(np.mean(waiting)) if waiting else 0.0,
        }
