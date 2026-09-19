"""
local_observation.py
=====================
Fixed-size per-TLS observation for parameter-shared decentralized training.

A single shared policy network is invoked once per intersection per decision
step, so every intersection must produce an observation of the *same shape* —
but InTAS's 98 TLS range from 4 to 19 controlled lanes and 3 to 11 phases.
A one-hot-per-phase / one-value-per-lane vector (what the centralized
CentralizedSumoEnv observation uses) can't be shared across agents that
disagree on the vector's length.

This module trades per-lane resolution for aggregates (mean/max over an
intersection's own lanes, a normalized phase index instead of a one-hot) so
every agent's observation is the same fixed size regardless of its own
lane/phase count. It reads from TLSRuntimeState fields that ObservationBuilder
already populates each decision step (raw units — see observation.py), so
building all 98 agents' local observations costs zero extra TraCI calls.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.environment.sumo_connection import traci
from src.scenario.tls_metadata import TLSRuntimeState, TLSSpec

# phase_elapsed_norm, phase_progress_norm, mean_queue_norm, max_queue_norm,
# mean_speed_norm, mean_density_norm, num_lanes_norm, downstream_queue_norm,
# total_vehicles_norm, total_waiting_norm
LOCAL_OBS_SIZE = 10


@dataclass
class LocalObservationConfig:
    max_lane_vehicles: float = 50.0
    max_speed: float = 13.9         # ~50 km/h in m/s
    max_phase_elapsed: float = 120.0
    max_total_vehicles: float = 2000.0
    max_waiting_time: float = 3600.0
    max_lanes_reference: float = 20.0  # normalizes "how big is this intersection"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def get_downstream_lanes(lane_id: str, cache: dict[str, list[str]]) -> list[str]:
    """Shared with max_pressure.py's notion of downstream lanes (traci.lane.getLinks)."""
    cached = cache.get(lane_id)
    if cached is not None:
        return cached
    try:
        links = traci.lane.getLinks(lane_id)
        downstream = [link[0] for link in links if link and link[0]]
    except traci.exceptions.TraCIException:
        downstream = []
    cache[lane_id] = downstream
    return downstream


def build_local_observation(
    spec: TLSSpec,
    state: TLSRuntimeState,
    cfg: LocalObservationConfig,
    downstream_cache: dict[str, list[str]],
    total_vehicles: float,
    total_waiting: float,
) -> np.ndarray:
    """
    Build this one TLS's fixed-size local observation.

    Requires state.lane_queue / lane_vehicles / lane_speed / lane_density to
    already be populated with this step's *raw* values (ObservationBuilder.build()
    does this for every TLS in one pass before any agent's local obs is built).
    """
    n_phases = len(spec.phases)
    phase_elapsed_norm = min(state.phase_elapsed / cfg.max_phase_elapsed, 1.0)
    phase_progress_norm = (state.current_phase / (n_phases - 1)) if n_phases > 1 else 0.0

    queue = state.lane_queue or [0.0]
    speed = state.lane_speed or [0.0]
    density = state.lane_density or [0.0]

    mean_queue_norm = min(_mean(queue) / cfg.max_lane_vehicles, 1.0)
    max_queue_norm = min(max(queue) / cfg.max_lane_vehicles, 1.0)
    mean_speed_norm = min(max(_mean(speed), 0.0) / cfg.max_speed, 1.0)
    mean_density_norm = min(_mean(density), 1.0)
    num_lanes_norm = min(len(spec.controlled_lanes) / cfg.max_lanes_reference, 1.0)

    downstream_queues: list[float] = []
    for lane_id in spec.controlled_lanes:
        for dl in get_downstream_lanes(lane_id, downstream_cache):
            try:
                downstream_queues.append(float(traci.lane.getLastStepHaltingNumber(dl)))
            except traci.exceptions.TraCIException:
                pass
    downstream_queue_norm = min(_mean(downstream_queues) / cfg.max_lane_vehicles, 1.0)

    total_vehicles_norm = min(total_vehicles / cfg.max_total_vehicles, 1.0)
    total_waiting_norm = min(total_waiting / cfg.max_waiting_time, 1.0)

    return np.array([
        phase_elapsed_norm,
        phase_progress_norm,
        mean_queue_norm,
        max_queue_norm,
        mean_speed_norm,
        mean_density_norm,
        num_lanes_norm,
        downstream_queue_norm,
        total_vehicles_norm,
        total_waiting_norm,
    ], dtype=np.float32)
