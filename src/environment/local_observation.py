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
building all 98 agents' local observations costs zero extra TraCI calls
beyond the downstream-lane queue lookups already required for
downstream_queue_norm (unchanged by the neighbor features below — those come
from OTHER agents' already-computed own-queue values, not new TraCI calls).

XAI evidence (2026-09-20 IPPO analysis) found a sampled junction's decision
was 94.1% driven by its own state and 0.1% by the single aggregate
downstream_queue_norm below — consistent with the network's 3.02s vs 4.15s
gap to Fixed-Time being a *missing-information* problem, not an
architecture problem. neighbor_queue_norm_1/2 give each agent explicit
visibility into its 2 geographically-nearest other TLS's own current
congestion, which downstream_queue_norm (this TLS's own outflow lanes, not
another TLS's state) never provided.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.environment.sumo_connection import traci
from src.scenario.tls_metadata import MAX_GREEN_PHASES, TLSRuntimeState, TLSSpec

# phase_elapsed_norm, phase_progress_norm, mean_queue_norm, max_queue_norm,
# mean_speed_norm, mean_density_norm, num_lanes_norm, downstream_queue_norm,
# neighbor_queue_norm_1, neighbor_queue_norm_2, total_vehicles_norm,
# total_waiting_norm, phase_pressure_norm_0..5 (RL-strategy Phase 2:
# phase-selection needs per-phase pressure, not just the aggregate
# intersection-wide pressure below — see compute_per_phase_pressure)
LOCAL_OBS_SIZE = 12 + MAX_GREEN_PHASES
N_NEIGHBORS = 2


@dataclass
class LocalObservationConfig:
    max_lane_vehicles: float = 50.0
    max_speed: float = 13.9         # ~50 km/h in m/s
    max_phase_elapsed: float = 120.0
    max_total_vehicles: float = 2000.0
    max_waiting_time: float = 3600.0
    max_lanes_reference: float = 20.0  # normalizes "how big is this intersection"
    max_phase_pressure: float = 30.0   # normalizes phase_pressure_norm_* (signed, clipped to [-1, 1])


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


def compute_mean_queue_norm(state: TLSRuntimeState, cfg: LocalObservationConfig) -> float:
    """This TLS's own mean queue, normalized — the value neighbors read about each other."""
    queue = state.lane_queue or [0.0]
    return min(_mean(queue) / cfg.max_lane_vehicles, 1.0)


def compute_intersection_pressure(
    spec: TLSSpec, state: TLSRuntimeState, downstream_cache: dict[str, list[str]]
) -> float:
    """Total Varaiya pressure for this intersection right now: this TLS's
    own total queue minus the total queue on the lanes it feeds into,
    summed over every controlled lane — the standard MPLight/PressLight
    per-step reward signal (max_pressure.py's _phase_pressure is the
    discrete phase-choice variant of the same idea, computed only over one
    candidate phase's movements rather than the whole intersection).

    Added 2026-09-21 (RL-strategy Phase 1) as an ablation arm against the
    existing waiting-delta reward — see LocalRewardConfig.pressure_weight.
    Positive pressure means this TLS's own approaches are more congested
    than what they feed into (the ordinary case: serve more). Large
    negative pressure means the opposite — this TLS is pushing traffic
    into lanes more congested than its own (spillback) — so the reward
    penalizes |pressure| in either direction, not just positive pressure:
    a good policy keeps upstream and downstream roughly balanced.

    NOTE: re-walks the same downstream lanes compute_downstream_queue_sum
    visits for the observation's downstream_queue_norm feature (a second
    TraCI query pass) — get_downstream_lanes's per-lane cache still avoids
    re-querying the STRUCTURE (traci.lane.getLinks), only the per-step
    occupancy values are queried twice. Acceptable for now; a lane-
    subscription rewrite (noted in the RL-strategy plan) would remove
    this."""
    upstream = float(sum(state.lane_queue)) if state.lane_queue else 0.0
    downstream = 0.0
    for lane_id in spec.controlled_lanes:
        for dl in get_downstream_lanes(lane_id, downstream_cache):
            try:
                downstream += float(traci.lane.getLastStepHaltingNumber(dl))
            except traci.exceptions.TraCIException:
                pass
    return upstream - downstream


def compute_per_phase_pressure(
    spec: TLSSpec, downstream_cache: dict[str, list[str]]
) -> list[float]:
    """Varaiya pressure (upstream halting - downstream halting, summed over
    that phase's own green movements) for each of this TLS's own green
    phases, in green_phase_indices order — position i is what selecting
    action i would relieve right now. Same per-movement computation as
    max_pressure.py's _phase_pressure (kept independent rather than shared,
    since MaxPressureController computes it for at most 2 candidate phases
    per step while this needs every phase, every step, for every agent).

    Unlike compute_intersection_pressure (which sums over ALL controlled
    lanes regardless of which phase is active), this is what actually makes
    phase-selection learnable: the aggregate can't tell the policy WHICH of
    its available phases would help, only that something is imbalanced.
    """
    pressures: list[float] = []
    for idx in spec.green_phase_indices:
        phase = spec.phases[idx]
        p = 0.0
        for i, sig in enumerate(phase.state):
            if sig not in ("G", "g") or i >= len(spec.controlled_lanes):
                continue
            lane_id = spec.controlled_lanes[i]
            try:
                upstream = traci.lane.getLastStepHaltingNumber(lane_id)
            except traci.exceptions.TraCIException:
                upstream = 0.0
            downstream = 0.0
            for dl in get_downstream_lanes(lane_id, downstream_cache):
                try:
                    downstream += traci.lane.getLastStepHaltingNumber(dl)
                except traci.exceptions.TraCIException:
                    pass
            p += upstream - downstream
        pressures.append(p)
    return pressures


def compute_downstream_queue_sum(
    spec: TLSSpec, downstream_cache: dict[str, list[str]]
) -> float:
    """Raw (non-normalized) sum-basis for this TLS's own downstream-lane queues.

    Pulled out of build_local_observation so the env can compute it once per
    TLS per step and reuse the same TraCI calls for both the observation
    feature and the reward's spillback penalty, instead of querying twice.
    """
    downstream_queues: list[float] = []
    for lane_id in spec.controlled_lanes:
        for dl in get_downstream_lanes(lane_id, downstream_cache):
            try:
                downstream_queues.append(float(traci.lane.getLastStepHaltingNumber(dl)))
            except traci.exceptions.TraCIException:
                pass
    return _mean(downstream_queues)


def build_local_observation(
    spec: TLSSpec,
    state: TLSRuntimeState,
    cfg: LocalObservationConfig,
    downstream_cache: dict[str, list[str]],
    total_vehicles: float,
    total_waiting: float,
    downstream_queue_sum: float | None = None,
    neighbor_queue_norms: list[float] | None = None,
    phase_pressures: list[float] | None = None,
) -> np.ndarray:
    """
    Build this one TLS's fixed-size local observation.

    Requires state.lane_queue / lane_vehicles / lane_speed / lane_density to
    already be populated with this step's *raw* values (ObservationBuilder.build()
    does this for every TLS in one pass before any agent's local obs is built).

    downstream_queue_sum: this TLS's own raw downstream-lane queue sum,
    precomputed once by the caller (compute_downstream_queue_sum) — computed
    internally here if omitted, for callers (e.g. the unit tests) that don't
    need to share it with the reward calculator too.

    neighbor_queue_norms: the 2 geographically-nearest other TLS's own
    compute_mean_queue_norm() values, precomputed by the caller (which has to
    already know every TLS's position and mean queue to build this list —
    see DecentralizedSumoEnv._neighbor_map). Padded/truncated to exactly
    N_NEIGHBORS; defaults to zeros when omitted (e.g. no neighbor data
    available, or fewer than N_NEIGHBORS TLS exist).

    phase_pressures: this TLS's own compute_per_phase_pressure() values (one
    per green phase, in green_phase_indices/action-index order), precomputed
    by the caller so it can be shared with the phase-selection action mask.
    Padded to MAX_GREEN_PHASES with 0.0 for TLS with fewer green phases —
    those slots are also masked out of the action space, so their pressure
    value is never actable on; 0.0 (neutral/balanced) rather than some
    extreme value keeps an unmasked forward pass numerically harmless.
    """
    n_phases = len(spec.phases)
    phase_elapsed_norm = min(state.phase_elapsed / cfg.max_phase_elapsed, 1.0)
    phase_progress_norm = (state.current_phase / (n_phases - 1)) if n_phases > 1 else 0.0

    queue = state.lane_queue or [0.0]
    speed = state.lane_speed or [0.0]
    density = state.lane_density or [0.0]

    mean_queue_norm = compute_mean_queue_norm(state, cfg)
    max_queue_norm = min(max(queue) / cfg.max_lane_vehicles, 1.0)
    mean_speed_norm = min(max(_mean(speed), 0.0) / cfg.max_speed, 1.0)
    mean_density_norm = min(_mean(density), 1.0)
    num_lanes_norm = min(len(spec.controlled_lanes) / cfg.max_lanes_reference, 1.0)

    if downstream_queue_sum is None:
        downstream_queue_sum = compute_downstream_queue_sum(spec, downstream_cache)
    downstream_queue_norm = min(downstream_queue_sum / cfg.max_lane_vehicles, 1.0)

    neighbors = list(neighbor_queue_norms or [])
    neighbors = (neighbors + [0.0] * N_NEIGHBORS)[:N_NEIGHBORS]

    total_vehicles_norm = min(total_vehicles / cfg.max_total_vehicles, 1.0)
    total_waiting_norm = min(total_waiting / cfg.max_waiting_time, 1.0)

    pressures = list(phase_pressures or [])
    pressures = (pressures + [0.0] * MAX_GREEN_PHASES)[:MAX_GREEN_PHASES]
    pressures_norm = [max(-1.0, min(1.0, p / cfg.max_phase_pressure)) for p in pressures]

    return np.array([
        phase_elapsed_norm,
        phase_progress_norm,
        mean_queue_norm,
        max_queue_norm,
        mean_speed_norm,
        mean_density_norm,
        num_lanes_norm,
        downstream_queue_norm,
        *neighbors,
        total_vehicles_norm,
        total_waiting_norm,
        *pressures_norm,
    ], dtype=np.float32)
