"""
local_observation.py
=====================
The agent's observation of ONE intersection: a fixed-size vector (22 numbers), the same shape for every intersection, so a
single shared policy can control all of them.

  0-5    active_slot_i         one-hot of the active green phase (slot i = this signal's i-th green phase)
  6      in_transition         1 while a yellow / all-red clearance is running
  7      elapsed_over_min      min(elapsed / min_green, 3) / 3          (time in the phase vs ITS OWN minimum)
  8      elapsed_over_max      min(elapsed / max_green, 1.5) / 1.5      (time in the phase vs ITS OWN maximum)
  9-14   slot_demand_i         d_i / (|d_i| + k)  in (-1, 1), d_i = demand of the movements green in slot i (padded: 0)
  15     max_queue             q / (q + k),  q = halted cars on the worst incoming lane (incl. unsignalised feeders)
  16     pending               n / (n + k),  n = cars waiting OUTSIDE the network whose route starts at this signal
  17     n_green               number of real green phases / 6
  18-19  upstream_queue_0/1    max queue of the 2 strongest road-connected UPSTREAM signals, q / (q + k)
  20-21  downstream_queue_0/1  max queue of the 2 strongest road-connected DOWNSTREAM signals, q / (q + k)

Phase demand (Varaiya pressure of the movements that are green in slot i):
    d_i = sum over the green movements of [ halted(lane) + halted(feeders of lane) (+ pending share) - halted(downstream lanes) ]
Feeders = lanes of the UNSIGNALISED roads that feed a short stop-line lane, walked upstream up to ``queue_reach_m`` metres and
stopping at the previous signal. Without them a 20 m stop-line lane (~2 cars per lane) hides a 60-car queue behind it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.environment.sumo_connection import traci
from src.scenario.tls_metadata import MAX_GREEN_PHASES, TLSRuntimeState, TLSSpec

N_ROAD_NEIGHBORS = 2
OBS_SIZE = 2 * MAX_GREEN_PHASES + 6 + 2 * N_ROAD_NEIGHBORS     # 6 + 3 + 6 + 3 = 18, + 4 neighbour queues = 22
OBS_NAMES = (
    [f"active_slot_{i}" for i in range(MAX_GREEN_PHASES)]
    + ["in_transition", "elapsed_over_min", "elapsed_over_max"]
    + [f"slot_demand_{i}" for i in range(MAX_GREEN_PHASES)]
    + ["max_queue", "pending", "n_green"]
    + [f"upstream_queue_{i}" for i in range(N_ROAD_NEIGHBORS)]
    + [f"downstream_queue_{i}" for i in range(N_ROAD_NEIGHBORS)]
)
assert len(OBS_NAMES) == OBS_SIZE


@dataclass
class LocalObservationConfig:
    demand_squash_k: float = 8.0       # d / (|d| + k): a demand of k cars maps to 0.5
    queue_squash_k: float = 8.0        # same squash for max_queue and the neighbour queues
    pending_squash_k: float = 20.0     # n / (n + k) for cars waiting outside the network
    queue_reach_m: float = 0.0         # > 0: count halted cars on unsignalised feeder roads this far upstream (m)
    # Source of the min/max green behind the two elapsed ratios: "safety" = the safety layer's bounds (fallback
    # min/max green for signals without an authored actuated range), "program" = the phase's own authored min / max
    # (== its duration for a fixed-time program).
    timing_source: str = "safety"
    # Add cars waiting outside the network to the demand of the stop-line lanes they will queue at.
    pending_in_demand: bool = False


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


def compute_per_phase_pressure(
    spec: TLSSpec,
    downstream_cache: dict[str, list[str]],
    feeders: dict[str, list[str]] | None = None,
    halting=None,
    extra_upstream: dict[str, float] | None = None,
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

    feeders (lane -> unsignalised upstream lanes, see compute_feeder_lanes) adds the cars halted behind a short
    stop-line lane to that lane's upstream term; halting (lane -> count, memoised per step) replaces the direct
    TraCI query so a lane that serves several phases is read once.
    """
    def _h(lane: str) -> float:
        if halting is not None:
            return halting(lane)
        try:
            return traci.lane.getLastStepHaltingNumber(lane)
        except traci.exceptions.TraCIException:
            return 0.0

    pressures: list[float] = []
    for idx in spec.green_phase_indices:
        phase = spec.phases[idx]
        p = 0.0
        for i, sig in enumerate(phase.state):
            if sig not in ("G", "g"):
                continue
            # rl-core-007: signal index -> its own incoming lane(s)
            # (spec.signal_lanes), not controlled_lanes[i] (flattened per link).
            for lane_id in spec.incoming_lanes_for_signal(i):
                upstream = _h(lane_id)
                if feeders:
                    upstream += sum(_h(f) for f in feeders.get(lane_id, ()))
                if extra_upstream:
                    upstream += extra_upstream.get(lane_id, 0.0)
                downstream = 0.0
                for dl in get_downstream_lanes(lane_id, downstream_cache):
                    downstream += _h(dl)
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


def build_observation(
    spec: TLSSpec,
    state: TLSRuntimeState,
    cfg: LocalObservationConfig,
    *,
    phase_pressures: list[float],
    min_green: float,
    max_green: float,
    pending_count: float = 0.0,
    upstream_max_queues: list[float] | None = None,
    downstream_max_queues: list[float] | None = None,
) -> np.ndarray:
    """The 22-number observation described in the module docstring. Missing neighbours are 0."""
    greens = spec.green_phase_indices
    n_green = len(greens)
    obs = np.zeros(OBS_SIZE, dtype=np.float32)

    if state.current_phase in greens:
        obs[greens.index(state.current_phase)] = 1.0
    obs[MAX_GREEN_PHASES] = 1.0 if state.yellow_pending else 0.0
    obs[MAX_GREEN_PHASES + 1] = min(state.phase_elapsed / max(min_green, 1.0), 3.0) / 3.0
    obs[MAX_GREEN_PHASES + 2] = min(state.phase_elapsed / max(max_green, 1.0), 1.5) / 1.5

    k = cfg.demand_squash_k
    for i, d in enumerate((list(phase_pressures) + [0.0] * MAX_GREEN_PHASES)[:MAX_GREEN_PHASES]):
        obs[MAX_GREEN_PHASES + 3 + i] = d / (abs(d) + k) if i < n_green else 0.0

    max_q = max(state.lane_queue) if state.lane_queue else 0.0
    obs[2 * MAX_GREEN_PHASES + 3] = max_q / (max_q + cfg.queue_squash_k)
    obs[2 * MAX_GREEN_PHASES + 4] = pending_count / (pending_count + cfg.pending_squash_k)
    obs[2 * MAX_GREEN_PHASES + 5] = n_green / MAX_GREEN_PHASES

    base = 2 * MAX_GREEN_PHASES + 6
    for off, qs in ((0, upstream_max_queues or []), (N_ROAD_NEIGHBORS, downstream_max_queues or [])):
        for i, q in enumerate(qs[:N_ROAD_NEIGHBORS]):
            obs[base + off + i] = q / (q + cfg.queue_squash_k)
    return obs



def compute_feeder_lanes(
    incoming_lanes: list[str],
    downstream_cache: dict[str, list[str]],
    all_lanes: list[str],
    signalised_edges: set[str],
    lane_length,
    reach_m: float,
    max_feeders: int = 60,
) -> dict[str, list[str]]:
    """For each stop-line lane, the lanes of the UNSIGNALISED roads that feed it, walking upstream through the
    lane-to-lane connections until ``reach_m`` metres are covered. A walk stops at a lane whose road ends at a signal
    (that queue belongs to another agent). Static for the network: call once."""
    pred: dict[str, list[str]] = {}
    for ln in all_lanes:
        if ln.startswith(":"):
            continue
        for dl in get_downstream_lanes(ln, downstream_cache):
            if not dl.startswith(":"):
                pred.setdefault(dl, []).append(ln)
    out: dict[str, list[str]] = {}
    for lane in dict.fromkeys(incoming_lanes):
        seen: set[str] = {lane}
        found: list[str] = []
        frontier = [(lane, 0.0)]
        while frontier and len(found) < max_feeders:
            cur, dist = frontier.pop(0)
            for p in pred.get(cur, ()):
                if p in seen or p.rsplit("_", 1)[0] in signalised_edges:
                    continue
                seen.add(p)
                found.append(p)
                nd = dist + lane_length(p)
                if nd < reach_m:
                    frontier.append((p, nd))
        out[lane] = found
    return out
