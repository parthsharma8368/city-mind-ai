"""The 22-number observation: layout, scaling, phase demand (queue reach, pending cars) and neighbour queues."""
from __future__ import annotations

from unittest.mock import patch

import pytest
import traci.exceptions as _texc

from src.environment.local_observation import (
    N_ROAD_NEIGHBORS,
    OBS_NAMES,
    OBS_SIZE,
    LocalObservationConfig,
    build_observation,
    compute_downstream_queue_sum,
    compute_feeder_lanes,
    compute_per_phase_pressure,
)
from src.scenario.tls_metadata import MAX_GREEN_PHASES, PhaseSpec, TLSRuntimeState, TLSSpec


def _spec(n_green: int = 3) -> TLSSpec:
    phases = []
    for g in range(n_green):  # green, yellow, green, yellow, ...
        phases.append(PhaseSpec(index=2 * g, state="G" * 4, duration=30.0))
        phases.append(PhaseSpec(index=2 * g + 1, state="y" * 4, duration=4.0))
    return TLSSpec(tls_id="T", program_id="0", type="static", junction_id="T", phases=phases,
                   controlled_lanes=["a_0", "a_1", "b_0", "c_0"])


def _obs(spec=None, *, phase=2, elapsed=10.0, pressures=(5.0, -3.0, 0.0), min_g=5.0, max_g=60.0, pending=0, yellow=False,
         up=None, down=None):
    spec = spec or _spec()
    st = TLSRuntimeState(tls_id="T", current_phase=phase, phase_elapsed=elapsed)
    st.yellow_pending = yellow
    st.lane_queue = [1.0, 7.0, 0.0, 2.0]
    return build_observation(spec, st, LocalObservationConfig(), phase_pressures=list(pressures), min_green=min_g,
                             max_green=max_g, pending_count=pending, upstream_max_queues=up, downstream_max_queues=down)


def test_layout_is_22_named_slots():
    assert OBS_SIZE == len(OBS_NAMES) == 2 * MAX_GREEN_PHASES + 6 + 2 * N_ROAD_NEIGHBORS == 22
    assert _obs().shape == (OBS_SIZE,)


def test_active_slot_one_hot_and_transition_flag():
    o = _obs(phase=2)                                 # phase 2 is the 2nd green -> slot 1
    assert o[:MAX_GREEN_PHASES].tolist() == [0, 1, 0, 0, 0, 0]
    assert o[OBS_NAMES.index("in_transition")] == 0.0
    assert _obs(phase=2, yellow=True)[OBS_NAMES.index("in_transition")] == 1.0


def test_elapsed_is_relative_to_the_phases_own_min_and_max():
    """10 s means 'twice the minimum' for a 5 s phase but 'a quarter' for a 40 s one."""
    short, long_ = _obs(elapsed=10.0, min_g=5.0), _obs(elapsed=10.0, min_g=40.0)
    i = OBS_NAMES.index("elapsed_over_min")
    assert short[i] == pytest.approx(2.0 / 3.0) and long_[i] == pytest.approx(0.25 / 3.0)
    assert _obs(elapsed=1000.0)[i] == 1.0            # clipped at 3x
    assert _obs(elapsed=1000.0)[OBS_NAMES.index("elapsed_over_max")] == 1.0


def test_demand_is_signed_squashed_and_padded_slots_are_zero():
    o = _obs(spec=_spec(3), pressures=(8.0, -8.0, 0.0, 99.0))   # slot 3 does not exist for a 3-green signal
    d = [OBS_NAMES.index(f"slot_demand_{i}") for i in range(MAX_GREEN_PHASES)]
    assert o[d[0]] == pytest.approx(0.5) and o[d[1]] == pytest.approx(-0.5) and o[d[2]] == 0.0
    assert o[d[3]] == 0.0 and o[d[5]] == 0.0 and (abs(o) <= 1.0).all()


def test_queue_pending_and_n_green_scale_into_a_usable_range():
    o = _obs(spec=_spec(3), pending=20)
    assert o[OBS_NAMES.index("max_queue")] == pytest.approx(7.0 / 15.0)
    assert o[OBS_NAMES.index("pending")] == pytest.approx(0.5)
    assert o[OBS_NAMES.index("n_green")] == pytest.approx(3 / 6)


def test_neighbour_queues_are_squashed_truncated_and_padded():
    o = _obs(up=[8.0], down=[8.0, 24.0, 99.0])
    i = OBS_NAMES.index("upstream_queue_0")
    assert o[i] == pytest.approx(0.5) and o[i + 1] == 0.0                       # one known, one missing -> 0
    assert o[i + 2] == pytest.approx(0.5) and o[i + 3] == pytest.approx(0.75)   # a third downstream is ignored
    assert not _obs()[i:].any()                                                 # no neighbours -> zeros, nothing else moves
    assert (_obs(up=[8.0], down=[8.0])[:i] == _obs()[:i]).all()


# ── phase demand (Varaiya pressure) ─────────────────────────────────────────────
def _wire(mock_traci):
    mock_traci.exceptions.TraCIException = _texc.TraCIException


@patch("src.environment.local_observation.traci")
def test_phase_demand_is_upstream_minus_downstream_one_entry_per_green_phase(mock_traci):
    _wire(mock_traci)
    halting = {"l0": 8.0, "l1": 2.0, "down_0": 3.0, "down_1": 1.0}
    mock_traci.lane.getLastStepHaltingNumber.side_effect = lambda lane: halting.get(lane, 0.0)
    mock_traci.lane.getLinks.side_effect = lambda lane: {"l0": [("down_0", 0)], "l1": [("down_1", 0)]}.get(lane, [])
    phases = [PhaseSpec(0, "Gr", 30.0), PhaseSpec(1, "yr", 3.0), PhaseSpec(2, "rG", 30.0)]
    spec = TLSSpec("t", "0", "static", "t", phases, controlled_lanes=["l0", "l1"])
    assert compute_per_phase_pressure(spec, {}) == [8.0 - 3.0, 2.0 - 1.0]      # yellow skipped, cycle order


@patch("src.environment.local_observation.traci")
def test_lowercase_g_movements_count_for_phase_demand(mock_traci):
    _wire(mock_traci)
    mock_traci.lane.getLinks.return_value = []
    mock_traci.lane.getLastStepHaltingNumber.side_effect = lambda lane: {"l0": 5.0, "l1": 7.0}[lane]
    spec = TLSSpec("t", "0", "static", "t", [PhaseSpec(0, "gr", 10.0), PhaseSpec(1, "rG", 10.0)], ["l0", "l1"])
    assert compute_per_phase_pressure(spec, {}) == [5.0, 7.0]


@patch("src.environment.local_observation.traci")
def test_phase_demand_uses_signal_lanes_not_the_flattened_controlled_lanes(mock_traci):
    """controlled_lanes is flattened per LINK, so index i is not signal i."""
    _wire(mock_traci)
    mock_traci.lane.getLinks.return_value = []
    halting = {"A": 1.0, "B": 20.0, "C": 300.0}
    mock_traci.lane.getLastStepHaltingNumber.side_effect = lambda lane: halting[lane]
    phases = [PhaseSpec(0, "rGr", 10.0), PhaseSpec(1, "rrG", 10.0)]
    spec = TLSSpec("t", "0", "static", "t", phases, controlled_lanes=["A", "A", "B", "C"],
                   signal_lanes=[["A"], ["B"], ["C"]])
    assert compute_per_phase_pressure(spec, {}) == [20.0, 300.0]


@patch("src.environment.local_observation.traci")
def test_downstream_queue_sum_is_the_mean_halted_per_downstream_lane(mock_traci):
    _wire(mock_traci)
    mock_traci.lane.getLinks.side_effect = lambda lane: {"l0": [("d0", 0)], "l1": [("d1", 0)]}.get(lane, [])
    mock_traci.lane.getLastStepHaltingNumber.side_effect = lambda lane: {"d0": 4.0, "d1": 2.0}[lane]
    spec = TLSSpec("t", "0", "static", "t", [PhaseSpec(0, "GG", 10.0)], ["l0", "l1"])
    assert compute_downstream_queue_sum(spec, {}) == pytest.approx(3.0)
    mock_traci.lane.getLinks.side_effect = None
    mock_traci.lane.getLinks.return_value = []
    assert compute_downstream_queue_sum(spec, {}) == 0.0         # no downstream lanes (a fresh cache)


# ── queue reach and pending cars ─────────────────────────────────────────────────
def test_feeder_lanes_stop_at_the_previous_signal_and_at_the_reach_limit():
    # up1 -> mid -> long -> X(stop-line, signalised end);  up1's edge ends at a signal (another agent's queue)
    links = {"up1_0": ["mid_0"], "mid_0": ["long_0"], "long_0": ["X_0"], "far_0": ["up1_0"], "X_0": []}
    lengths = {"mid_0": 100.0, "long_0": 221.0, "up1_0": 80.0, "far_0": 50.0}
    lanes = ["up1_0", "mid_0", "long_0", "X_0", "far_0"]
    out = compute_feeder_lanes(["X_0"], links, lanes, {"X", "up1"}, lambda lane: lengths[lane], reach_m=300.0)
    assert out["X_0"] == ["long_0", "mid_0"]            # up1 ends at a signal -> not included, nothing beyond it
    short = compute_feeder_lanes(["X_0"], links, lanes, {"X", "up1"}, lambda lane: lengths[lane], reach_m=100.0)
    assert short["X_0"] == ["long_0"]                    # 221 m already covers 100 m


def test_phase_demand_sees_cars_queued_behind_a_short_stop_line_lane_and_cars_waiting_outside():
    spec = TLSSpec(tls_id="T", program_id="0", type="static", junction_id="T",
                   phases=[PhaseSpec(0, "GG", 30.0), PhaseSpec(1, "yy", 3.0)],
                   controlled_lanes=["X_0", "Y_0"], signal_lanes=[["X_0"], ["Y_0"]])
    halting = {"X_0": 2.0, "Y_0": 0.0, "long_0": 30.0, "mid_0": 4.0}
    no_links = {"X_0": [], "Y_0": []}
    plain = compute_per_phase_pressure(spec, no_links, halting=halting.__getitem__)
    reach = compute_per_phase_pressure(spec, no_links, feeders={"X_0": ["long_0", "mid_0"]}, halting=halting.__getitem__)
    outside = compute_per_phase_pressure(spec, no_links, halting=halting.__getitem__, extra_upstream={"X_0": 30.0})
    assert plain == [2.0] and reach == [36.0] and outside == [32.0]
