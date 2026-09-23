"""Unit tests for the per-TLS local observation (parameter-shared IPPO)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.environment.local_observation import (
    LOCAL_OBS_SIZE,
    LocalObservationConfig,
    build_local_observation,
    compute_per_phase_pressure,
)
from src.scenario.tls_metadata import MAX_GREEN_PHASES, PhaseSpec, TLSRuntimeState, TLSSpec


def make_spec(tls_id: str, n_lanes: int, n_phases: int) -> TLSSpec:
    phases = [PhaseSpec(index=i, state="G" * n_lanes, duration=30.0) for i in range(n_phases)]
    return TLSSpec(
        tls_id=tls_id, program_id="0", type="static", junction_id=tls_id,
        phases=phases, controlled_lanes=[f"{tls_id}_lane_{j}" for j in range(n_lanes)],
    )


@patch("src.environment.local_observation.traci")
def test_local_observation_is_fixed_size_regardless_of_lane_count(mock_traci):
    """Two TLS with very different lane counts must produce the same-shaped vector (parameter sharing requires it)."""
    mock_traci.lane.getLinks.return_value = []
    mock_traci.exceptions.TraCIException = Exception

    small = make_spec("small", n_lanes=4, n_phases=3)
    big = make_spec("big", n_lanes=19, n_phases=11)

    state_small = TLSRuntimeState(tls_id="small", current_phase=1, phase_elapsed=15.0)
    state_small.lane_queue = [1.0, 2.0, 0.0, 3.0]
    state_small.lane_speed = [5.0, 6.0, 7.0, 4.0]
    state_small.lane_density = [0.1, 0.2, 0.1, 0.3]

    state_big = TLSRuntimeState(tls_id="big", current_phase=5, phase_elapsed=15.0)
    state_big.lane_queue = [1.0] * 19
    state_big.lane_speed = [5.0] * 19
    state_big.lane_density = [0.1] * 19

    cfg = LocalObservationConfig()
    obs_small = build_local_observation(small, state_small, cfg, {}, total_vehicles=100, total_waiting=50)
    obs_big = build_local_observation(big, state_big, cfg, {}, total_vehicles=100, total_waiting=50)

    assert obs_small.shape == obs_big.shape == (LOCAL_OBS_SIZE,)
    assert obs_small.min() >= 0.0 and obs_small.max() <= 1.0
    assert obs_big.min() >= 0.0 and obs_big.max() <= 1.0


@patch("src.environment.local_observation.traci")
def test_local_observation_reflects_own_queue_not_network(mock_traci):
    """A busy intersection's mean_queue feature should differ from an empty one's, independent of network totals."""
    mock_traci.lane.getLinks.return_value = []
    mock_traci.exceptions.TraCIException = Exception

    spec = make_spec("tls_1", n_lanes=2, n_phases=4)
    busy = TLSRuntimeState(tls_id="tls_1", current_phase=0, phase_elapsed=5.0)
    busy.lane_queue = [40.0, 45.0]
    busy.lane_speed = [0.0, 0.0]
    busy.lane_density = [0.9, 0.9]

    empty = TLSRuntimeState(tls_id="tls_1", current_phase=0, phase_elapsed=5.0)
    empty.lane_queue = [0.0, 0.0]
    empty.lane_speed = [10.0, 10.0]
    empty.lane_density = [0.05, 0.05]

    cfg = LocalObservationConfig()
    obs_busy = build_local_observation(spec, busy, cfg, {}, total_vehicles=500, total_waiting=1000)
    obs_empty = build_local_observation(spec, empty, cfg, {}, total_vehicles=500, total_waiting=1000)

    # mean_queue_norm (index 2) and max_queue_norm (index 3) differ; global features (10, 11) match.
    assert obs_busy[2] > obs_empty[2]
    assert obs_busy[3] > obs_empty[3]
    assert obs_busy[10] == obs_empty[10]
    assert obs_busy[11] == obs_empty[11]


@patch("src.environment.local_observation.traci")
def test_neighbor_queue_norms_land_at_the_documented_indices(mock_traci):
    """neighbor_queue_norm_1/2 (indices 8, 9) should carry through whatever the caller passes,
    independent of this TLS's own state — this is what lets one agent see a neighbor's congestion."""
    mock_traci.lane.getLinks.return_value = []
    mock_traci.exceptions.TraCIException = Exception

    spec = make_spec("tls_1", n_lanes=2, n_phases=4)
    state = TLSRuntimeState(tls_id="tls_1", current_phase=0, phase_elapsed=5.0)
    state.lane_queue = [1.0, 1.0]
    state.lane_speed = [5.0, 5.0]
    state.lane_density = [0.1, 0.1]
    cfg = LocalObservationConfig()

    obs_no_neighbors = build_local_observation(spec, state, cfg, {}, total_vehicles=100, total_waiting=50)
    assert obs_no_neighbors[8] == 0.0 and obs_no_neighbors[9] == 0.0

    obs_with_neighbors = build_local_observation(
        spec, state, cfg, {}, total_vehicles=100, total_waiting=50,
        neighbor_queue_norms=[0.6, 0.3],
    )
    assert obs_with_neighbors[8] == pytest.approx(0.6, abs=1e-6)
    assert obs_with_neighbors[9] == pytest.approx(0.3, abs=1e-6)
    # Everything else about this TLS's own observation is unaffected by its neighbors' values.
    assert obs_with_neighbors[2] == obs_no_neighbors[2]


@patch("src.environment.local_observation.traci")
def test_downstream_queue_sum_precomputed_matches_internal_computation(mock_traci):
    """Passing a precomputed downstream_queue_sum (what the env now does, to
    share the TraCI call with the reward's spillback penalty) must produce
    the same normalized feature as letting build_local_observation compute
    it internally."""
    mock_traci.lane.getLinks.return_value = [("downstream_lane_1", 0, "", "", "", "", "", "")]
    mock_traci.lane.getLastStepHaltingNumber.return_value = 5.0
    mock_traci.exceptions.TraCIException = Exception

    spec = make_spec("tls_1", n_lanes=1, n_phases=2)
    state = TLSRuntimeState(tls_id="tls_1", current_phase=0, phase_elapsed=5.0)
    state.lane_queue = [1.0]
    state.lane_speed = [5.0]
    state.lane_density = [0.1]
    cfg = LocalObservationConfig()

    obs_internal = build_local_observation(spec, state, cfg, {}, total_vehicles=100, total_waiting=50)
    obs_precomputed = build_local_observation(
        spec, state, cfg, {}, total_vehicles=100, total_waiting=50, downstream_queue_sum=5.0,
    )
    assert obs_internal[7] == obs_precomputed[7]


@patch("src.environment.local_observation.traci")
def test_phase_pressures_land_at_documented_indices_and_are_padded(mock_traci):
    """phase_pressure_norm_0..5 (indices 12-17, RL-strategy Phase 2) carry
    through whatever the caller passes, padded with 0.0 for TLS with fewer
    than MAX_GREEN_PHASES green phases."""
    mock_traci.lane.getLinks.return_value = []
    mock_traci.exceptions.TraCIException = Exception

    spec = make_spec("tls_1", n_lanes=2, n_phases=4)
    state = TLSRuntimeState(tls_id="tls_1", current_phase=0, phase_elapsed=5.0)
    state.lane_queue = [1.0, 1.0]
    state.lane_speed = [5.0, 5.0]
    state.lane_density = [0.1, 0.1]
    cfg = LocalObservationConfig(max_phase_pressure=30.0)

    obs_no_pressure = build_local_observation(spec, state, cfg, {}, total_vehicles=100, total_waiting=50)
    assert list(obs_no_pressure[12:18]) == [0.0] * MAX_GREEN_PHASES

    obs_with_pressure = build_local_observation(
        spec, state, cfg, {}, total_vehicles=100, total_waiting=50,
        phase_pressures=[15.0, -60.0],  # only 2 real values; -60 exceeds max_phase_pressure -> clipped
    )
    assert obs_with_pressure[12] == pytest.approx(0.5, abs=1e-6)
    assert obs_with_pressure[13] == pytest.approx(-1.0, abs=1e-6)  # clipped, not just scaled
    assert list(obs_with_pressure[14:18]) == [0.0] * (MAX_GREEN_PHASES - 2)  # padding for missing phases


@patch("src.environment.local_observation.traci")
def test_compute_per_phase_pressure_matches_max_pressure_controller_convention(mock_traci):
    """Same upstream-minus-downstream-halting convention as
    max_pressure.py's _phase_pressure, computed for every green phase at once."""
    mock_traci.exceptions.TraCIException = Exception
    mock_traci.lane.getLastStepHaltingNumber.side_effect = lambda lane_id: {
        "l0": 8.0, "l1": 2.0, "down_0": 3.0, "down_1": 1.0,
    }.get(lane_id, 0.0)
    mock_traci.lane.getLinks.side_effect = lambda lane_id: {
        "l0": [("down_0", 0, "", "", "", "", "", "")],
        "l1": [("down_1", 0, "", "", "", "", "", "")],
    }.get(lane_id, [])

    phases = [
        PhaseSpec(index=0, state="Gr", duration=30.0),  # green on l0 only
        PhaseSpec(index=1, state="yr", duration=3.0),   # yellow, not a green target
        PhaseSpec(index=2, state="rG", duration=30.0),  # green on l1 only
    ]
    spec = TLSSpec(tls_id="t", program_id="0", type="static", junction_id="t",
                    phases=phases, controlled_lanes=["l0", "l1"])

    pressures = compute_per_phase_pressure(spec, {})
    assert pressures == [8.0 - 3.0, 2.0 - 1.0]  # one entry per green phase, cycle order, yellow skipped
