"""Unit tests for the per-TLS local observation (parameter-shared IPPO)."""
from __future__ import annotations

from unittest.mock import patch

from src.environment.local_observation import (
    LOCAL_OBS_SIZE,
    LocalObservationConfig,
    build_local_observation,
)
from src.scenario.tls_metadata import PhaseSpec, TLSRuntimeState, TLSSpec


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

    # mean_queue_norm (index 2) and max_queue_norm (index 3) differ; global features (8, 9) match.
    assert obs_busy[2] > obs_empty[2]
    assert obs_busy[3] > obs_empty[3]
    assert obs_busy[8] == obs_empty[8]
    assert obs_busy[9] == obs_empty[9]
