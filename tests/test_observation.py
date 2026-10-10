"""
test_observation.py
===================
Unit tests for ObservationBuilder — shape/bounds/normalization validation.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch, MagicMock
import numpy as np

from src.scenario.tls_metadata import TLSSpec, PhaseSpec, TLSRuntimeState
from src.environment.observation import ObservationBuilder, ObservationConfig


def make_simple_spec(tls_id: str, n_lanes: int = 2, n_phases: int = 4) -> TLSSpec:
    phases = [
        PhaseSpec(index=i, state="G" * n_lanes + "r" * n_lanes, duration=30.0)
        for i in range(n_phases)
    ]
    return TLSSpec(
        tls_id=tls_id,
        program_id="0",
        type="static",
        junction_id=tls_id,
        phases=phases,
        controlled_lanes=[f"{tls_id}_lane_{j}" for j in range(n_lanes)],
    )


def make_state(tls_id: str) -> TLSRuntimeState:
    return TLSRuntimeState(tls_id=tls_id, current_phase=0, phase_elapsed=10.0)


@patch("src.environment.observation.traci")
def test_obs_size_computation(mock_traci):
    specs = [make_simple_spec("TLS_1", n_lanes=2, n_phases=4),
             make_simple_spec("TLS_2", n_lanes=3, n_phases=4)]
    cfg = ObservationConfig(
        include_queue=True, include_vehicle_count=True,
        include_mean_speed=True, include_density=True,
        include_global=True, normalize=True
    )
    builder = ObservationBuilder(specs, cfg)
    size = builder.compute_obs_size()

    # TLS_1: 4 (one_hot) + 1 (elapsed) + 2 (queue) + 2 (vcount) + 2 (speed) + 2 (density) = 13
    # TLS_2: 4 + 1 + 3 + 3 + 3 + 3 = 17
    # Global: 2
    expected = 13 + 17 + 2
    assert size == expected, f"Expected {expected}, got {size}"


@patch("src.environment.observation.traci")
def test_obs_values_in_range(mock_traci):
    """All observation values should be in [0, 1] when normalized."""
    # Mock TraCI lane calls
    mock_traci.lane.getLastStepHaltingNumber.return_value = 10
    mock_traci.lane.getLastStepVehicleNumber.return_value = 15
    mock_traci.lane.getLastStepMeanSpeed.return_value = 5.0
    mock_traci.lane.getLastStepOccupancy.return_value = 40.0  # percent
    mock_traci.vehicle.getIDCount.return_value = 100
    mock_traci.vehicle.getIDList.return_value = ["v1", "v2"]
    mock_traci.vehicle.getWaitingTime.return_value = 5.0
    mock_traci.exceptions.TraCIException = Exception

    specs = [make_simple_spec("TLS_1", n_lanes=2, n_phases=4)]
    states = {"TLS_1": make_state("TLS_1")}
    builder = ObservationBuilder(specs, ObservationConfig(normalize=True))
    obs = builder.build(states)

    assert obs.min() >= 0.0, f"obs min < 0: {obs.min()}"
    assert obs.max() <= 1.001, f"obs max > 1: {obs.max()}"  # small tolerance


@patch("src.environment.observation.traci")
def test_obs_shape_stable(mock_traci):
    """Observation shape should be identical on consecutive calls."""
    mock_traci.lane.getLastStepHaltingNumber.return_value = 5
    mock_traci.lane.getLastStepVehicleNumber.return_value = 8
    mock_traci.lane.getLastStepMeanSpeed.return_value = 3.0
    mock_traci.lane.getLastStepOccupancy.return_value = 20.0
    mock_traci.vehicle.getIDCount.return_value = 50
    mock_traci.vehicle.getIDList.return_value = []
    mock_traci.exceptions.TraCIException = Exception

    specs = [make_simple_spec("TLS_1"), make_simple_spec("TLS_2")]
    states = {s.tls_id: make_state(s.tls_id) for s in specs}
    builder = ObservationBuilder(specs)
    builder.compute_obs_size()

    obs1 = builder.build(states)
    obs2 = builder.build(states)
    assert obs1.shape == obs2.shape, "Observation shape changed between calls"


@patch("src.environment.observation.traci")
def test_obs_exact_values_and_write_back_of_raw_units(mock_traci):
    """Exact normalized vector for a known lane state (replaces the shape-only
    check): [one_hot(4), elapsed, queue x2, vcount x2, speed x2, density x2, global x2]."""
    mock_traci.lane.getLastStepHaltingNumber.side_effect = lambda lid: {"TLS_1_lane_0": 10, "TLS_1_lane_1": 100}[lid]
    mock_traci.lane.getLastStepVehicleNumber.return_value = 25
    mock_traci.lane.getLastStepMeanSpeed.side_effect = lambda lid: {"TLS_1_lane_0": 6.95, "TLS_1_lane_1": -1.0}[lid]
    mock_traci.lane.getLastStepOccupancy.return_value = 40.0
    mock_traci.vehicle.getIDList.return_value = ["v1", "v2"]
    mock_traci.vehicle.getWaitingTime.side_effect = lambda v: {"v1": 30.0, "v2": 90.0}[v]
    mock_traci.exceptions.TraCIException = Exception

    spec = make_simple_spec("TLS_1", n_lanes=2, n_phases=4)
    state = make_state("TLS_1")             # phase 0, elapsed 10
    builder = ObservationBuilder([spec], ObservationConfig())
    obs = builder.build({"TLS_1": state})

    expected = [1, 0, 0, 0, 10 / 120,
                10 / 50, 1.0,               # queue: clipped at 1
                25 / 50, 25 / 50,
                6.95 / 13.9, 0.0,           # negative speed clamped to 0
                0.4, 0.4,
                2 / 2000, 120 / 3600]
    assert obs.dtype == np.float32 and obs.shape == (builder.compute_obs_size(),)
    np.testing.assert_allclose(obs, expected, rtol=1e-5, atol=1e-6)
    # shared runtime state keeps RAW units for reward/metrics
    assert state.lane_queue == [10.0, 100.0] and state.lane_speed == [6.95, -1.0]
    assert builder.last_step_cache["total_waiting"] == 120.0
