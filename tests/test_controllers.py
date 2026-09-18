"""Unit tests for baseline traffic signal controllers."""

import pytest
from unittest.mock import patch

from src.controllers.base import BaseController, ControllerResult
from src.controllers.fixed_time import FixedTimeController
from src.controllers.max_pressure import MaxPressureController
from src.scenario.tls_metadata import TLSSpec, PhaseSpec, TLSRuntimeState


def test_fixed_time_controller_init():
    """Verify FixedTimeController initialization and properties."""
    controller = FixedTimeController(step_length=1.0, episode_duration=1800)
    assert controller.name == "fixed_time"
    assert controller.step_length == 1.0
    assert controller.episode_duration == 1800
    assert isinstance(controller, BaseController)


def test_max_pressure_controller_init():
    """Verify MaxPressureController initialization and properties."""
    controller = MaxPressureController(
        min_green=10.0,
        decision_interval=5,
        step_length=1.0,
        episode_duration=1800,
    )
    assert controller.name == "max_pressure"
    assert controller.safety_cfg.min_green == 10.0
    assert controller.decision_interval == 5
    assert controller.episode_duration == 1800
    assert isinstance(controller, BaseController)


def _two_phase_spec() -> TLSSpec:
    """A TLS with two green phases, each controlling one lane."""
    return TLSSpec(
        tls_id="TLS_1",
        program_id="0",
        type="static",
        junction_id="TLS_1",
        phases=[
            PhaseSpec(index=0, state="Gr", duration=30.0),
            PhaseSpec(index=1, state="rG", duration=30.0),
        ],
        controlled_lanes=["lane_ns", "lane_ew"],
    )


@patch("src.controllers.max_pressure.traci")
def test_max_pressure_holds_when_current_phase_has_more_pressure(mock_traci):
    """Current phase serves the busier lane -> controller should hold (action=0)."""
    spec = _two_phase_spec()
    state = TLSRuntimeState(tls_id="TLS_1", current_phase=0)  # phase 0 = lane_ns green

    def halting(lane_id):
        return {"lane_ns": 8.0, "lane_ew": 1.0}[lane_id]

    mock_traci.lane.getLastStepHaltingNumber.side_effect = halting
    mock_traci.lane.getLinks.return_value = []  # no downstream data -> pure upstream pressure
    mock_traci.exceptions.TraCIException = Exception

    controller = MaxPressureController()
    action = controller._decide_action(spec, state, downstream_cache={})
    assert action == 0


@patch("src.controllers.max_pressure.traci")
def test_max_pressure_advances_when_next_phase_has_more_pressure(mock_traci):
    """Next phase serves the busier lane -> controller should request an advance (action=1)."""
    spec = _two_phase_spec()
    state = TLSRuntimeState(tls_id="TLS_1", current_phase=0)  # phase 0 = lane_ns green

    def halting(lane_id):
        return {"lane_ns": 1.0, "lane_ew": 9.0}[lane_id]

    mock_traci.lane.getLastStepHaltingNumber.side_effect = halting
    mock_traci.lane.getLinks.return_value = []
    mock_traci.exceptions.TraCIException = Exception

    controller = MaxPressureController()
    action = controller._decide_action(spec, state, downstream_cache={})
    assert action == 1


@patch("src.controllers.max_pressure.traci")
def test_max_pressure_pressure_subtracts_downstream_queue(mock_traci):
    """Pressure = upstream - downstream; a jammed downstream lane should lower it."""
    spec = _two_phase_spec()

    def halting(lane_id):
        return {"lane_ns": 5.0, "out_ns": 5.0}[lane_id]

    mock_traci.lane.getLastStepHaltingNumber.side_effect = halting
    mock_traci.lane.getLinks.return_value = [("out_ns", True, True, False, "", "", "", 0.0)]
    mock_traci.exceptions.TraCIException = Exception

    controller = MaxPressureController()
    pressure = controller._phase_pressure(spec, spec.phases[0], downstream_cache={})
    assert pressure == pytest.approx(0.0)  # 5 upstream - 5 downstream


def test_controller_result_dataclass():
    """Verify ControllerResult structure and defaults."""
    result = ControllerResult(
        controller="fixed_time",
        seed=42,
        demand_id="1.0x",
        total_steps=100,
        episode_duration=500.0,
        mean_waiting_time=12.5,
        throughput=0.85,
    )
    assert result.controller == "fixed_time"
    assert result.seed == 42
    assert result.mean_waiting_time == 12.5
    assert result.total_switches == 0
