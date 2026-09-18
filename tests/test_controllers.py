"""Unit tests for baseline traffic signal controllers."""

import pytest
from src.controllers.base import BaseController, ControllerResult
from src.controllers.fixed_time import FixedTimeController
from src.controllers.max_pressure import MaxPressureController, MPState


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
    assert controller.min_green == 10.0
    assert controller.decision_interval == 5
    assert controller.episode_duration == 1800
    assert isinstance(controller, BaseController)


def test_mp_state_dataclass():
    """Verify MPState tracking structure."""
    state = MPState(tls_id="tls_1", current_phase=2, phase_elapsed=15.0)
    assert state.tls_id == "tls_1"
    assert state.current_phase == 2
    assert state.phase_elapsed == 15.0


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
