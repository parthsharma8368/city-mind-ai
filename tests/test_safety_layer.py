"""
test_safety_layer.py
====================
Unit tests for the SafetyLayer — independent of SUMO.

Tests the timing constraint logic without running an actual simulation.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock, patch
from src.scenario.tls_metadata import TLSSpec, PhaseSpec, TLSRuntimeState
from src.environment.safety import SafetyLayer, SafetyConfig


def make_spec(tls_id: str = "TLS_1") -> TLSSpec:
    """Two-phase TLS: green (phase 0) and yellow (phase 1)."""
    phases = [
        PhaseSpec(index=0, state="GGrr", duration=30.0, min_dur=10.0, max_dur=90.0),
        PhaseSpec(index=1, state="yyrr", duration=3.0),   # yellow
        PhaseSpec(index=2, state="rrGG", duration=30.0, min_dur=10.0, max_dur=90.0),
        PhaseSpec(index=3, state="rryy", duration=3.0),   # yellow
    ]
    return TLSSpec(
        tls_id=tls_id,
        program_id="0",
        type="static",
        junction_id=tls_id,
        phases=phases,
        controlled_lanes=["lane_0", "lane_1", "lane_2", "lane_3"],
    )


def make_state(tls_id: str = "TLS_1") -> TLSRuntimeState:
    return TLSRuntimeState(
        tls_id=tls_id,
        current_phase=0,
        phase_elapsed=0.0,
    )


class TestPhaseSpec:
    def test_is_green(self):
        ph = PhaseSpec(index=0, state="GGrr", duration=30.0)
        assert ph.is_green

    def test_is_not_green(self):
        ph = PhaseSpec(index=0, state="rrrr", duration=3.0)
        assert not ph.is_green

    def test_is_yellow(self):
        ph = PhaseSpec(index=0, state="yyyy", duration=3.0)
        assert ph.is_yellow

    def test_is_all_red(self):
        ph = PhaseSpec(index=0, state="rrrr", duration=1.0)
        assert ph.is_all_red


class TestTLSSpec:
    def test_next_green_phase_from_green(self):
        spec = make_spec()
        # From phase 0 (GGrr), next green should be phase 2 (rrGG)
        assert spec.next_green_phase(0) == 2

    def test_next_green_phase_wraps(self):
        spec = make_spec()
        # From phase 2, next green wraps to phase 0
        assert spec.next_green_phase(2) == 0

    def test_green_phase_count(self):
        spec = make_spec()
        assert spec.green_phase_count == 2


class TestSafetyLayerMinGreen:
    """Verify that min_green is respected — no switch before min_green seconds."""

    @patch("src.environment.safety.traci")
    def test_no_switch_before_min_green(self, mock_traci):
        spec = make_spec()
        state = make_state()
        cfg = SafetyConfig(min_green=10.0, yellow_duration=3.0, all_red_duration=0.0)
        layer = SafetyLayer([spec], cfg)
        layer.register_states({spec.tls_id: state})

        # Request advance but only 5 seconds elapsed
        state.phase_elapsed = 5.0
        layer.step(sim_time=5.0, actions={spec.tls_id: 1}, dt=1.0)

        # Phase should NOT have changed (still in phase 0, yellow_pending False)
        assert not state.yellow_pending
        assert state.current_phase == 0

    @patch("src.environment.safety.traci")
    def test_switch_after_min_green(self, mock_traci):
        spec = make_spec()
        state = make_state()
        cfg = SafetyConfig(min_green=10.0, yellow_duration=3.0, all_red_duration=0.0)
        layer = SafetyLayer([spec], cfg)
        layer.register_states({spec.tls_id: state})

        # Simulate 11 seconds elapsed, then request advance
        state.phase_elapsed = 11.0
        layer.step(sim_time=11.0, actions={spec.tls_id: 1}, dt=1.0)

        # Yellow transition should begin
        assert state.yellow_pending
        assert state.yellow_target_phase == 2  # next green is phase 2


class TestSafetyLayerMaxGreen:
    """Verify that max_green forces a phase advance."""

    @patch("src.environment.safety.traci")
    def test_force_advance_at_max_green(self, mock_traci):
        spec = make_spec()
        state = make_state()
        cfg = SafetyConfig(min_green=10.0, max_green=90.0, yellow_duration=3.0, all_red_duration=0.0)
        layer = SafetyLayer([spec], cfg)
        layer.register_states({spec.tls_id: state})

        # 91 seconds elapsed — exceeds max_green
        state.phase_elapsed = 91.0
        layer.step(sim_time=91.0, actions={spec.tls_id: 0}, dt=1.0)  # action=0 (keep)

        # Even with action=0, should still force advance
        assert state.yellow_pending


class TestSafetyLayerYellowTransition:
    """Verify yellow → green transition sequence."""

    @patch("src.environment.safety.traci")
    def test_yellow_clears_to_green(self, mock_traci):
        spec = make_spec()
        state = make_state()
        cfg = SafetyConfig(min_green=10.0, yellow_duration=3.0, all_red_duration=0.0)
        layer = SafetyLayer([spec], cfg)
        layer.register_states({spec.tls_id: state})

        # Trigger yellow
        state.phase_elapsed = 15.0
        layer.step(sim_time=15.0, actions={spec.tls_id: 1}, dt=1.0)
        assert state.yellow_pending

        # Advance 3 yellow seconds
        for t in range(3):
            layer.step(sim_time=16.0 + t, actions={spec.tls_id: 0}, dt=1.0)

        # After 3 seconds of yellow, should switch to target green phase
        assert not state.yellow_pending
        assert state.current_phase == 2
