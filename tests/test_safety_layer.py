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


def make_three_phase_spec(tls_id: str = "TLS_3") -> TLSSpec:
    """Three-green-phase TLS (green/yellow x3) — for phase-selection jumps
    that skip over an intervening green phase (e.g. green0 -> green2)."""
    phases = [
        PhaseSpec(index=0, state="GGrrrr", duration=30.0, min_dur=10.0, max_dur=90.0),
        PhaseSpec(index=1, state="yyrrrr", duration=3.0),   # yellow
        PhaseSpec(index=2, state="rrGGrr", duration=30.0, min_dur=10.0, max_dur=90.0),
        PhaseSpec(index=3, state="rryyrr", duration=3.0),   # yellow
        PhaseSpec(index=4, state="rrrrGG", duration=30.0, min_dur=10.0, max_dur=90.0),
        PhaseSpec(index=5, state="rrrryy", duration=3.0),   # yellow
    ]
    return TLSSpec(
        tls_id=tls_id, program_id="0", type="static", junction_id=tls_id,
        phases=phases, controlled_lanes=["l0", "l1", "l2", "l3", "l4", "l5"],
    )


def make_one_phase_spec(tls_id: str = "TLS_1PHASE") -> TLSSpec:
    """Degenerate single-green-phase TLS (2/98 real InTAS TLS have this)."""
    phases = [PhaseSpec(index=0, state="GGGG", duration=30.0, min_dur=10.0, max_dur=90.0)]
    return TLSSpec(
        tls_id=tls_id, program_id="0", type="static", junction_id=tls_id,
        phases=phases, controlled_lanes=["l0", "l1", "l2", "l3"],
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

    def test_mixed_yellow_and_green_is_not_a_green_target(self):
        """Regression test (2026-09-21): a phase retaining a permissive
        green movement through its own yellow clearance (real InTAS
        pattern, e.g. "yyggrrrryyygrrrr") must not be treated as a legal
        cycle target — it is a transition, not a stable phase."""
        ph = PhaseSpec(index=5, state="yyggrrrryyygrrrr", duration=3.0)
        assert not ph.is_green
        assert ph.is_yellow

    def test_is_green_and_is_yellow_are_mutually_exclusive(self):
        for state in ("GGrr", "rrrr", "yyyy", "yyGGrr", "ggrrYY".lower()):
            ph = PhaseSpec(index=0, state=state, duration=1.0)
            assert not (ph.is_green and ph.is_yellow)


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

    def test_next_green_phase_skips_mixed_yellow_green_phase(self):
        """A phase containing both G and y (permissive-through-yellow) must
        be skipped as a target, not selected — real InTAS pattern."""
        phases = [
            PhaseSpec(index=0, state="GGgrrrr", duration=33.0, min_dur=33.0, max_dur=33.0),
            PhaseSpec(index=1, state="yyyrrrr", duration=3.0),
            PhaseSpec(index=2, state="rrrGGGg", duration=6.0, min_dur=6.0, max_dur=6.0),
            PhaseSpec(index=3, state="yyggrrrryyygrrrr"[:7], duration=3.0),  # mixed y+g, not a target
        ]
        spec = TLSSpec(tls_id="X", program_id="0", type="actuated", junction_id="X",
                        phases=phases, controlled_lanes=["l0", "l1", "l2", "l3", "l4", "l5", "l6"])
        assert spec.next_green_phase(0) == 2
        # from phase 2, next green wraps past phase 3 (mixed, not green) to phase 0
        assert spec.next_green_phase(2) == 0


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

    @patch("src.environment.safety.traci")
    def test_current_phase_frozen_during_transition(self, mock_traci):
        """current_phase (read by observation/reward/switch-counting) must
        stay at the pre-switch value for the whole transition — only
        transition_display_phase (safety-layer-internal) tracks the
        native sub-phase actually being shown to SUMO. This preserves
        exactly-one-change-per-logical-switch semantics."""
        spec = make_spec()
        state = make_state()
        cfg = SafetyConfig(min_green=10.0)
        layer = SafetyLayer([spec], cfg)
        layer.register_states({spec.tls_id: state})

        state.phase_elapsed = 15.0
        layer.step(sim_time=15.0, actions={spec.tls_id: 1}, dt=1.0)
        assert state.current_phase == 0          # unchanged mid-transition
        assert state.transition_display_phase == 1  # showing the native yellow phase

        for t in range(3):
            layer.step(sim_time=16.0 + t, actions={spec.tls_id: 0}, dt=1.0)
        assert state.current_phase == 2           # changes exactly once, on landing
        assert state.transition_display_phase == -1

    @patch("src.environment.safety.traci")
    def test_uses_native_yellow_duration_not_config_default(self, mock_traci):
        """The program's own yellow-phase duration (5s here) must govern
        the transition even when cfg.yellow_duration disagrees (3s)."""
        phases = [
            PhaseSpec(index=0, state="GGrr", duration=30.0, min_dur=10.0, max_dur=90.0),
            PhaseSpec(index=1, state="yyrr", duration=5.0),   # native yellow = 5s
            PhaseSpec(index=2, state="rrGG", duration=30.0, min_dur=10.0, max_dur=90.0),
            PhaseSpec(index=3, state="rryy", duration=5.0),
        ]
        spec = TLSSpec(tls_id="Y", program_id="0", type="actuated", junction_id="Y",
                        phases=phases, controlled_lanes=["l0", "l1", "l2", "l3"])
        state = make_state("Y")
        cfg = SafetyConfig(min_green=10.0, yellow_duration=3.0)  # config disagrees with program
        layer = SafetyLayer([spec], cfg)
        layer.register_states({spec.tls_id: state})

        state.phase_elapsed = 15.0
        layer.step(sim_time=15.0, actions={spec.tls_id: 1}, dt=1.0)
        assert state.yellow_pending

        # After only 3s (the config value) it must NOT have landed yet
        for t in range(3):
            layer.step(sim_time=16.0 + t, actions={spec.tls_id: 0}, dt=1.0)
        assert state.yellow_pending
        assert state.current_phase == 0

        # After 2 more seconds (5s total, the native duration) it lands
        for t in range(2):
            layer.step(sim_time=19.0 + t, actions={spec.tls_id: 0}, dt=1.0)
        assert not state.yellow_pending
        assert state.current_phase == 2

    @patch("src.environment.safety.traci")
    def test_two_hop_transition_walks_both_native_sub_phases(self, mock_traci):
        """Real InTAS pattern (TLS 279014956, 3/303 real greens need 2
        hops): a mixed yellow+green phase followed by a pure-yellow phase
        must both be walked in order, each for its own native duration,
        before landing on the target green."""
        phases = [
            PhaseSpec(index=0, state="GGGGGrrrrr", duration=50.0, min_dur=50.0, max_dur=50.0),
            PhaseSpec(index=1, state="GGyyyrrrrr", duration=5.0),   # mixed y+g sub-phase 1
            PhaseSpec(index=2, state="yyrrrrrrrr", duration=3.0),   # pure yellow sub-phase 2
            PhaseSpec(index=3, state="rrrrrGGGGG", duration=25.0, min_dur=25.0, max_dur=25.0),
            PhaseSpec(index=4, state="rrrrryyyyy", duration=3.0),
        ]
        spec = TLSSpec(tls_id="M", program_id="0", type="actuated", junction_id="M",
                        phases=phases, controlled_lanes=["l"] * 10)
        state = make_state("M")
        cfg = SafetyConfig(min_green=10.0)
        layer = SafetyLayer([spec], cfg)
        layer.register_states({spec.tls_id: state})

        state.phase_elapsed = 60.0
        layer.step(sim_time=60.0, actions={spec.tls_id: 1}, dt=1.0)
        assert state.transition_display_phase == 1   # mixed sub-phase first
        assert state.current_phase == 0               # frozen

        for t in range(5):  # native duration of sub-phase 1
            layer.step(sim_time=61.0 + t, actions={spec.tls_id: 0}, dt=1.0)
        assert state.yellow_pending
        assert state.transition_display_phase == 2    # then pure yellow
        assert state.current_phase == 0                # still frozen

        for t in range(3):  # native duration of sub-phase 2
            layer.step(sim_time=66.0 + t, actions={spec.tls_id: 0}, dt=1.0)
        assert not state.yellow_pending
        assert state.current_phase == 3                # lands on the real target

    @patch("src.environment.safety.traci")
    def test_zero_hop_transition_switches_immediately(self, mock_traci):
        """When the program's own sequence has no phase between two green
        phases (3/303 real greens network-wide — e.g. adding a compatible
        protected movement), the switch must be immediate: no invented
        yellow the program's authors didn't specify."""
        phases = [
            PhaseSpec(index=0, state="GGrr", duration=30.0, min_dur=10.0, max_dur=90.0),
            PhaseSpec(index=1, state="GGGG", duration=10.0, min_dur=10.0, max_dur=30.0),  # adjacent green
        ]
        spec = TLSSpec(tls_id="W", program_id="0", type="actuated", junction_id="W",
                        phases=phases, controlled_lanes=["l0", "l1", "l2", "l3"])
        state = make_state("W")
        cfg = SafetyConfig(min_green=10.0)
        layer = SafetyLayer([spec], cfg)
        layer.register_states({spec.tls_id: state})

        state.phase_elapsed = 15.0
        layer.step(sim_time=15.0, actions={spec.tls_id: 1}, dt=1.0)
        assert not state.yellow_pending   # never entered a transition
        assert state.current_phase == 1   # switched immediately
        assert state.phase_elapsed == 0.0

    @patch("src.environment.safety.traci")
    def test_static_program_phase_falls_back_to_config_min_green(self, mock_traci):
        """A phase with no distinct min_dur/max_dur (SUMO "static" program,
        min_dur == duration per the loader default) must use cfg.min_green,
        not be locked to its own long fixed duration."""
        phases = [
            PhaseSpec(index=0, state="GGrr", duration=26.0, min_dur=26.0, max_dur=26.0),
            PhaseSpec(index=1, state="yyrr", duration=4.0),
            PhaseSpec(index=2, state="rrGG", duration=26.0, min_dur=26.0, max_dur=26.0),
            PhaseSpec(index=3, state="rryy", duration=4.0),
        ]
        spec = TLSSpec(tls_id="S", program_id="0", type="static", junction_id="S",
                        phases=phases, controlled_lanes=["l0", "l1", "l2", "l3"])
        state = make_state("S")
        cfg = SafetyConfig(min_green=5.0)  # permissive fallback
        layer = SafetyLayer([spec], cfg)
        layer.register_states({spec.tls_id: state})

        # Only 6s elapsed — less than the phase's own fixed 26s, but past
        # the permissive cfg.min_green=5.0 fallback.
        state.phase_elapsed = 6.0
        layer.step(sim_time=6.0, actions={spec.tls_id: 1}, dt=1.0)
        assert state.yellow_pending  # allowed to advance early


class TestSafetyLayerPhaseSelect:
    """mode="phase_select": action is a direct target green-phase index
    (via TLSSpec.action_to_phase_index), not binary keep/advance. Binary
    mode's own tests above are the regression guard that this mode's
    existence doesn't change binary behavior at all."""

    @patch("src.environment.safety.traci")
    def test_selecting_current_phase_holds(self, mock_traci):
        spec = make_three_phase_spec()
        state = make_state(spec.tls_id)
        cfg = SafetyConfig(min_green=10.0, all_red_duration=2.0)
        layer = SafetyLayer([spec], cfg)
        layer.register_states({spec.tls_id: state})

        state.phase_elapsed = 15.0
        layer.step(sim_time=15.0, actions={spec.tls_id: 0}, dt=1.0, mode="phase_select")

        assert not state.yellow_pending
        assert state.current_phase == 0

    @patch("src.environment.safety.traci")
    def test_adjacent_target_uses_native_transition(self, mock_traci):
        """action=1 (this TLS's own 2nd green, native phase index 2) is the
        immediate next green in the cycle — must reuse the exact native
        yellow transition, not the synthesized bridge."""
        spec = make_three_phase_spec()
        state = make_state(spec.tls_id)
        cfg = SafetyConfig(min_green=10.0, all_red_duration=2.0)
        layer = SafetyLayer([spec], cfg)
        layer.register_states({spec.tls_id: state})

        state.phase_elapsed = 15.0
        layer.step(sim_time=15.0, actions={spec.tls_id: 1}, dt=1.0, mode="phase_select")

        assert state.yellow_pending
        assert state.yellow_target_phase == 2
        assert state.transition_display_phase == 1  # native yellow (index 1), not the synthetic sentinel

    @patch("src.environment.safety.traci")
    def test_nonadjacent_jump_uses_synthetic_yellow_then_all_red_bridge(self, mock_traci):
        """action=2 (this TLS's own 3rd green, native phase index 4) skips
        over phase 2's green entirely -- must never display phase 2's own
        green state, and must route through a SYNTHESIZED yellow (every G/g of
        the current state -> y, rl-core-001) then a synthesized all-red before
        landing on phase 4."""
        spec = make_three_phase_spec()
        state = make_state(spec.tls_id)
        cfg = SafetyConfig(min_green=10.0, yellow_duration=3.0, all_red_duration=2.0)
        layer = SafetyLayer([spec], cfg)
        layer.register_states({spec.tls_id: state})
        applied_states = []
        mock_traci.trafficlight.setRedYellowGreenState.side_effect = (
            lambda tls_id, s: applied_states.append(s)
        )

        state.phase_elapsed = 15.0
        layer.step(sim_time=15.0, actions={spec.tls_id: 2}, dt=1.0, mode="phase_select")

        assert state.yellow_pending
        assert state.yellow_target_phase == 4
        assert state.transition_display_phase == SafetyLayer._SYNTHETIC_YELLOW
        assert state.transition_queue == [SafetyLayer._SYNTHETIC_ALL_RED]
        assert applied_states == ["yyrrrr"]  # exactly the current green with G->y

        layer.step(sim_time=18.0, actions={spec.tls_id: 2}, dt=3.0, mode="phase_select")  # yellow_duration elapses
        assert state.transition_display_phase == SafetyLayer._SYNTHETIC_ALL_RED
        assert applied_states[-1] == "rrrrrr"  # sized to this TLS's own link count

        layer.step(sim_time=20.0, actions={spec.tls_id: 2}, dt=2.0, mode="phase_select")  # all_red_duration elapses
        assert not state.yellow_pending
        assert state.current_phase == 4
        assert applied_states == ["yyrrrr", "rrrrrr", "rrrrGG"]
        assert "rrGGrr" not in applied_states  # intervening green never displayed

    @patch("src.environment.safety.traci")
    def test_out_of_range_action_falls_back_to_hold(self, mock_traci):
        """Action value beyond this TLS's own green-phase count (e.g. a
        3-green TLS receiving action=5 from the shared Discrete(6) space)
        must hold rather than error or pick something arbitrary — this is
        the defensive fallback underneath MaskablePPO's masking, not a
        replacement for it."""
        spec = make_three_phase_spec()
        state = make_state(spec.tls_id)
        layer = SafetyLayer([spec], SafetyConfig(min_green=10.0))
        layer.register_states({spec.tls_id: state})

        state.phase_elapsed = 15.0
        layer.step(sim_time=15.0, actions={spec.tls_id: 5}, dt=1.0, mode="phase_select")

        assert not state.yellow_pending
        assert state.current_phase == 0

    @patch("src.environment.safety.traci")
    def test_single_phase_tls_never_switches(self, mock_traci):
        """Degenerate 1-green-phase TLS (2/98 real InTAS TLS) — every action
        value maps to the same phase, so it must never enter a transition."""
        spec = make_one_phase_spec()
        state = make_state(spec.tls_id)
        layer = SafetyLayer([spec], SafetyConfig(min_green=10.0))
        layer.register_states({spec.tls_id: state})

        state.phase_elapsed = 50.0
        for action in range(6):
            layer.step(sim_time=50.0, actions={spec.tls_id: action}, dt=1.0, mode="phase_select")
            assert not state.yellow_pending
            assert state.current_phase == 0

    @patch("src.environment.safety.traci")
    def test_action_mask_matches_safety_layer_acceptance(self, mock_traci):
        """TLSSpec.action_mask() must agree with what the safety layer will
        actually accept: masked-in actions map to real green phases,
        masked-out ones fall back to hold."""
        spec = make_three_phase_spec()
        mask = spec.action_mask()
        assert mask == [True, True, True, False, False, False]

        state = make_state(spec.tls_id)
        layer = SafetyLayer([spec], SafetyConfig(min_green=10.0, all_red_duration=1.0))
        layer.register_states({spec.tls_id: state})
        state.phase_elapsed = 15.0
        layer.step(sim_time=15.0, actions={spec.tls_id: 2}, dt=1.0, mode="phase_select")
        assert state.yellow_pending  # masked-in action accepted as a real switch


# ──────────────────────────────────────────────────────────────────────────────
# Mutation-killing tests (tests-reliability-002): exact boundaries, exact strings
# ──────────────────────────────────────────────────────────────────────────────

def _layer(spec, state, cfg=None):
    layer = SafetyLayer([spec], cfg or SafetyConfig())
    layer.register_states({spec.tls_id: state})
    return layer


class TestExactBoundaries:
    @patch("src.environment.safety.traci")
    def test_min_green_boundary_is_inclusive_to_the_second(self, mock_traci):
        """Phase 0 has min_dur=10: elapsed 9 -> no switch, elapsed 10 -> switch."""
        spec, state = make_spec(), make_state()
        layer = _layer(spec, state)
        state.phase_elapsed = 8.0
        layer.step(0.0, {spec.tls_id: 1}, 1.0)          # elapsed becomes 9
        assert not state.yellow_pending
        layer.step(1.0, {spec.tls_id: 1}, 1.0)          # elapsed becomes 10 == min_dur
        assert state.yellow_pending
        assert state.yellow_target_phase == 2

    @patch("src.environment.safety.traci")
    def test_max_green_boundary_forces_exactly_at_max(self, mock_traci):
        spec, state = make_spec(), make_state()          # max_dur=90
        layer = _layer(spec, state)
        state.phase_elapsed = 88.0
        layer.step(0.0, {spec.tls_id: 0}, 1.0)           # 89: below max
        assert not state.yellow_pending
        layer.step(1.0, {spec.tls_id: 0}, 1.0)           # 90: forced
        assert state.yellow_pending

    @patch("src.environment.safety.traci")
    def test_binary_mode_only_action_one_advances(self, mock_traci):
        spec, state = make_spec(), make_state()
        layer = _layer(spec, state)
        state.phase_elapsed = 20.0
        for a in (0, 2, 5, -1):
            layer.step(0.0, {spec.tls_id: a}, 1.0)
            assert not state.yellow_pending, f"action {a} must not advance in binary mode"
        layer.step(0.0, {spec.tls_id: 1}, 1.0)
        assert state.yellow_pending

    @patch("src.environment.safety.traci")
    def test_exact_strings_written_and_last_switch_time_set(self, mock_traci):
        spec, state = make_spec(), make_state()
        layer = _layer(spec, state)
        written = []
        mock_traci.trafficlight.setRedYellowGreenState.side_effect = lambda t, s: written.append((t, s))
        state.phase_elapsed = 20.0
        layer.step(100.0, {spec.tls_id: 1}, 1.0)         # starts yellow (phase 1)
        for t in (101.0, 102.0, 103.0):
            layer.step(t, {spec.tls_id: 1}, 1.0)         # 3 s yellow, then phase 2
        assert written == [("TLS_1", "yyrr"), ("TLS_1", "rrGG")]
        assert state.current_phase == 2
        assert state.last_switch_time == 103.0
        assert state.phase_elapsed == 0.0
        assert state.displayed_state == "rrGG"

    @patch("src.environment.safety.traci")
    def test_writes_the_state_of_the_phase_not_phase_zero(self, mock_traci):
        spec = make_three_phase_spec()
        state = make_state(spec.tls_id)
        layer = _layer(spec, state, SafetyConfig(min_green=1.0))
        written = []
        mock_traci.trafficlight.setRedYellowGreenState.side_effect = lambda t, s: written.append(s)
        state.phase_elapsed = 50.0
        layer.step(0.0, {spec.tls_id: 1}, 1.0)
        for t in range(1, 5):
            layer.step(float(t), {spec.tls_id: 1}, 1.0)
        assert written[-1] == "rrGGrr"                   # phase 2's string, not phase 0's "GGrrrr"

    @patch("src.environment.safety.traci")
    def test_synthetic_all_red_actually_lasts_all_red_duration(self, mock_traci):
        spec = make_three_phase_spec()
        state = make_state(spec.tls_id)
        cfg = SafetyConfig(min_green=1.0, yellow_duration=3.0, all_red_duration=4.0)
        layer = _layer(spec, state, cfg)
        state.phase_elapsed = 50.0
        layer.step(0.0, {spec.tls_id: 2}, 1.0, mode="phase_select")
        layer.step(1.0, {spec.tls_id: 2}, 1.0, mode="phase_select")
        layer.step(2.0, {spec.tls_id: 2}, 1.0, mode="phase_select")
        layer.step(3.0, {spec.tls_id: 2}, 1.0, mode="phase_select")      # yellow done -> all-red
        assert state.transition_display_phase == SafetyLayer._SYNTHETIC_ALL_RED
        for t in (4.0, 5.0, 6.0):
            layer.step(t, {spec.tls_id: 2}, 1.0, mode="phase_select")
            assert state.yellow_pending, "all-red must hold for the full 4 s"
        layer.step(7.0, {spec.tls_id: 2}, 1.0, mode="phase_select")
        assert not state.yellow_pending and state.current_phase == 4


class TestBridgeYellowNeverSkipped:
    @patch("src.environment.safety.traci")
    def test_residual_g_link_goes_through_yellow_before_red(self, mock_traci):
        """rl-core-001: an authored yellow like 'rGgy'-style keeps a g link green;
        the bridge must yellow EVERY green link of the displayed state first."""
        phases = [
            PhaseSpec(index=0, state="GGrrrrrg", duration=30.0, min_dur=5.0, max_dur=60.0),
            PhaseSpec(index=1, state="yyrrrrrg", duration=3.0),
            PhaseSpec(index=2, state="rrGGrrrr", duration=30.0, min_dur=5.0, max_dur=60.0),
            PhaseSpec(index=3, state="rryyrrrr", duration=3.0),
            PhaseSpec(index=4, state="rrrrGGrr", duration=30.0, min_dur=5.0, max_dur=60.0),
            PhaseSpec(index=5, state="rrrryyrr", duration=3.0),
        ]
        spec = TLSSpec("T", "0", "actuated", "T", phases, ["l"] * 8)
        state = TLSRuntimeState(tls_id="T")
        layer = _layer(spec, state)
        written = []
        mock_traci.trafficlight.setRedYellowGreenState.side_effect = lambda t, s: written.append(s)
        state.displayed_state = "GGrrrrrg"
        state.phase_elapsed = 30.0
        layer.step(0.0, {"T": 2}, 1.0, mode="phase_select")
        assert written == ["yyrrrrry"]                   # the residual 'g' link is yellowed too

    def test_yellow_of_maps_only_green_links(self):
        assert SafetyLayer.yellow_of("GgrsyuG") == "yyrsyuy"


class TestHoldAndOwnership:
    @patch("src.environment.safety.traci")
    def test_missing_action_holds_in_phase_select_mode(self, mock_traci):
        spec = make_three_phase_spec()
        state = make_state(spec.tls_id)
        state.current_phase = 2
        layer = _layer(spec, state)
        state.phase_elapsed = 50.0
        layer.step(0.0, {}, 1.0, mode="phase_select")      # TLS omitted entirely
        assert not state.yellow_pending and state.current_phase == 2

    @patch("src.environment.safety.traci")
    def test_take_over_writes_believed_phase_and_defers_first_tick(self, mock_traci):
        spec = make_three_phase_spec()
        state = TLSRuntimeState(tls_id=spec.tls_id)
        layer = _layer(spec, state)
        mock_traci.trafficlight.getPhase.return_value = 2
        layer.take_over()
        mock_traci.trafficlight.setRedYellowGreenState.assert_called_once_with(spec.tls_id, "rrGGrr")
        assert state.current_phase == 2 and state.displayed_state == "rrGGrr"
        assert layer.displayed_state(spec.tls_id) == "rrGGrr"
        layer.step(0.0, {}, 1.0)
        assert state.phase_elapsed == 0.0                  # first step must not count a not-yet-displayed second
        layer.step(1.0, {}, 1.0)
        assert state.phase_elapsed == 1.0

    @patch("src.environment.safety.traci")
    def test_take_over_falls_back_to_phase_zero_on_traci_error(self, mock_traci):
        import traci.exceptions as texc
        mock_traci.exceptions.TraCIException = texc.TraCIException
        mock_traci.trafficlight.getPhase.side_effect = texc.TraCIException("x")
        spec, state = make_spec(), make_state()
        _layer(spec, state).take_over()
        assert state.current_phase == 0 and state.displayed_state == "GGrr"

    @patch("src.environment.safety.traci")
    def test_displayed_state_during_transition_is_the_clearance_not_the_frozen_green(self, mock_traci):
        spec, state = make_spec(), make_state()
        layer = _layer(spec, state)
        state.phase_elapsed = 20.0
        layer.step(0.0, {spec.tls_id: 1}, 1.0)
        assert state.current_phase == 0                    # frozen green...
        assert layer.displayed_state(spec.tls_id) == "yyrr"  # ...but the junction shows yellow

    @patch("src.environment.safety.traci")
    def test_stable_non_green_start_phase_is_left_after_yellow_duration(self, mock_traci):
        spec, state = make_spec(), make_state()
        state.current_phase = 1                            # authored yellow as the starting phase
        layer = _layer(spec, state, SafetyConfig(yellow_duration=3.0))
        layer.step(0.0, {}, 1.0)
        layer.step(1.0, {}, 1.0)
        assert state.current_phase == 1                    # elapsed 2 < yellow_duration: still held
        layer.step(2.0, {}, 1.0)                           # elapsed 3 == yellow_duration: forced out
        assert state.current_phase == 2


class TestPhaseObsVector:
    def test_one_hot_bounds(self):
        s = TLSRuntimeState(tls_id="t", current_phase=3, phase_elapsed=7.0)
        assert s.phase_obs_vector(4) == [0.0, 0.0, 0.0, 1.0, 7.0]
        s.current_phase = 4                                  # == num_phases: out of range
        assert s.phase_obs_vector(4) == [0.0, 0.0, 0.0, 0.0, 7.0]
        s.current_phase = -1
        assert s.phase_obs_vector(4) == [0.0, 0.0, 0.0, 0.0, 7.0]
        s.current_phase = 0
        assert s.phase_obs_vector(4) == [1.0, 0.0, 0.0, 0.0, 7.0]


class TestLoader:
    def test_loader_sorts_by_tls_id_defaults_min_max_to_duration_and_reads_signal_lanes(self, tmp_path):
        import json

        from src.scenario.tls_metadata import load_tls_specs

        def entry(tid, **kw):
            e = {"tls_id": tid, "program_id": "0", "type": "static",
                 "phases": [{"index": 0, "state": "Gr", "duration": 12.0}], "controlled_lanes": ["a", "b"]}
            e.update(kw)
            return e

        p = tmp_path / "m.json"
        p.write_text(json.dumps([entry("zz"), entry("aa", signal_lanes=[["a"], ["b", "c"]])]))
        specs = load_tls_specs(p)
        assert [s.tls_id for s in specs] == ["aa", "zz"]
        ph = specs[0].phases[0]
        assert (ph.min_dur, ph.max_dur) == (12.0, 12.0)    # default to the nominal duration
        assert specs[0].junction_id == "aa"
        assert specs[0].signal_lanes == [["a"], ["b", "c"]]
        assert specs[0].incoming_lanes_for_signal(1) == ["b", "c"]
        assert specs[0].incoming_lanes_for_signal(7) == []
        # legacy metadata (no signal_lanes) falls back to positional lookup
        assert specs[1].incoming_lanes_for_signal(1) == ["b"]
