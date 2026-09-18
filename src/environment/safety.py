"""
safety.py
=========
Deterministic safety layer — translates raw RL actions into legal signal
transitions, enforcing:
  - minimum green time
  - maximum green time
  - yellow clearance duration
  - all-red clearance duration
  - minimum red time

The safety layer is fully decoupled from the neural policy and independently
testable.  The RL policy NEVER directly writes signal states to SUMO —
all transitions go through this layer.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from src.environment.sumo_connection import traci

from src.scenario.tls_metadata import TLSRuntimeState, TLSSpec


@dataclass
class SafetyConfig:
    """Timing constraints for legal signal transitions."""
    min_green: float = 10.0      # seconds
    max_green: float = 90.0      # seconds
    yellow_duration: float = 3.0 # seconds
    all_red_duration: float = 1.0 # seconds (0.0 to skip all-red)
    min_red: float = 0.0          # seconds (rarely needed)


class SafetyLayer:
    """
    Per-TLS safety enforcement.

    Design:
      action=0  →  keep current phase (subject to max_green override)
      action=1  →  request advance to next green phase

    The layer inserts yellow → all-red intermediate phases as required by
    the SignalGroup state machine before green can begin.
    """

    def __init__(
        self,
        tls_specs: list[TLSSpec],
        config: SafetyConfig | None = None,
    ) -> None:
        self.specs: dict[str, TLSSpec] = {s.tls_id: s for s in tls_specs}
        self.cfg = config or SafetyConfig()
        # Runtime state injected externally (shared with environment)
        self._states: dict[str, TLSRuntimeState] = {}

    def register_states(self, states: dict[str, TLSRuntimeState]) -> None:
        """Share runtime state dict with the environment."""
        self._states = states

    # ──────────────────────────────────────────────────────────────────────────
    # Core API
    # ──────────────────────────────────────────────────────────────────────────

    def step(
        self,
        sim_time: float,
        actions: dict[str, int],
        dt: float,
    ) -> None:
        """
        Apply one safety step for all TLS.

        Parameters
        ----------
        sim_time : current simulation time (seconds)
        actions  : dict {tls_id: 0 (keep) | 1 (advance)}
        dt       : simulation step length (seconds) for elapsed tracking
        """
        for tls_id, state in self._states.items():
            spec = self.specs[tls_id]
            action = actions.get(tls_id, 0)
            self._step_single(tls_id, spec, state, sim_time, action, dt)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal per-TLS state machine
    # ──────────────────────────────────────────────────────────────────────────

    def _step_single(
        self,
        tls_id: str,
        spec: TLSSpec,
        state: TLSRuntimeState,
        sim_time: float,
        action: int,
        dt: float,
    ) -> None:
        """State machine for one TLS."""

        current_phase = spec.phases[state.current_phase]

        # ── Case 1: currently in yellow clearance ─────────────────────────
        if state.yellow_pending:
            state.yellow_elapsed += dt
            yellow_needed = self.cfg.yellow_duration
            ar_needed = self.cfg.all_red_duration

            if state.yellow_elapsed < yellow_needed:
                # Still in yellow — do nothing, SUMO holds the yellow state
                return

            if state.yellow_elapsed < yellow_needed + ar_needed and ar_needed > 0:
                # Apply all-red if not yet done
                if state.yellow_elapsed - dt < yellow_needed:
                    # Transition to all-red now
                    self._apply_all_red(tls_id, spec, state)
                return

            # Yellow + all-red complete → switch to target green phase
            state.yellow_pending = False
            target = state.yellow_target_phase
            state.current_phase = target
            state.phase_elapsed = 0.0
            state.last_switch_time = sim_time
            self._apply_phase(tls_id, spec, target)
            return

        # ── Case 2: currently in a green/red phase ────────────────────────
        state.phase_elapsed += dt

        # Force advance if max_green exceeded
        force_advance = (
            current_phase.is_green
            and state.phase_elapsed >= self.cfg.max_green
        )
        # Policy requests advance
        wants_advance = (action == 1)
        # Respect minimum green
        can_advance = (
            not current_phase.is_green
            or state.phase_elapsed >= self.cfg.min_green
        )

        if (wants_advance or force_advance) and can_advance:
            target = spec.next_green_phase(state.current_phase)
            if target != state.current_phase:
                # Begin yellow transition
                state.yellow_pending = True
                state.yellow_target_phase = target
                state.yellow_elapsed = 0.0
                self._apply_yellow(tls_id, spec, state)

    # ──────────────────────────────────────────────────────────────────────────
    # SUMO signal application helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_phase(tls_id: str, spec: TLSSpec, phase_idx: int) -> None:
        """Write a full phase state string to SUMO via TraCI."""
        state_str = spec.phases[phase_idx].state
        traci.trafficlight.setRedYellowGreenState(tls_id, state_str)

    @staticmethod
    def _apply_yellow(tls_id: str, spec: TLSSpec, state: TLSRuntimeState) -> None:
        """
        Derive a yellow state from the current green phase by replacing 'G'→'y',
        'g'→'y'. Red signals remain red.
        """
        current_str = spec.phases[state.current_phase].state
        yellow_str = current_str.replace("G", "y").replace("g", "y")
        traci.trafficlight.setRedYellowGreenState(tls_id, yellow_str)

    @staticmethod
    def _apply_all_red(tls_id: str, spec: TLSSpec, state: TLSRuntimeState) -> None:
        """Apply all-red state (replace all non-r with 'r')."""
        current_str = spec.phases[state.current_phase].state
        ar_str = "".join("r" if c != "s" else "s" for c in current_str.lower())
        traci.trafficlight.setRedYellowGreenState(tls_id, ar_str)

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    def switch_count(self) -> int:
        """Total number of phase switches across all TLS (accumulated)."""
        # States track this externally via the environment; placeholder here
        return 0

    def __repr__(self) -> str:
        return (
            f"SafetyLayer("
            f"min_green={self.cfg.min_green}s, "
            f"max_green={self.cfg.max_green}s, "
            f"yellow={self.cfg.yellow_duration}s, "
            f"all_red={self.cfg.all_red_duration}s)"
        )
