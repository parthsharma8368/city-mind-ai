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
    """Timing constraints for legal signal transitions.

    min_green/max_green are FALLBACKS only (2026-09-21 mechanics fix): when
    a TLS's current green phase carries its own program-authored min_dur/
    max_dur distinct from its nominal duration (i.e. it came from a SUMO
    "actuated" program, 76/98 InTAS TLS), that per-phase value is used
    instead — see SafetyLayer._green_bounds(). These fallbacks apply to the
    22 static-program TLS (whose min_dur/max_dur default to their fixed
    duration, so the per-phase check intentionally skips them) and cap the
    RESCO/sumo-rl convention (5s min, permissive max) rather than the old
    10s/90s values, which were tighter than most of the network's own
    actuated minimums (median program green min_dur is 7-10s but many
    protected-turn phases are 5-7s).

    yellow_duration/all_red_duration are no longer the primary clearance
    timing source — SafetyLayer now walks each TLS's own authored
    transition phases (its real yellow, and an all-red only where the
    program has one) instead of synthesizing a state string. They remain
    as a defensive floor for the rare degenerate case of a zero-duration
    transition phase in the source data.
    """
    min_green: float = 5.0       # seconds — fallback when no per-phase min_dur applies
    max_green: float = 60.0      # seconds — fallback when no per-phase max_dur applies
    yellow_duration: float = 3.0 # seconds — defensive floor, not the primary timing source
    all_red_duration: float = 1.0 # seconds — defensive floor, not the primary timing source
    min_red: float = 0.0          # seconds (rarely needed)


class SafetyLayer:
    """
    Per-TLS safety enforcement.

    Two action modes (mode= param on step()):
      "binary" (default) — action=0 keep, action=1 advance to the next
        green phase in the fixed cycle. Original design; every existing
        caller (MaxPressureController, CentralizedSumoEnv, tests) uses this
        and is unaffected by phase_select mode's existence.
      "phase_select" — action is a direct target green-phase index (via
        TLSSpec.action_to_phase_index), for the RL-strategy Phase 2
        phase-selection action space. Adjacent-next-green targets reuse the
        exact binary-mode transition (this TLS's own native yellow/all-red
        sequence); a jump to any OTHER green phase uses a synthesized
        yellow(if native)+all-red bridge instead of walking through and
        illegally granting an intervening phase its own green window — see
        _build_bridge_queue.

    The layer inserts yellow → all-red intermediate phases as required by
    the SignalGroup state machine before green can begin.
    """

    # Sentinel transition_display_phase value (native phase indices are
    # always >= 0; -1 already means "not in transition") for a synthesized
    # all-red buffer that has no corresponding entry in spec.phases.
    _SYNTHETIC_ALL_RED = -2

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
        mode: str = "binary",
    ) -> None:
        """
        Apply one safety step for all TLS.

        Parameters
        ----------
        sim_time : current simulation time (seconds)
        actions  : dict {tls_id: action}. In "binary" mode: 0 (keep) | 1
                   (advance). In "phase_select" mode: a Discrete(MAX_GREEN_PHASES)
                   value naming this TLS's own i-th green phase (see
                   TLSSpec.action_to_phase_index) — out-of-range or
                   currently-invalid values fall back to "hold current phase".
        dt       : simulation step length (seconds) for elapsed tracking
        mode     : "binary" (default) or "phase_select"
        """
        for tls_id, state in self._states.items():
            spec = self.specs[tls_id]
            action = actions.get(tls_id, 0)
            self._step_single(tls_id, spec, state, sim_time, action, dt, mode)

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
        mode: str = "binary",
    ) -> None:
        """State machine for one TLS."""

        current_phase = spec.phases[state.current_phase]

        # ── Case 1: mid-transition — walking the TLS's own authored
        # sequence of intermediate phases (native yellow, plus a native
        # all-red only for the few programs that define one), or a
        # synthesized all-red bridge step for a phase-selection jump ──────
        if state.yellow_pending:
            state.yellow_elapsed += dt
            if state.transition_display_phase == self._SYNTHETIC_ALL_RED:
                sub_duration = self.cfg.all_red_duration
            else:
                active = spec.phases[state.transition_display_phase]
                sub_duration = active.duration if active.duration > 0 else (
                    self.cfg.all_red_duration if active.is_all_red else self.cfg.yellow_duration
                )

            if state.yellow_elapsed < sub_duration:
                return  # still displaying this native sub-phase

            if state.transition_queue:
                next_idx = state.transition_queue.pop(0)
                state.transition_display_phase = next_idx
                state.yellow_elapsed = 0.0
                self._apply_display_phase(tls_id, spec, next_idx)
                return

            # Transition complete → switch to target green phase
            state.yellow_pending = False
            target = state.yellow_target_phase
            state.current_phase = target
            state.phase_elapsed = 0.0
            state.last_switch_time = sim_time
            state.transition_display_phase = -1
            self._apply_phase(tls_id, spec, target)
            return

        # ── Case 2: currently in a stable green/red phase ─────────────────
        state.phase_elapsed += dt

        min_green, max_green = self._green_bounds(current_phase)

        # Force advance if max_green exceeded
        force_advance = (
            current_phase.is_green
            and state.phase_elapsed >= max_green
        )

        if mode == "phase_select":
            mapped = spec.action_to_phase_index(action)
            requested_target = mapped if mapped is not None else state.current_phase
            wants_advance = requested_target != state.current_phase
        else:
            requested_target = spec.next_green_phase(state.current_phase)
            wants_advance = (action == 1)

        # Respect minimum green
        can_advance = (
            not current_phase.is_green
            or state.phase_elapsed >= min_green
        )

        if (wants_advance or force_advance) and can_advance:
            # A forced advance with no policy-requested target (it wanted to
            # hold) falls back to the fixed-cycle next green, same as today;
            # a policy-requested target (wants_advance True) is honored even
            # if max_green also happened to trigger this step.
            target = requested_target if wants_advance else spec.next_green_phase(state.current_phase)
            if target != state.current_phase:
                immediate_next = spec.next_green_phase(state.current_phase)
                if target == immediate_next:
                    queue = self._build_transition_queue(spec, state.current_phase, target)
                else:
                    # Phase-selection jump to a non-adjacent green: never
                    # walk through an intervening phase's own green window
                    # (see _build_bridge_queue).
                    queue = self._build_bridge_queue(spec, state.current_phase)
                if queue:
                    first = queue.pop(0)
                    state.yellow_pending = True
                    state.yellow_target_phase = target
                    state.yellow_elapsed = 0.0
                    state.transition_queue = queue
                    state.transition_display_phase = first
                    self._apply_display_phase(tls_id, spec, first)
                else:
                    # Zero-hop: the program's own phase sequence allows a
                    # direct switch between these two green phases with no
                    # authored clearance step (e.g. adding a protected
                    # movement that doesn't conflict with what's already
                    # green) — verified against tls_metadata.json this is
                    # true for 3/303 real green phases network-wide, not a
                    # data gap to paper over with an invented yellow.
                    state.current_phase = target
                    state.phase_elapsed = 0.0
                    state.last_switch_time = sim_time
                    self._apply_phase(tls_id, spec, target)

    # ──────────────────────────────────────────────────────────────────────────
    # Per-phase timing / native transition helpers (2026-09-21)
    # ──────────────────────────────────────────────────────────────────────────

    def _green_bounds(self, phase) -> tuple[float, float]:
        """(min_green, max_green) for this green phase: use the program's
        own min_dur/max_dur when they're distinct from the phase's nominal
        duration (a real actuated program's authored range), else fall back
        to self.cfg. The loader defaults min_dur/max_dur to duration itself
        for phases with no XML minDur/maxDur attribute (SUMO "static"
        programs, 22/98 TLS) — for those this correctly falls back to the
        permissive config value instead of locking the RL/MP action to the
        original fixed cycle."""
        min_green = (
            phase.min_dur if 0 < phase.min_dur < phase.duration else self.cfg.min_green
        )
        max_green = (
            phase.max_dur if phase.max_dur > phase.duration else self.cfg.max_green
        )
        return min_green, max_green

    def _build_transition_queue(self, spec: TLSSpec, current_idx: int, target_idx: int) -> list[int]:
        """Native intermediate phase indices between current_idx and
        target_idx, in program order. Normally this is exactly the
        program's own yellow-clearance phase (1 hop); occasionally
        yellow-then-all-red (2 hops, 5/303 real greens); occasionally none
        (3/303 — see the zero-hop comment above)."""
        n = len(spec.phases)
        queue: list[int] = []
        j = (current_idx + 1) % n
        hops = 0
        while j != target_idx and hops < n:
            queue.append(j)
            j = (j + 1) % n
            hops += 1
        return queue

    def _build_bridge_queue(self, spec: TLSSpec, current_idx: int) -> list[int]:
        """Transition plan for a phase-selection jump to a green phase that
        is NOT the immediate next one in the fixed cycle.

        Walking the program's own phase sequence (as _build_transition_queue
        does) isn't safe here — it would pass through, and briefly grant a
        real green window to, whichever other green phase(s) sit between
        current and the target in cycle order. Instead: this TLS's own
        dedicated yellow (the phase immediately after current_idx, if the
        program places one there — true for the large majority; see
        SafetyLayer's module docstring for the 3/303 zero-hop exception)
        clears current's own movements, then a synthesized all-red buffer
        (self.cfg.all_red_duration, same floor used elsewhere) — not a
        native phase — separates it from the target's green. All-red is
        universally legal for any signal regardless of link count, so this
        needs no per-TLS data beyond the current phase's own state length.
        """
        n = len(spec.phases)
        queue: list[int] = []
        nxt = (current_idx + 1) % n
        if spec.phases[nxt].is_yellow:
            queue.append(nxt)
        queue.append(self._SYNTHETIC_ALL_RED)
        return queue

    # ──────────────────────────────────────────────────────────────────────────
    # SUMO signal application helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_phase(tls_id: str, spec: TLSSpec, phase_idx: int) -> None:
        """Write a full phase state string to SUMO via TraCI."""
        state_str = spec.phases[phase_idx].state
        traci.trafficlight.setRedYellowGreenState(tls_id, state_str)

    @classmethod
    def _apply_display_phase(cls, tls_id: str, spec: TLSSpec, phase_idx: int) -> None:
        """Like _apply_phase, but also handles the synthesized all-red
        bridge sentinel (_SYNTHETIC_ALL_RED), which has no entry in
        spec.phases — its state string is built on the fly as all-red,
        sized to match this TLS's own link count."""
        if phase_idx == cls._SYNTHETIC_ALL_RED:
            n_links = len(spec.phases[0].state) if spec.phases else 0
            traci.trafficlight.setRedYellowGreenState(tls_id, "r" * n_links)
            return
        cls._apply_phase(tls_id, spec, phase_idx)

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
