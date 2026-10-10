"""
tls_metadata.py
===============
Runtime TLS state dataclasses used by the environment during simulation.
Loaded from artifacts/scenario/tls_metadata.json after inspect_intas.py runs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# RL-strategy Phase 2 (phase-selection action space): the largest number of
# green phases any InTAS TLS has, verified against tls_metadata.json — 2
# signals have 1, 18 have 2, 50 have 3, 26 have 4, 1 has 5, 1 has 6. The
# parameter-shared Discrete(MAX_GREEN_PHASES) action space and the per-phase
# pressure observation are both sized off this constant; a TLS with fewer
# phases has its extra action slots masked out (see TLSSpec.action_mask).
MAX_GREEN_PHASES = 6


@dataclass
class PhaseSpec:
    """Static definition of one traffic-light phase."""
    index: int
    state: str       # SUMO signal state string, e.g. "GGrrGGrr"
    duration: float  # default duration (seconds)
    min_dur: float = 0.0
    max_dur: float = 0.0

    @property
    def is_green(self) -> bool:
        """True if this is a stable green phase the safety layer may hold or
        target — i.e. at least one green signal AND no yellow signal.

        German actuated programs commonly retain a permissive movement
        (lowercase g) through their own yellow-clearance phase, e.g.
        "yyggrrrryyygrrrr" or "rryyyyg". The old definition ("G" or "g"
        present) counted these transitional phases as legal cycle targets
        too — measured on 2026-09-21: 85 of 388 phases the old definition
        called "green" across 65/98 InTAS TLS actually contained a yellow
        signal. Landing on one of them as a target forced a 10s min_green
        hold on what should have been a ~3-5s clearance, and was the
        single largest contributor to the ~1s gap between every trained
        policy (~4.0-4.15s) and the SUMO-actuated baseline (3.07s) found
        by the RL-strategy ceiling test the same day. Excluding yellow
        drops the true green-target count to 303/608.
        """
        s_lower = self.state.lower()
        return ("g" in s_lower) and ("y" not in s_lower)

    @property
    def is_yellow(self) -> bool:
        """True if this phase contains a yellow signal — a clearance /
        transition phase, which may still show a residual green movement
        (see is_green's docstring). Any phase with a yellow signal is a
        transition regardless of what else is present, so this and
        is_green are mutually exclusive by construction."""
        return "y" in self.state.lower()

    @property
    def is_all_red(self) -> bool:
        s = self.state.lower()
        return all(c in ("r", "s") for c in s)


@dataclass
class TLSSpec:
    """Static specification of one traffic-light system (from inspection)."""
    tls_id: str
    program_id: str
    type: str
    junction_id: str
    phases: list[PhaseSpec] = field(default_factory=list)
    # traci.trafficlight.getControlledLanes(): flattened PER LINK, de-duplication
    # NOT applied -- length != len(state) for some TLS and index i is NOT
    # signal i's lane. Use only for lane SETS (observation features, rewards).
    controlled_lanes: list[str] = field(default_factory=list)
    # rl-core-007: signal_lanes[i] = incoming lane(s) of signal/link index i of
    # the state strings (from traci.trafficlight.getControlledLinks). Empty for
    # metadata written before this field existed (legacy fallback below).
    signal_lanes: list[list[str]] = field(default_factory=list)

    def incoming_lanes_for_signal(self, i: int) -> list[str]:
        """Distinct incoming lanes controlled by signal index ``i`` of a phase
        state string. Falls back to the legacy (buggy for 5 TLS) positional
        lookup only when the metadata has no ``signal_lanes``."""
        if self.signal_lanes:
            if 0 <= i < len(self.signal_lanes):
                return list(dict.fromkeys(self.signal_lanes[i]))
            return []
        if 0 <= i < len(self.controlled_lanes):
            return [self.controlled_lanes[i]]
        return []

    @property
    def green_phase_count(self) -> int:
        return sum(1 for p in self.phases if p.is_green)

    @property
    def yellow_phase_indices(self) -> list[int]:
        return [p.index for p in self.phases if p.is_yellow]

    def next_green_phase(self, current_phase: int) -> int:
        """Return the index of the next green phase after current_phase."""
        n = len(self.phases)
        for offset in range(1, n + 1):
            candidate = (current_phase + offset) % n
            if self.phases[candidate].is_green:
                return candidate
        return current_phase  # fallback: stay

    @property
    def green_phase_indices(self) -> list[int]:
        """Native phase indices of this TLS's own green phases, in program
        (cycle) order. Position i in this list is what a parameter-shared
        phase-selection action value `i` refers to — the RL action space is
        Discrete(MAX_GREEN_PHASES) shared across every TLS regardless of how
        many green phases it actually has, so actions must be expressed as
        "my i-th own green phase," not as a raw phase index (which isn't
        comparable across TLS with different phase layouts)."""
        return [p.index for p in self.phases if p.is_green]

    def action_to_phase_index(self, action: int) -> int | None:
        """Map a Discrete(MAX_GREEN_PHASES) action value to this TLS's own
        native phase index, or None if the action doesn't correspond to one
        of this TLS's real green phases (out of range, or masked-out slot —
        the safety layer treats None as "hold current phase")."""
        indices = self.green_phase_indices
        return indices[action] if 0 <= action < len(indices) else None

    def action_mask(self, max_actions: int = MAX_GREEN_PHASES) -> list[bool]:
        """Boolean mask of length max_actions: True for action values that
        map to one of this TLS's real green phases. Used by MaskablePPO so
        the policy never has probability mass on a phase this intersection
        doesn't have."""
        n_green = len(self.green_phase_indices)
        return [i < n_green for i in range(max_actions)]


@dataclass
class TLSRuntimeState:
    """Mutable runtime state for one TLS during simulation."""
    tls_id: str
    current_phase: int = 0
    phase_elapsed: float = 0.0      # seconds in current phase
    last_switch_time: float = -999.0
    yellow_pending: bool = False
    yellow_target_phase: int = -1
    yellow_elapsed: float = 0.0
    # Which native phase index is currently being displayed to SUMO while
    # yellow_pending is True (2026-09-21 mechanics fix). Distinct from
    # current_phase, which stays frozen at the pre-switch green phase for
    # the whole transition — exactly as before this fix — so observation
    # features, reward switch-counting, and metrics all still see exactly
    # one phase change per logical switch, whether the program's own
    # sequence needs zero, one, or two intermediate clearance phases.
    transition_display_phase: int = -1
    # Items are native phase indices (>=0) or SafetyLayer synthetic sentinels (<0).
    transition_queue: list[int] = field(default_factory=list)
    # The exact signal state string most recently written to SUMO for this TLS
    # (rl-core-008): what the junction actually DISPLAYS, including yellow /
    # all-red clearance while current_phase stays frozen at the old green.
    # Empty until the safety layer first writes (or takes over) the TLS.
    displayed_state: str = ""
    # Set by SafetyLayer.take_over(): the believed phase was written at the SAME
    # sim time as the first layer.step(), so that first step must not count a
    # second of dwell that has not been displayed yet (otherwise the initial
    # green would be shown 1 s less than min_green).
    defer_first_tick: bool = False

    # Collected per-step (reset each step)
    lane_queue: list[float] = field(default_factory=list)    # halting vehicles per lane
    lane_vehicles: list[float] = field(default_factory=list)
    lane_speed: list[float] = field(default_factory=list)
    lane_density: list[float] = field(default_factory=list)

    def phase_obs_vector(self, num_phases: int) -> list[float]:
        """One-hot encode current phase + elapsed time."""
        one_hot = [0.0] * num_phases
        if 0 <= self.current_phase < num_phases:
            one_hot[self.current_phase] = 1.0
        return one_hot + [self.phase_elapsed]


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_tls_specs(json_path: Path) -> list[TLSSpec]:
    """
    Load TLS specifications from the JSON artifact produced by inspect_intas.py.
    Returns a list of TLSSpec objects sorted by tls_id for reproducibility.
    """
    with open(json_path) as f:
        raw = json.load(f)

    specs: list[TLSSpec] = []
    for entry in raw:
        phases = [
            PhaseSpec(
                index=ph["index"],
                state=ph["state"],
                duration=ph["duration"],
                min_dur=ph.get("min_dur", ph["duration"]),
                max_dur=ph.get("max_dur", ph["duration"]),
            )
            for ph in entry.get("phases", [])
        ]
        specs.append(TLSSpec(
            tls_id=entry["tls_id"],
            program_id=entry["program_id"],
            type=entry.get("type", "static"),
            junction_id=entry.get("junction_id", entry["tls_id"]),
            phases=phases,
            controlled_lanes=entry.get("controlled_lanes", []),
            signal_lanes=[list(x) for x in entry.get("signal_lanes", [])],
        ))

    specs.sort(key=lambda s: s.tls_id)
    return specs
