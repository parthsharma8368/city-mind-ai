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
        """True if this phase contains at least one Green signal."""
        return "G" in self.state or "g" in self.state

    @property
    def is_yellow(self) -> bool:
        return "y" in self.state.lower() and "G" not in self.state

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
    controlled_lanes: list[str] = field(default_factory=list)

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
        ))

    specs.sort(key=lambda s: s.tls_id)
    return specs
