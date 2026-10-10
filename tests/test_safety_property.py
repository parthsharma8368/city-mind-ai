"""Property/fuzz tests of the SafetyLayer over EVERY real TLS in tls_metadata.json.

Random actions for thousands of steps, both action modes, every possible starting
phase. Asserts the legality invariants the module docstring promises:

  I1  every state written to SUMO is an authored phase state, or contains no
      G/g at all (synthesized yellow / all-red)  -> no conflicting greens;
  I2  no link goes green -> red in one write unless the (prev, next) pair is an
      authored consecutive pair of the TLS's own program  -> always via yellow;
  I3  min green and max green are respected for every stable authored green;
  I4  synthesized yellow lasts >= yellow_duration, synthesized all-red
      >= all_red_duration;
  I5  no stuck state: a hold-only TLS with >= 2 greens still leaves its green
      (max_green force-advance) and never stays mid-transition forever.
"""
from __future__ import annotations

import random
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import traci.exceptions as texc

from src.environment.safety import SafetyConfig, SafetyLayer
from src.scenario.tls_metadata import TLSRuntimeState

STEPS = 2500


def _is_g(c: str) -> bool:
    return c in "Gg"


def _authored_pairs(spec) -> set[tuple[str, str]]:
    n = len(spec.phases)
    return {(spec.phases[i].state, spec.phases[(i + 1) % n].state) for i in range(n)}


class _Recorder:
    def __init__(self, start_phase: dict[str, int]):
        self.t = 0.0
        self.writes: dict[str, list[tuple[float, str]]] = {}
        self.start_phase = start_phase
        self.exceptions = SimpleNamespace(TraCIException=texc.TraCIException)
        rec = self

        class _TL:
            @staticmethod
            def getPhase(tid):
                return rec.start_phase[tid]

            @staticmethod
            def setRedYellowGreenState(tid, s):
                rec.writes.setdefault(tid, []).append((rec.t, s))

        self.trafficlight = _TL


def _run(specs, start_phase, actions_fn, steps, mode, cfg=None):
    cfg = cfg or SafetyConfig()
    rec = _Recorder(start_phase)
    states = {s.tls_id: TLSRuntimeState(tls_id=s.tls_id) for s in specs}
    layer = SafetyLayer(specs, cfg)
    layer.register_states(states)
    with patch("src.environment.safety.traci", rec):
        layer.take_over()
        for _ in range(steps):
            acts = actions_fn(rec.t)
            layer.step(rec.t, acts, 1.0, mode=mode)
            rec.t += 1.0
    return layer, states, rec


def _check_invariants(spec, writes, cfg, steps, multi_green_should_leave: bool):
    authored = {p.state for p in spec.phases}
    pairs = _authored_pairs(spec)
    prev_state = None
    for _t, s in writes:
        # I1: authored, or no green at all
        assert s in authored or not any(_is_g(c) for c in s), f"{spec.tls_id}: illegal state {s!r}"
        assert len(s) == len(spec.phases[0].state)
        # I2: green -> red only across an authored pair
        if prev_state is not None and prev_state != s:
            direct_g_to_r = any(_is_g(a) and b == "r" for a, b in zip(prev_state, s))
            if direct_g_to_r:
                assert (prev_state, s) in pairs, (
                    f"{spec.tls_id}: green->red without yellow {prev_state!r} -> {s!r}"
                )
        prev_state = s

    # I3/I4: timing from the write timeline
    greens = {p.state: p for p in spec.phases if p.is_green}
    for k in range(len(writes) - 1):
        t0, s0 = writes[k]
        t1, s1 = writes[k + 1]
        if s0 in greens and len(greens) >= 2:
            ph = greens[s0]
            min_g = ph.min_dur if 0 < ph.min_dur < ph.duration else cfg.min_green
            max_g = ph.max_dur if ph.max_dur > ph.duration else cfg.max_green
            assert t1 - t0 >= min_g, f"{spec.tls_id}: left green {s0!r} after {t1 - t0}s < min {min_g}"
            assert t1 - t0 <= max(max_g, min_g) + 1.0, (
                f"{spec.tls_id}: held green {s0!r} for {t1 - t0}s > max {max_g}"
            )
        if s0 not in authored and not any(_is_g(c) for c in s0):
            if "y" in s0:
                assert t1 - t0 >= cfg.yellow_duration, f"{spec.tls_id}: synthetic yellow {t1 - t0}s"
            elif set(s0) == {"r"}:
                assert t1 - t0 >= cfg.all_red_duration, f"{spec.tls_id}: synthetic all-red {t1 - t0}s"


@pytest.mark.parametrize("mode", ["binary", "phase_select"])
def test_random_actions_keep_every_real_tls_legal(real_tls_specs, mode):
    cfg = SafetyConfig()
    rng = random.Random(1234)
    specs = real_tls_specs
    assert len(specs) == 98
    start = {s.tls_id: rng.randrange(len(s.phases)) for s in specs}
    hi = 2 if mode == "binary" else 6

    def acts(_t):
        return {s.tls_id: rng.randrange(hi) for s in specs}

    layer, states, rec = _run(specs, start, acts, STEPS, mode, cfg)
    n_switchers = 0
    for spec in specs:
        _check_invariants(spec, rec.writes[spec.tls_id], cfg, STEPS, True)
        if spec.green_phase_count >= 2 and len(rec.writes[spec.tls_id]) > 1:
            n_switchers += 1
        st = states[spec.tls_id]
        assert st.displayed_state == rec.writes[spec.tls_id][-1][1]
    # random actions must actually exercise transitions on (almost) every multi-green TLS
    multi = sum(1 for s in specs if s.green_phase_count >= 2)
    assert n_switchers >= multi - 2


@pytest.mark.parametrize("mode", ["binary", "phase_select"])
def test_every_possible_start_phase_is_legal_and_never_stuck(real_tls_specs, mode):
    """Start each TLS in EVERY one of its phases (yellow/all-red included), hold
    forever (binary action 0 / phase_select: missing action = hold): the layer
    must leave non-green starts and force-advance any green at max_green."""
    cfg = SafetyConfig()
    n_phases_max = max(len(s.phases) for s in real_tls_specs)
    for start_idx in range(n_phases_max):
        specs = [s for s in real_tls_specs if start_idx < len(s.phases)]
        start = {s.tls_id: start_idx for s in specs}
        bound = 400  # > any max_green + transition chain in the real data
        layer, states, rec = _run(specs, start, lambda _t: {}, bound, mode, cfg)
        for spec in specs:
            writes = rec.writes[spec.tls_id]
            _check_invariants(spec, writes, cfg, bound, False)
            if spec.green_phase_count >= 2:
                assert len({s for _, s in writes}) > 1, (
                    f"{spec.tls_id} start={start_idx} mode={mode}: never left its start phase (stuck)"
                )
            assert not (states[spec.tls_id].yellow_pending and states[spec.tls_id].yellow_elapsed > 30), (
                f"{spec.tls_id}: stuck mid-transition"
            )


def test_missing_action_in_phase_select_holds_not_first_green(real_tls_specs):
    """rl-core-013a: an omitted TLS used to be treated as action 0 (= 'serve my
    first green'), forcing every such TLS to switch."""
    specs = [s for s in real_tls_specs if s.green_phase_count >= 3][:20]
    start = {s.tls_id: s.green_phase_indices[1] for s in specs}   # NOT the first green
    layer, states, rec = _run(specs, start, lambda _t: {}, 30, "phase_select", SafetyConfig(max_green=60.0))
    left = [s.tls_id for s in specs if len({x for _, x in rec.writes[s.tls_id]}) > 1]
    # Only TLS whose own authored max_dur < 30 s may legitimately leave via force-advance.
    for tid in left:
        spec = next(s for s in specs if s.tls_id == tid)
        ph = spec.phases[start[tid]]
        max_g = ph.max_dur if ph.max_dur > ph.duration else 60.0
        assert max_g <= 30.0, f"{tid} switched on a missing action although max_green={max_g}"
