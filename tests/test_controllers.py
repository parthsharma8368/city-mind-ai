"""Unit tests for baseline traffic signal controllers."""

import pytest
from unittest.mock import patch

from src.controllers.base import ControllerResult
from src.controllers.fixed_time import FixedTimeController
from src.controllers.max_pressure import MaxPressureController
from src.scenario.tls_metadata import TLSSpec, PhaseSpec, TLSRuntimeState


def _fake_with_static(tmp_path, n_tls=2, end_time=40.0):
    from tests.fake_traci import FakeTraci

    tls = {f"T{i}": ["GGrr", "yyrr", "rrGG", "rryy"] for i in range(n_tls)}
    return FakeTraci(tls, end_time=end_time, begin=0.0)


def _metadata_for(tmp_path, fake):
    import json

    entries = []
    for tid, states in fake.tls_native.items():
        entries.append({
            "tls_id": tid, "program_id": "0", "type": "actuated",
            "phases": [{"index": i, "state": st, "duration": 10.0, "min_dur": 3.0, "max_dur": 40.0}
                       if st.lower().count("g") else
                       {"index": i, "state": st, "duration": 3.0} for i, st in enumerate(states)],
            "controlled_lanes": ["l0", "l1", "l0", "l1"],
            "signal_lanes": [["l0"], ["l0"], ["l1"], ["l1"]],
        })
    p = tmp_path / "meta.json"
    p.write_text(json.dumps(entries))
    return str(p)


def test_fixed_time_selects_static_program_for_every_tls_and_records_command(monkeypatch, tmp_path):
    from tests.fake_traci import install

    fake = install(monkeypatch, _fake_with_static(tmp_path))
    add = tmp_path / "static.add.xml"
    add.write_text("<additional/>")
    cfg = tmp_path / "s.sumocfg"
    cfg.write_text("<configuration/>")
    c = FixedTimeController(static_add_file=str(add), sumo_extra_args=["--begin", "0", "--end", "40"],
                            episode_duration=40)
    res = c.run_episode(str(cfg), "unused", seed=5)
    assert sorted(fake.programs_set) == [("T0", "static_fixed"), ("T1", "static_fixed")]
    assert fake.closed == 1
    assert "--seed" in res.sumo_command and res.sumo_command[res.sumo_command.index("--seed") + 1] == "5"
    assert str(add) in res.sumo_command[res.sumo_command.index("--additional-files") + 1]
    assert res.metrics_schema_version == 2
    assert res.total_steps == 8                       # 40 s sampled every 5 s: t = 5, 10, ..., 40
    assert res.episode_duration == 35.0


def test_fixed_time_raises_and_still_closes_sumo_when_a_tls_lacks_the_static_program(monkeypatch, tmp_path):
    from tests.fake_traci import install

    fake = _fake_with_static(tmp_path)
    fake.fail_on["setProgram_missing"] = ("T1",)
    install(monkeypatch, fake)
    add = tmp_path / "static.add.xml"
    add.write_text("<additional/>")
    cfg = tmp_path / "s.sumocfg"
    cfg.write_text("<configuration/>")
    with pytest.raises(RuntimeError, match="1/2 TLS"):
        FixedTimeController(static_add_file=str(add)).run_episode(str(cfg), "unused")
    assert fake.closed == 1                              # try/finally closed SUMO


def test_sumo_actuated_counts_green_switches_and_samples_on_the_decision_grid(monkeypatch, tmp_path):
    from src.controllers.sumo_actuated import SumoActuatedController
    from tests.fake_traci import install

    fake = install(monkeypatch, _fake_with_static(tmp_path, end_time=24.0))
    cfg = tmp_path / "s.sumocfg"
    cfg.write_text("<configuration/>")
    traj = tmp_path / "t.parquet"
    res = SumoActuatedController(episode_duration=24).run_episode(str(cfg), "unused", trajectory_path=str(traj))
    # native program: phase = (t//2) % 4 -> greens GGrr (t in [0,2)), rrGG ([4,6)), ... every 4 s
    assert res.total_steps == 4                          # t = 5, 10, 15, 20 (24 s ends mid-interval)
    assert res.total_switches > 0                        # no longer hard-coded 0
    assert fake.closed == 1
    import pandas as pd
    df = pd.read_parquet(traj)
    assert sorted(df.time.unique()) == [5.0, 10.0, 15.0, 20.0]


def test_max_pressure_takes_over_at_t0_and_records_displayed_state(monkeypatch, tmp_path):
    from tests.fake_traci import install

    fake = install(monkeypatch, _fake_with_static(tmp_path, end_time=30.0))
    fake.halting = {"l0": 0, "l1": 9}
    meta = _metadata_for(tmp_path, fake)
    cfg = tmp_path / "s.sumocfg"
    cfg.write_text("<configuration/>")
    traj = tmp_path / "t.parquet"
    res = MaxPressureController(episode_duration=30, min_green=3.0).run_episode(
        str(cfg), meta, trajectory_path=str(traj))
    # takeover: both TLS got their believed state written BEFORE the first simulation step
    first_writes = [w for w in fake.written if w[0] == 0.0]
    assert {w[1] for w in first_writes} >= {"T0", "T1"}
    assert res.total_switches >= 1                       # l1 is busy -> advances to the rrGG phase
    import pandas as pd
    df = pd.read_parquet(traj)
    shown = set(df[df.tls_id.notna()].tls_state)
    assert "yyrr" in shown or "rryy" in shown or "rrrr" in shown, (
        "trajectory must show yellow/all-red clearance, not only the frozen green")
    assert fake.closed == 1


def test_controller_loop_closes_sumo_when_traci_dies_mid_episode(monkeypatch, tmp_path):
    from tests.fake_traci import FatalTraCIError, install

    fake = _fake_with_static(tmp_path, end_time=100.0)
    install(monkeypatch, fake)
    cfg = tmp_path / "s.sumocfg"
    cfg.write_text("<configuration/>")
    fake.fail_on["simulationStep"] = FatalTraCIError("connection closed")
    from src.controllers.sumo_actuated import SumoActuatedController

    with pytest.raises(FatalTraCIError):
        SumoActuatedController().run_episode(str(cfg), "unused")
    assert fake.closed == 1


def test_controller_result_carries_provenance_defaults():
    r = ControllerResult(controller="x", seed=1, demand_id="1.0x", total_steps=1, episode_duration=5.0)
    assert r.metrics_schema_version == 2 and r.sumo_command == []


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


@patch("src.controllers.max_pressure.traci")
def test_max_pressure_phase_pressure_uses_signal_lanes_mapping(mock_traci):
    """rl-core-007: signal index -> its own incoming lane, not controlled_lanes[i]."""
    import traci.exceptions as texc

    mock_traci.exceptions.TraCIException = texc.TraCIException
    halting = {"A": 1.0, "B": 20.0, "C": 300.0}
    mock_traci.lane.getLastStepHaltingNumber.side_effect = lambda lane: halting[lane]
    mock_traci.lane.getLinks.return_value = []
    spec = TLSSpec("t", "0", "static", "t", [PhaseSpec(0, "rrG", 10.0)],
                   controlled_lanes=["A", "A", "B", "C"], signal_lanes=[["A"], ["B"], ["C"]])
    assert MaxPressureController()._phase_pressure(spec, spec.phases[0], {}) == 300.0
