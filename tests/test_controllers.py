"""Unit tests for baseline traffic signal controllers."""

import pytest
from unittest.mock import patch

from src.controllers.base import ControllerResult
from src.controllers.fixed_time import FixedTimeController
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








