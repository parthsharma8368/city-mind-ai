"""EvaluationRunner persistence: overwrite guard, window-aware paths, atomic/combined parquet, provenance.

No SUMO is started: the simulation methods are not called.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.controllers.base import ControllerResult
from src.evaluation import provenance as prov
from src.evaluation.evaluate import EvaluationRunner, ResultExistsError
from tests._eval_synth import tripinfo_row, write_scenario, write_tripinfo


def _cfg(tmp_path, **over):
    cfg = {"seeds": [42], "demand_scales": [1.0], "begin_time": 25200, "episode_duration": 3600, **over}
    p = tmp_path / "eval.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


@pytest.fixture()
def runner_factory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sumocfg = write_scenario(tmp_path / "scn")

    def make(**kw):
        eval_cfg = kw.pop("eval_cfg", None) or _cfg(tmp_path)
        return EvaluationRunner(str(sumocfg), "tls.json", eval_config_path=eval_cfg, **kw)
    return make


def _result(ctrl="sumo_actuated", seed=42):
    return ControllerResult(controller=ctrl, seed=seed, demand_id="1.0x", total_steps=5, episode_duration=10.0,
                            total_departed=3, trips_completed=1)


def test_canonical_window_uses_legacy_names_and_default_dir(runner_factory):
    r = runner_factory()
    assert r.canonical and r.output_dir == Path("artifacts/metrics")
    assert r._stem("ippo", 42, "1.0x") == "ippo_seed42_1.0x"


def test_non_canonical_window_never_lands_in_canonical_dir(runner_factory, tmp_path):
    r = runner_factory(eval_cfg=_cfg(tmp_path, begin_time=0, episode_duration=1800))
    assert not r.canonical
    assert r.output_dir == Path("artifacts/metrics/runs/b0_d1800")
    assert r.trajectory_dir == Path("artifacts/trajectories/runs/b0_d1800")
    assert r._stem("ippo", 42, "1.0x") == "ippo_seed42_1.0x_b0_d1800"


def test_explicit_output_dir_wins(runner_factory, tmp_path):
    r = runner_factory(output_dir=str(tmp_path / "custom"))
    assert r.output_dir == tmp_path / "custom"


def test_default_controllers_follow_model_paths(runner_factory):
    assert runner_factory().controllers == ["sumo_actuated", "fixed_time", "max_pressure"]
    r = runner_factory(ippo_phase_select_model_path="m.zip")
    assert r.controllers[-1] == "ippo_phase_select"
    with pytest.raises(ValueError, match="no model path"):
        runner_factory(controllers=["ppo"])
    with pytest.raises(ValueError, match="unknown"):
        runner_factory(controllers=["nope"])


def _save(runner, tmp_path, ctrl="sumo_actuated", drain=False):
    runner.compute_drain = drain
    trip = write_tripinfo(tmp_path / "ti.xml", [tripinfo_row("a", 0, 100, 300), tripinfo_row("h", 5, 6, 0.0)])
    runner._save_result(_result(ctrl), runner._stem(ctrl, 42, "1.0x"), str(trip), str(tmp_path / "t.parquet"),
                        42, "1.0x")
    return runner.output_dir / f"{runner._stem(ctrl, 42, '1.0x')}.json"


def test_result_json_has_provenance_and_new_fields(runner_factory, tmp_path):
    r = runner_factory()
    data = json.loads(_save(r, tmp_path).read_text())
    p = data["provenance"]
    for key in ("git_commit", "git_dirty", "sumocfg_path", "sumocfg_sha256", "sumo_version", "begin_time",
                "end_time", "step_length", "seed", "model_path", "model_sha256", "command_line", "config_hash"):
        assert key in p
    assert p["sumocfg_sha256"] and p["begin_time"] == 25200 and p["end_time"] == 28800 and p["seed"] == 42
    # legacy fields kept, new tripinfo fields added, zero-length artifact excluded from the new means
    assert data["trips_completed"] == 1
    assert data["trips_completed_nonzero"] == 1 and data["zero_length_trips"] == 1


def test_guard_refuses_legacy_or_different_config_but_allows_same_hash(runner_factory, tmp_path):
    r = runner_factory()
    path = _save(r, tmp_path)
    r._guard("sumo_actuated", 42, "1.0x")                              # same config hash: rerun allowed
    legacy = json.loads(path.read_text())
    legacy.pop("provenance")
    path.write_text(json.dumps(legacy))                                # legacy result: unknown config
    with pytest.raises(ResultExistsError, match="legacy"):
        r._guard("sumo_actuated", 42, "1.0x")
    r.overwrite = True
    r._guard("sumo_actuated", 42, "1.0x")                              # explicit overwrite


def test_guard_refuses_a_different_sumocfg(runner_factory, tmp_path):
    path = _save(runner_factory(), tmp_path)
    assert path.exists()
    other = tmp_path / "scn2"
    cfg2 = write_scenario(other)
    cfg2.write_text(cfg2.read_text().replace('value="1"', 'value="0.5"'))          # different step length
    r2 = EvaluationRunner(str(cfg2), "tls.json", eval_config_path=_cfg(tmp_path))
    with pytest.raises(ResultExistsError, match="config_hash"):
        r2._guard("sumo_actuated", 42, "1.0x")


def test_config_hash_changes_with_window_and_cfg(tmp_path):
    cfg = write_scenario(tmp_path / "s")
    h = prov.config_hash(cfg, 0, 100, "1.0x")
    assert h == prov.config_hash(cfg, 0, 100, "1.0x")
    assert h != prov.config_hash(cfg, 10, 100, "1.0x")
    assert h != prov.config_hash(cfg, 0, 100, "1.5x")


def test_combined_parquet_merges_all_jsons_and_is_atomic(runner_factory, tmp_path):
    r = runner_factory()
    _save(r, tmp_path, "sumo_actuated")
    _save(r, tmp_path, "fixed_time")
    path = r._save_combined()
    df = pd.read_parquet(path)
    assert sorted(df.controller) == ["fixed_time", "sumo_actuated"]
    assert "provenance" not in df.columns and "drain_metrics" not in df.columns   # nested blocks not flattened
    assert not list(r.output_dir.glob(".*.tmp*"))                                  # no temp litter
    # a later partial invocation must not shrink the table
    r2 = runner_factory()
    assert len(pd.read_parquet(r2._save_combined())) == 2


def test_full_payload_with_drain_metrics_on_synthetic_scenario(runner_factory, tmp_path):
    r = runner_factory(eval_cfg=_cfg(tmp_path, begin_time=0, episode_duration=200))
    data = json.loads(_save(r, tmp_path, drain=True).read_text())
    assert data["drain_protocol"] == "drain-v1"
    assert data["lag_per_scheduled_car_s"] is not None and 0.0 <= data["due_completion_ratio"] <= 1.0
    assert data["teleports_source"].startswith("MetricsCollector")        # no statistic file in this fake run
    assert data["drain_metrics"]["entry_edges"] and data["drain_metrics"]["cohorts"]
