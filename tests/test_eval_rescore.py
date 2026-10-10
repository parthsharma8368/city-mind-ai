"""Offline re-scoring CLI (scripts/rescore_drain_metrics.py): zero simulation, synthetic artifacts."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.evaluation import drain_metrics as dm
from src.evaluation.provenance import read_sumocfg
from tests._eval_synth import BEGIN, END, tripinfo_row, write_scenario, write_tripinfo

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "rescore_drain_metrics.py"


@pytest.fixture()
def artifacts(tmp_path):
    cfg = write_scenario(tmp_path / "scn")
    info = read_sumocfg(cfg)
    demand = dm.build_scheduled_demand(info["route_files"], info["net_file"], BEGIN, END)
    ideal = demand.cars.set_index("id")["ideal"]
    mdir = tmp_path / "metrics"
    for ctrl, late in (("good", 5.0), ("bad", None)):
        for seed in (42, 123):
            rows = [tripinfo_row("hz1", 50, ideal["hz1"], 0.0, loss=1.0)]
            if late is not None:
                rows += [tripinfo_row("carIn1", 0, ideal["carIn1"] + late, 210),
                         tripinfo_row("carIn2", 100, ideal["carIn2"] + late, 300)]
            write_tripinfo(mdir / "tripinfo" / f"{ctrl}_seed{seed}_1.0x.xml", rows)
            (mdir / f"{ctrl}_seed{seed}_1.0x.json").write_text(json.dumps(
                {"controller": ctrl, "seed": seed, "total_departed": 3, "total_teleports": 2, "backlog": 1}))
    return cfg, mdir


def _run(cfg, mdir, *extra):
    cmd = [sys.executable, str(SCRIPT), "--sumo-cfg", str(cfg), "--metrics-dir", str(mdir),
           "--begin", str(BEGIN), "--end", str(END), "--default-calibration", "--no-trajectories", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=False)


def test_rescore_writes_json_csv_md_and_ranks_correctly(artifacts):
    cfg, mdir = artifacts
    before = sorted(p.name for p in mdir.rglob("*") if p.is_file())
    r = _run(cfg, mdir, "--baseline", "good")
    assert r.returncode == 0, r.stderr + r.stdout

    out = json.loads((mdir / "drain_scores.json").read_text())
    assert out["drain_protocol"] == dm.PROTOCOL_ID
    assert len(out["runs"]) == 4
    by = {s["controller"]: s for s in out["summary"]}
    assert by["good"]["lag_mean"] < by["bad"]["lag_mean"]            # finishing late beats finishing nothing
    assert by["good"]["n_seeds"] == 2
    assert "paired_vs_baseline" in out and out["paired_vs_baseline"]["lag_per_scheduled_car_s"][0]["n_seeds"] == 2
    assert "p_value" not in json.dumps(out)

    rows = list(csv.DictReader((mdir / "drain_scores.csv").open()))
    assert len(rows) == 4 and "lag_per_scheduled_car_s" in rows[0] and "entry_edges" not in rows[0]
    assert "| good |" in (mdir / "drain_scores.md").read_text()

    # inputs untouched, only drain_scores.* and the cache were added
    after = sorted(p.name for p in mdir.rglob("*") if p.is_file())
    new = set(after) - set(before)
    assert all(n.startswith("drain_scores") or n.startswith("scheduled_cars_") for n in new), new


def test_rescore_fails_loudly_on_truncated_tripinfo(artifacts):
    cfg, mdir = artifacts
    bad = mdir / "tripinfo" / "good_seed42_1.0x.xml"
    bad.write_text(bad.read_text()[:-40])                               # cut the closing tag + part of a row
    r = _run(cfg, mdir)
    assert r.returncode != 0
    assert "truncated" in (r.stderr + r.stdout)
    assert not (mdir / "drain_scores.json").exists()


def test_rescore_json_is_not_a_controller_record(artifacts):
    """The API globs artifacts/metrics/*.json and matches on a top-level 'controller' key."""
    cfg, mdir = artifacts
    assert _run(cfg, mdir).returncode == 0
    assert "controller" not in json.loads((mdir / "drain_scores.json").read_text())
