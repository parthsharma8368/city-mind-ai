"""Drain-aware protocol on a hand-checkable synthetic scenario (no SUMO, no real artifacts)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import drain_metrics as dm
from src.evaluation.provenance import read_sumocfg
from tests._eval_synth import BEGIN, END, tripinfo_row, write_scenario, write_tripinfo


@pytest.fixture()
def demand(tmp_path):
    cfg = read_sumocfg(write_scenario(tmp_path / "scn"))
    return dm.build_scheduled_demand(cfg["route_files"], cfg["net_file"], BEGIN, END)


def _ideal(demand, vid):
    return float(demand.cars.set_index("id").loc[vid, "ideal"])


def test_scheduled_cars_free_flow_by_hand(demand):
    cars = demand.cars.set_index("id")
    assert set(cars.index) == {"carIn1", "carIn2", "hz1"}          # 'late' departs after the window
    assert demand.scheduled_buses == 2                              # flow period 100 from 0: buses at 0 and 100 are in [0, 200)
    assert bool(cars.loc["hz1", "zero_len"]) and not bool(cars.loc["carIn1", "zero_len"])
    # carIn1: E0 from pos 0 (100 m @10) + via 10 m @10 + E1 to arrivalPos 100 (@10) = 10 + 1 + 10 = 21 s
    assert cars.loc["carIn1", "tff_exp"] == pytest.approx(21.0)
    a, c = dm.DEFAULT_FF_CALIBRATION["a"], dm.DEFAULT_FF_CALIBRATION["c"]
    assert cars.loc["carIn1", "ff_cal"] == pytest.approx(a * 21.0 + c)
    assert cars.loc["carIn1", "ideal"] == pytest.approx(0 + a * 21.0 + c)
    assert cars.loc["hz1", "ff_cal"] == 1.0                         # zero-length trips: 1 s


def test_lag_per_scheduled_car_and_due_ratio_by_hand(demand, tmp_path):
    i1, i2, iz = _ideal(demand, "carIn1"), _ideal(demand, "carIn2"), _ideal(demand, "hz1")
    assert i2 <= END                                               # carIn2 is zero-delay feasible
    p = write_tripinfo(tmp_path / "t.xml", [
        tripinfo_row("carIn1", 0, i1 + 20, 210),                   # finishes 20 s late
        tripinfo_row("hz1", 50, iz, 0.0, loss=1.0),                # zero-length artifact, on time
        # carIn2 never completes (still driving / never inserted)
    ])
    s = dm.score_run(demand, dm.trips_frame(p))
    expected_lag = (20 + (END - i2) + 0.0) / 3                     # unfinished car is charged until END
    assert s["lag_per_scheduled_car_s"] == pytest.approx(expected_lag, abs=0.05)   # tripinfo times are 2-decimal
    assert s["scheduled_cars"] == 3 and s["scheduled_zero_length_cars"] == 1
    # due non-zero cars: carIn1, carIn2; only carIn1 completed
    assert s["due_nz_n"] == 2 and s["due_nz_completed"] == 1
    assert s["due_completion_ratio"] == pytest.approx(0.5)
    assert s["overdue_cars"] == 1
    assert s["cars_completed_nonzero"] == 1 and s["zero_len_completed"] == 1
    # per-trip means exclude the zero-length row
    assert s["mean_timeloss_cars_nonzero"] == pytest.approx(20.0)
    assert s["tripinfo_zero_length_rows"] == 1


def test_perfect_controller_has_zero_lag_and_full_due_ratio(demand, tmp_path):
    rows = [tripinfo_row(v, _ideal(demand, v) - 1, _ideal(demand, v),
                         0.0 if v == "hz1" else 100.0) for v in ("carIn1", "carIn2", "hz1")]
    s = dm.score_run(demand, dm.trips_frame(write_tripinfo(tmp_path / "t.xml", rows)))
    assert s["lag_per_scheduled_car_s"] == pytest.approx(0.0, abs=0.05)
    assert s["due_completion_ratio"] == pytest.approx(1.0)


def test_survivor_bias_unfinished_cars_are_charged_not_ignored(demand, tmp_path):
    """A controller finishing nothing must score WORSE than one finishing everything late,
    even though its per-completed-trip mean (over an empty set) looks 'perfect'."""
    none = dm.score_run(demand, dm.trips_frame(write_tripinfo(tmp_path / "n.xml", [])))
    late = dm.score_run(demand, dm.trips_frame(write_tripinfo(tmp_path / "l.xml", [
        tripinfo_row("carIn1", 0, _ideal(demand, "carIn1") + 20, 210),
        tripinfo_row("carIn2", 100, _ideal(demand, "carIn2") + 20, 300),
        tripinfo_row("hz1", 50, _ideal(demand, "hz1"), 0.0),
    ])))
    assert none["mean_timeloss_cars_nonzero"] is None
    assert none["lag_per_scheduled_car_s"] > late["lag_per_scheduled_car_s"]
    assert none["due_completion_ratio"] == 0.0


def test_entry_edge_table_and_cohorts(demand, tmp_path):
    p = write_tripinfo(tmp_path / "t.xml", [tripinfo_row("carIn1", 0, _ideal(demand, "carIn1"), 210, delay=299.0)])
    s = dm.score_run(demand, dm.trips_frame(p), entry_edges=("E0", "missing_edge"))
    table = {r["edge"]: r for r in s["entry_edges"]}
    assert table["E0"]["role"] == "entry"
    assert table["E0"]["scheduled_nonzero"] == 2 and table["E0"]["completed"] == 1
    assert table["E0"]["completion_ratio"] == pytest.approx(0.5)
    assert table["E0"]["completed_at_cap"] == 1                    # depart delay 299 s ~ max-depart-delay cap
    assert table["missing_edge"]["scheduled_nonzero"] == 0 and table["missing_edge"]["completion_ratio"] is None
    assert "E1" not in table or table["E1"]["role"] == "top_origin"  # generic top-origin edges are appended
    assert sum(c["scheduled_nonzero"] for c in s["cohorts"]) == 2
    assert s["cohorts"][0]["cohort"] == "0-3" and s["cohorts"][0]["completed"] == 1


def test_seen_ids_split_never_inserted_from_in_network(demand, tmp_path):
    p = write_tripinfo(tmp_path / "t.xml", [])
    s = dm.score_run(demand, dm.trips_frame(p), entry_edges=("E0",), seen_ids={"carIn1"})
    row = {r["edge"]: r for r in s["entry_edges"]}["E0"]
    assert row["inserted"] == 1 and row["never_inserted"] == 1


def test_never_inserted_uses_scheduled_total(demand, tmp_path):
    p = write_tripinfo(tmp_path / "t.xml", [])
    s = dm.score_run(demand, dm.trips_frame(p), inserted=4)
    assert s["never_inserted"] == demand.n_scheduled_total - 4


def test_teleports_come_from_statistic_output_not_tripinfo(demand, tmp_path):
    stat = tmp_path / "stat.xml"
    stat.write_text('<statistics><vehicles loaded="10" inserted="8" running="2" waiting="1"/>'
                    '<teleports total="7" jam="1" yield="3" wrongLane="3"/>'
                    '<safety collisions="2" emergencyStops="0" emergencyBraking="1"/></statistics>')
    assert dm.parse_statistic_output(stat)["teleports_total"] == 7
    p = write_tripinfo(tmp_path / "t.xml", [])
    s = dm.score_tripinfo_file(demand, p, statistic_path=stat, json_teleports=99)
    assert s["teleports"] == 7 and s["teleports_source"] == "sumo --statistic-output"
    assert s["inserted"] == 8
    s2 = dm.score_tripinfo_file(demand, p, json_teleports=99)
    assert s2["teleports"] == 99 and "MetricsCollector" in s2["teleports_source"]


def test_statistic_output_rejects_garbage(tmp_path):
    bad = tmp_path / "b.xml"
    bad.write_text("<statistics/>")
    with pytest.raises(ValueError):
        dm.parse_statistic_output(bad)
    with pytest.raises(ValueError):
        dm.parse_statistic_output(tmp_path / "missing.xml")


def test_summary_discarded_reads_last_step(tmp_path):
    p = tmp_path / "s.xml"
    p.write_text('<summary><step time="1" discarded="3"/><step time="2" discarded="9"/></summary>')
    assert dm.parse_summary_discarded(p) == 9
    assert dm.parse_summary_discarded(tmp_path / "none.xml") is None


def test_calibration_recovers_linear_relation():
    rng = np.random.default_rng(0)
    n = 80
    tff = rng.uniform(30, 600, n)
    cars = pd.DataFrame({"id": [f"c{i}" for i in range(n)], "nRoutes": 1, "tff_exp": tff})
    trips = pd.DataFrame({"id": cars["id"], "rerouteNo": 0, "routeLength": 500.0,
                          "duration": 0.95 * tff + 7.0 + 20.0, "timeLoss": 20.0})
    cal = dm.calibrate_free_flow(cars, trips)
    assert cal["a"] == pytest.approx(0.95, abs=1e-6) and cal["c"] == pytest.approx(7.0, abs=1e-5)
    with pytest.raises(ValueError):
        dm.calibrate_free_flow(cars.head(10), trips.head(10))


def test_scheduled_cars_cache_roundtrip(tmp_path):
    cfg = read_sumocfg(write_scenario(tmp_path / "scn"))
    a = dm.load_scheduled_cars(cfg["route_files"], cfg["net_file"], BEGIN, END, cache_dir=tmp_path / "cache")
    assert len(list((tmp_path / "cache").glob("*.parquet"))) == 1
    b = dm.load_scheduled_cars(cfg["route_files"], cfg["net_file"], BEGIN, END, cache_dir=tmp_path / "cache")
    pd.testing.assert_frame_equal(a, b)


def test_flat_and_json_fields(demand, tmp_path):
    s = dm.score_run(demand, dm.trips_frame(write_tripinfo(tmp_path / "t.xml", [])))
    flat = dm.flat_fields(s)
    assert flat["drain_protocol"] == dm.PROTOCOL_ID and "lag_per_scheduled_car_s" in flat
    nested = dm.drain_json_fields(s)["drain_metrics"]
    assert "entry_edges" in nested and "cohorts" in nested
