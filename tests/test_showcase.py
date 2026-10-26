"""The one-click showcase's data endpoints: compact frames and stored metrics."""
from __future__ import annotations

import gzip
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import results
from api.main import app
from api.routers import showcase

client = TestClient(app, client=("127.0.0.1", 50000))


@pytest.fixture
def fake_results(tmp_path, monkeypatch):
    """A tiny evaluation tree: two controllers, one seed, three frames of cars and two junctions."""
    root = tmp_path / "eval"
    monkeypatch.setattr(results, "RESULTS_ROOT", root)
    showcase._frames_bytes.cache_clear()
    rows = []
    for t, cars in ((26400.0, [("a", 1000.0, 2000.0, 5.0), ("b", 1100.0, 2100.0, 0.0)]),
                    (26405.0, [("a", 1030.0, 2000.0, 6.0), ("b", 1100.0, 2100.0, 0.0), ("c", 50.0, 60.0, 9.0)]),
                    (26410.0, [("a", 1060.0, 2000.0, 6.5)])):
        for vid, x, y, v in cars:
            rows.append({"time": t, "vehicle_id": vid, "x": x + showcase.ORIGIN_X, "y": y + showcase.ORIGIN_Y, "speed": v,
                         "tls_id": None, "phase": None, "tls_state": None})
        rows.append({"time": t, "vehicle_id": None, "x": None, "y": None, "speed": None, "tls_id": "J1", "phase": 0, "tls_state": "GGrr"})
        rows.append({"time": t, "vehicle_id": None, "x": None, "y": None, "speed": None, "tls_id": "J2", "phase": 1, "tls_state": "rryy"})
    for c in ("sumo_actuated", "ippo_A"):
        path = results.trajectory_path(c, 42)
        path.parent.mkdir(parents=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
    report = root / "report"
    report.mkdir(parents=True)
    table = pd.DataFrame([{"name": c, "seed": s, "lag_per_scheduled_car_s": lag + s, "trips_completed": 1000.0 + s}
                          for c, lag in (("sumo_actuated", 276.0), ("ippo_A", 235.0)) for s in (7, 42)])
    table.to_csv(report / "results.csv", index=False)
    yield root
    showcase._frames_bytes.cache_clear()


def test_frames_are_columnar_and_consistent(fake_results):
    r = client.get("/api/showcase/frames", params={"controller": "ippo_A", "start": 26400, "end": 26410})
    assert r.status_code == 200
    d = r.json()
    assert d["times"] == [26400.0, 26405.0, 26410.0] and d["counts"] == [2, 3, 1]
    assert len(d["vid"]) == len(d["x"]) == len(d["y"]) == len(d["v"]) == sum(d["counts"])
    assert d["vehicle_count"] == 3 and max(d["vid"]) < d["vehicle_count"]
    assert d["origin"] == [showcase.ORIGIN_X, showcase.ORIGIN_Y]
    assert d["x"][:2] == [1000.0, 1100.0]                       # relative to the origin
    # a car keeps its index across frames, so the client can interpolate it
    assert d["vid"][0] == d["vid"][2]
    assert d["tls_ids"] == ["J1", "J2"] and d["tls"] == ["gy", "gy", "gy"]   # J1 "GGrr" is green, J2 "rryy" shows yellow


def test_signal_character_rules():
    assert showcase._signal_char("GGrr") == "g"
    assert showcase._signal_char("Grrr") == "r"
    assert showcase._signal_char("ggyr") == "y"
    assert showcase._signal_char("") == "n" and showcase._signal_char(None) == "n" and showcase._signal_char(float("nan")) == "n"


def test_frames_are_gzipped_when_asked(fake_results):
    r = client.get("/api/showcase/frames", params={"controller": "sumo_actuated", "start": 26400, "end": 26410},
                   headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200
    assert json.loads(r.content)["controller"] == "sumo_actuated"


@pytest.mark.parametrize("params,code", [
    ({"controller": "nope"}, 404),
    ({"controller": "ippo_A", "start": 26400, "end": 26400}, 422),
    ({"controller": "ippo_A", "start": 26400, "end": 28000}, 422),            # longer than the 900 s cap
    ({"controller": "ippo_A", "start": 1, "end": 100}, 422),                   # outside the recorded hour
    ({"controller": "ippo_A", "demand_id": "../x"}, 400),
    ({"controller": "fixed_time"}, 404),                                       # no recording in the fake tree
    ({"controller": "ippo_A", "start": 27000, "end": 27500}, 404),             # no frames in that window
])
def test_frames_reject_bad_requests(fake_results, params, code):
    assert client.get("/api/showcase/frames", params=params).status_code == code


def test_controller_metrics_have_per_seed_values(fake_results):
    r = client.get("/api/showcase/metrics/ippo_A")
    assert r.status_code == 200
    d = r.json()
    assert d["n_runs"] == 2 and d["seeds"] == [7, 42] and d["kind"] == "model"
    lag = next(m for m in d["measures"] if m["key"] == "lag_per_scheduled_car_s")
    assert lag["per_seed"] == [242.0, 277.0] and lag["mean"] == pytest.approx(259.5) and lag["better"] == "lower"
    assert client.get("/api/showcase/metrics/nope").status_code == 404
    assert client.get("/api/showcase/metrics/fixed_time").status_code == 404   # not in the fake table


def test_config_lists_the_controllers_and_the_window(fake_results):
    d = client.get("/api/showcase/config").json()
    assert (d["start"], d["end"], d["speed"], d["step"]) == (26400, 27000, 10, 5)
    by_id = {c["id"]: c for c in d["controllers"]}
    assert by_id["ippo_A"]["has_recording"] and not by_id["fixed_time"]["has_recording"]
    assert [c["id"] for c in d["controllers"]] == list(results.CONTROLLERS)


# ── /policy: the trained weights the "inside the model" figure runs ─────────────────────────────────────────────────────
@pytest.fixture
def fake_models(tmp_path, monkeypatch):
    """A project root holding a model zip (policy.pth with the real layer shapes) for ippo_A only."""
    import io
    import zipfile

    import torch

    def lin(out, inp, fill):
        return {"weight": torch.full((out, inp), fill), "bias": torch.zeros(out)}

    state = {}
    for key, out_rows, fill in (("mlp_extractor.policy_net.0", 256, 0.01), ("mlp_extractor.policy_net.2", 256, 0.02), ("action_net", 2, 0.03),
                                ("mlp_extractor.value_net.0", 256, 0.04), ("mlp_extractor.value_net.2", 256, 0.05), ("value_net", 1, 0.06)):
        inp = 22 if key.endswith(".0") else 256
        for name, tensor in lin(out_rows, inp, fill).items():
            state[f"{key}.{name}"] = tensor
    buf = io.BytesIO()
    torch.save(state, buf)
    rel = showcase.MODEL_FILES["ippo_A"]
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("policy.pth", buf.getvalue())
    monkeypatch.setattr(showcase, "_PROJECT_ROOT", tmp_path)
    showcase._policy_payload.cache_clear()
    return tmp_path


def test_policy_serves_both_networks_with_their_shapes(fake_models):
    r = client.get("/api/showcase/policy/ippo_A")
    assert r.status_code == 200
    d = r.json()
    assert d["controller"] == "ippo_A" and d["activation"] == "tanh" and len(d["obs_names"]) == 22
    assert d["actor"]["l1"]["shape"] == [256, 22] and len(d["actor"]["l1"]["w"]) == 256 * 22
    assert d["actor"]["out"]["shape"] == [2, 256] and d["critic"]["out"]["shape"] == [1, 256]
    assert d["actor"]["l1"]["w"][0] == pytest.approx(0.01) and d["critic"]["l2"]["w"][0] == pytest.approx(0.05)


def test_policy_only_serves_the_fixed_model_files(fake_models):
    assert client.get("/api/showcase/policy/sumo_actuated").status_code == 404      # a baseline has no weights
    assert client.get("/api/showcase/policy/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.get("/api/showcase/policy/ippo_B").status_code == 404             # known id, file absent in the fake root
    assert "missing" in client.get("/api/showcase/policy/ippo_B").json()["detail"]


def test_policy_reports_a_corrupt_model_without_leaking_the_cause(fake_models):
    (fake_models / showcase.MODEL_FILES["ippo_A"]).write_bytes(b"not a zip")
    showcase._policy_payload.cache_clear()
    r = client.get("/api/showcase/policy/ippo_A")
    assert r.status_code == 500 and r.json()["detail"] == "Could not read the weights of ippo_A."
