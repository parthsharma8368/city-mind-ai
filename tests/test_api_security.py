"""Regression tests for the API's security controls. No test may start a real process."""

import json
import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from api.main import app
from api import results
from api.routers import experiments, replay, simulation

LOCAL = TestClient(app, client=("127.0.0.1", 50000))
REMOTE = TestClient(app, client=("203.0.113.9", 50000))
GOOD_ORIGIN = "http://localhost:5173"
EVIL_ORIGIN = "http://evil.example"


class _FakeProc:
    pid = 424242

    def poll(self):
        return None


@pytest.fixture(autouse=True)
def no_real_processes(monkeypatch):
    monkeypatch.delenv("CITYMIND_API_KEY", raising=False)
    monkeypatch.delenv("CITYMIND_ALLOW_INSECURE", raising=False)
    monkeypatch.delenv("CITYMIND_CORS_ORIGINS", raising=False)
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _FakeProc()

    monkeypatch.setattr(simulation, "_demo_script_exists", lambda: True)   # the launcher script is retired; keep testing the API's guards
    monkeypatch.setattr(simulation.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(simulation.time, "sleep", lambda _s: None)
    monkeypatch.setattr(simulation.os, "killpg", lambda *a, **k: None)
    monkeypatch.setattr(simulation.os, "getpgid", lambda pid: pid)
    with simulation._gui_lock:
        simulation._gui_processes.clear()
    yield calls
    with simulation._gui_lock:
        simulation._gui_processes.clear()


LAUNCH = "/api/simulation/launch-gui"


# ── auth modes ───────────────────────────────────────────────────────────────
def test_no_key_loopback_client_allowed(no_real_processes):
    r = LOCAL.post(LAUNCH, params={"controller": "fixed_time"})
    assert r.status_code == 200 and r.json()["status"] == "launched"
    assert len(no_real_processes) == 1


def test_no_key_remote_client_refused(no_real_processes):
    assert REMOTE.post(LAUNCH, params={"controller": "fixed_time"}).status_code == 403
    assert REMOTE.post("/api/simulation/stop-gui").status_code == 403
    assert no_real_processes == []


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_key_is_treated_as_unset(monkeypatch, blank, no_real_processes):
    monkeypatch.setenv("CITYMIND_API_KEY", blank)
    assert REMOTE.post(LAUNCH, params={"controller": "fixed_time"}).status_code == 403
    assert LOCAL.post(LAUNCH, params={"controller": "fixed_time"}).status_code == 200


def test_key_required_and_compared(monkeypatch, no_real_processes):
    monkeypatch.setenv("CITYMIND_API_KEY", "s3cret")
    p = {"controller": "fixed_time"}
    assert LOCAL.post(LAUNCH, params=p).status_code == 401
    assert LOCAL.post(LAUNCH, params=p, headers={"X-API-Key": "wrong"}).status_code == 401
    assert REMOTE.post(LAUNCH, params=p, headers={"X-API-Key": "s3cret"}).status_code == 200


def test_stop_gui_requires_key(monkeypatch):
    monkeypatch.setenv("CITYMIND_API_KEY", "s3cret")
    assert LOCAL.post("/api/simulation/stop-gui").status_code == 401
    ok = LOCAL.post("/api/simulation/stop-gui", headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200 and ok.json()["status"] == "not_running"


def test_allow_insecure_opt_out(monkeypatch, no_real_processes):
    monkeypatch.setenv("CITYMIND_ALLOW_INSECURE", "1")
    assert REMOTE.post(LAUNCH, params={"controller": "fixed_time"}).status_code == 200


# ── CSRF ─────────────────────────────────────────────────────────────────────
def test_cross_origin_post_rejected(no_real_processes):
    r = LOCAL.post(
        LAUNCH, params={"controller": "fixed_time"},
        headers={"Origin": EVIL_ORIGIN, "Content-Type": "text/plain"},
    )
    assert r.status_code == 403
    assert LOCAL.post("/api/simulation/stop-gui", headers={"Origin": EVIL_ORIGIN}).status_code == 403
    assert no_real_processes == []


def test_simple_content_type_and_sec_fetch_rejected(no_real_processes):
    p = {"controller": "fixed_time"}
    assert LOCAL.post(LAUNCH, params=p, headers={"Content-Type": "text/plain"}).status_code == 415
    assert LOCAL.post(LAUNCH, params=p, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert LOCAL.post(LAUNCH, params=p, headers={"Sec-Fetch-Site": "cross-site", "Origin": EVIL_ORIGIN}).status_code == 403
    assert no_real_processes == []


def test_allowed_origin_cross_site_dev_ui_works(no_real_processes):
    """localhost:5173 -> 127.0.0.1:8000 is 'cross-site' to browsers but its Origin is allow-listed."""
    r = LOCAL.post(LAUNCH, params={"controller": "fixed_time"},
                   headers={"Origin": GOOD_ORIGIN, "Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 200


def test_allowed_origin_post_works(no_real_processes):
    r = LOCAL.post(LAUNCH, params={"controller": "fixed_time"}, headers={"Origin": GOOD_ORIGIN})
    assert r.status_code == 200


# ── launch-gui input validation ──────────────────────────────────────────────
def test_unknown_controller_rejected(no_real_processes):
    assert LOCAL.post(LAUNCH, params={"controller": "../x"}).status_code == 400
    assert no_real_processes == []


@pytest.mark.parametrize("params", [
    {"delay": -5}, {"delay": 5000}, {"duration": 99999999999}, {"duration": 1}, {"delay": "x"},
])
def test_delay_duration_bounds(params, no_real_processes):
    assert LOCAL.post(LAUNCH, params={"controller": "fixed_time", **params}).status_code == 422
    assert no_real_processes == []


def test_model_path_jail(tmp_path, no_real_processes):
    outside = tmp_path / "m.zip"
    outside.write_bytes(b"x")
    for bad in (str(outside), "../../etc/passwd.zip", "/etc/hosts"):
        r = LOCAL.post(LAUNCH, params={"controller": "ppo", "model": bad})
        assert r.status_code in (400, 404), bad
    # inside artifacts/ but outside artifacts/experiments/ is also refused
    r = LOCAL.post(LAUNCH, params={"controller": "ppo", "model": "artifacts/metrics/x.zip"})
    assert r.status_code == 400
    assert no_real_processes == []


def test_model_path_inside_experiments_accepted_and_relative(monkeypatch, no_real_processes):
    exp = simulation._EXPERIMENTS_DIR
    zip_path = exp / "ppo" / "_pytest_tmp_run" / "model_final.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.write_bytes(b"x")
    try:
        r = LOCAL.post(LAUNCH, params={"controller": "ppo", "model": str(zip_path)})
        assert r.status_code == 200
        assert r.json()["model"] == "artifacts/experiments/ppo/_pytest_tmp_run/model_final.zip"
        assert "/Users" not in r.text and not r.json()["model"].startswith("/")
    finally:
        zip_path.unlink()
        zip_path.parent.rmdir()


def test_latest_model_and_status_have_relative_paths():
    for path in ("/api/simulation/status", "/api/simulation/latest-model"):
        r = LOCAL.get(path)
        if r.status_code == 200:
            body = r.json()
            val = body.get("latest_model") or body.get("model_path")
            assert val is None or not val.startswith("/")


# ── other path jails / allow-lists ───────────────────────────────────────────
def test_token_regex_blocks_traversal_on_replay():
    for field in ("controller", "demand_id"):
        params = {"controller": "fixed_time", "demand_id": "1.0x", field: "../../etc"}
        assert LOCAL.get("/api/replay/frames", params=params).status_code == 400


def test_experiments_run_id_validation_and_containment():
    assert LOCAL.get("/api/experiments/bad%20id").status_code == 400
    assert LOCAL.get("/api/experiments/bad%2Fid").status_code in (400, 404)
    with pytest.raises(Exception) as ei:
        experiments._resolve_run_dir("bad id")
    assert getattr(ei.value, "status_code", None) == 400
    # ".." passes the regex but must never resolve outside the experiment roots
    for token in ("..", "."):
        assert not experiments._resolve_run_dir(token).exists()
        assert LOCAL.get(f"/api/experiments/{token}x/summary").status_code == 404


def test_xai_controller_and_tls_id_validation():
    assert LOCAL.get("/api/xai/explain/bad%20id").status_code == 400
    assert LOCAL.get("/api/xai/driver-plot/bad%20id").status_code == 400
    assert LOCAL.get("/api/xai/summary", params={"controller": "evil"}).status_code == 400


# ── CORS ─────────────────────────────────────────────────────────────────────
def test_cors_foreign_origin_gets_no_allow_header():
    r = LOCAL.get("/api/health", headers={"Origin": EVIL_ORIGIN})
    assert "access-control-allow-origin" not in r.headers


def test_cors_allowed_origin_without_credentials():
    r = LOCAL.get("/api/health", headers={"Origin": GOOD_ORIGIN})
    assert r.headers["access-control-allow-origin"] == GOOD_ORIGIN
    assert "access-control-allow-credentials" not in r.headers


def test_cors_env_wildcard_is_rejected():
    from api.security import parse_origins

    assert parse_origins("*,http://a.example, null ,http://b.example/") == [
        "http://a.example", "http://b.example",
    ]


def test_security_headers_present():
    assert LOCAL.get("/api/health").headers["x-content-type-options"] == "nosniff"


# ── replay window clamp ──────────────────────────────────────────────────────
@pytest.fixture
def traj_dir(tmp_path, monkeypatch):
    rows = []
    for t in range(0, 2000):
        rows.append({"time": float(t), "vehicle_id": f"v{t}", "x": 1.0, "y": 2.0, "speed": 3.0,
                     "tls_id": "J0", "phase": 0, "tls_state": "GG"})
    folder = tmp_path / "fixed_time" / "s42" / "traj"
    folder.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(folder / "fixed_time_seed42_1.0x.parquet")
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path)
    return tmp_path


def _frames(**kw):
    p = {"controller": "fixed_time", "seed": 42, "demand_id": "1.0x"}
    p.update(kw)
    return LOCAL.get("/api/replay/frames", params=p)


def test_replay_window_ok_and_filtered(traj_dir):
    r = _frames(start_time=100, end_time=400)
    assert r.status_code == 200
    body = r.json()
    assert body["frame_count"] == 301
    assert min(float(k) for k in body["frames"]) == 100.0
    assert body["tls_frames"]["100.0"][0]["state"] == "GG"


@pytest.mark.parametrize("start,end", [
    (0, 100000), (0, 601), (500, 100), ("nan", "nan"), (0, "inf"), (-5, 10),
])
def test_replay_window_rejected(traj_dir, start, end):
    assert _frames(start_time=start, end_time=end).status_code == 422


def test_replay_max_window_accepted(traj_dir):
    assert _frames(start_time=0, end_time=600).status_code == 200


def test_replay_missing_file_404(traj_dir):
    assert _frames(seed=999).status_code == 404


# ── YAML safety / malformed configs ──────────────────────────────────────────
def test_yaml_python_tags_are_never_executed(tmp_path):
    marker = tmp_path / "pwned"
    payload = f"a: !!python/object/apply:os.system ['touch {marker}']\nseed: 1\n"
    cfg = experiments._load_yaml(payload)
    assert not marker.exists()
    assert cfg["seed"] == 1


def test_yaml_garbage_returns_empty():
    assert experiments._load_yaml("a: [unclosed") == {}
    assert experiments._load_yaml("- just\n- a list") == {}


def test_malformed_config_does_not_break_listing(tmp_path, monkeypatch):
    root = tmp_path / "ppo"
    for name, text in {
        "run_good": "algorithm: PPO\nseed: 42\ntotal_timesteps: 100\n",
        "run_nullseed": "seed: null\n",
        "run_text": "total_timesteps: 40M\n",
        "run_list": "- 1\n- 2\n",
        "run_broken": "a: [unclosed\n",
    }.items():
        (root / name).mkdir(parents=True)
        (root / name / "config.yaml").write_text(text)
    monkeypatch.setattr(experiments, "EXPERIMENTS_ROOTS", {"ppo": root})
    r = LOCAL.get("/api/experiments/")
    assert r.status_code == 200
    ids = {x["run_id"] for x in r.json()}
    assert "run_good" in ids and "run_nullseed" in ids
    assert LOCAL.get("/api/experiments/run_list").status_code == 200


# ── metrics: final-model results ─────────────────────────────────────────────
def _write_run(root, name, seed=42, **kw):
    folder = root / name / f"s{seed}" / "metrics"
    folder.mkdir(parents=True)
    stem = f"{results.CONTROLLERS[name][1]}_seed{seed}_1.0x"
    (folder / f"{stem}.json").write_text(json.dumps({"seed": seed, **kw}, allow_nan=True))


def test_summary_picks_the_final_model_with_most_trips_and_survives_nan(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path)
    _write_run(tmp_path, "sumo_actuated", trips_completed=100, mean_waiting_time=6.0)
    _write_run(tmp_path, "ippo_A", trips_completed=110, mean_waiting_time=math.nan)
    _write_run(tmp_path, "ippo_B", trips_completed=105, mean_waiting_time=4.0)
    r = LOCAL.get("/api/metrics/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["controller"] == "ippo_A" and body["mean_waiting_time"] is None
    assert body["trips_completed_gap_pct"] == 10.0 and body["starvation_flag"] is False


def test_summary_404_without_results(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path)
    assert LOCAL.get("/api/metrics/summary").status_code == 404
    assert LOCAL.get("/api/metrics/comparison").json()["results"] == []
    assert LOCAL.get("/api/metrics/ten-seed").status_code == 404


def test_partial_json_is_skipped_not_substituted(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path)
    _write_run(tmp_path, "ippo_A", seed=123, trips_completed=1)
    folder = tmp_path / "ippo_A" / "s42" / "metrics"
    folder.mkdir(parents=True)
    (folder / "ippo_seed42_1.0x.json").write_text('{"seed": 42, "trips_')
    monkeypatch.setattr("api.util.time.sleep", lambda _s: None)
    assert results.load_metrics("ippo_A", 42) == {}
    assert results.load_metrics("evil", 42) == {}


def test_read_json_file_retries_until_complete(tmp_path, monkeypatch):
    from api import util

    f = tmp_path / "x.json"
    f.write_text('{"a": ')
    calls = {"n": 0}

    def heal(_s):
        calls["n"] += 1
        f.write_text('{"a": 1}')

    monkeypatch.setattr(util.time, "sleep", heal)
    assert util.read_json_file(f) == {"a": 1}
    assert calls["n"] == 1


# ── WebSocket ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("path", ["/ws/simulation", "/api/simulation/ws"])
def test_ws_foreign_origin_rejected(path):
    with pytest.raises(WebSocketDisconnect), LOCAL.websocket_connect(
        path, headers={"Origin": EVIL_ORIGIN}
    ):
        pass


def test_ws_remote_client_rejected_without_key():
    with pytest.raises(WebSocketDisconnect), REMOTE.websocket_connect("/ws/simulation"):
        pass


def test_ws_key_required_when_configured(monkeypatch):
    monkeypatch.setenv("CITYMIND_API_KEY", "s3cret")
    with pytest.raises(WebSocketDisconnect), LOCAL.websocket_connect("/ws/simulation"):
        pass
    with LOCAL.websocket_connect("/ws/simulation", headers={"X-API-Key": "s3cret"}) as ws:
        assert ws.receive_json()["type"] == "status"


def test_ws_loopback_allowed_origin_works():
    with LOCAL.websocket_connect("/ws/simulation", headers={"Origin": GOOD_ORIGIN}) as ws:
        msg = ws.receive_json()
        assert msg["type"] == "status" and msg["gui_running"] is False


def test_launch_gui_reports_501_when_the_launcher_script_is_retired(monkeypatch):
    monkeypatch.setattr(simulation, "_demo_script_exists", lambda: False)
    r = LOCAL.post(f"{LAUNCH}?controller=sumo_actuated")
    assert r.status_code == 501 and "retired" in r.json()["detail"]
