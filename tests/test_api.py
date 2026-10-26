"""Tests for FastAPI backend endpoints."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    """Verify health endpoint responds with 200 and healthy status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "CityMind AI"


def test_list_experiments_empty():
    """Verify listing experiments works even if metrics folder is fresh."""
    response = client.get("/api/experiments/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_network_geojson_endpoint():
    """Verify network geojson is served."""
    response = client.get("/api/network/geojson")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) > 0


def test_network_geojson_path_param_removed():
    """The old geojson_path override must not exist — it was an arbitrary file read."""
    response = client.get("/api/network/geojson", params={"geojson_path": "/etc/hosts"})
    # FastAPI ignores unknown query params; this must still return the real
    # network file, never the contents of an arbitrary path.
    data = response.json()
    assert data.get("type") == "FeatureCollection"


def test_replay_frames_rejects_path_traversal():
    """controller/demand_id must be filename-safe tokens, not traversal payloads."""
    response = client.get("/api/replay/frames", params={"controller": "../../etc", "demand_id": "1.0x"})
    assert response.status_code == 400


def test_experiments_get_rejects_path_traversal():
    # Percent-encoded so the HTTP client cannot normalise it away before the validator sees it.
    assert client.get("/api/experiments/bad%20id").status_code == 400
    assert client.get("/api/experiments/bad%20id/summary").status_code == 400


class _FakeProc:
    pid = 424242
    returncode = None

    def poll(self):
        return None


@pytest.fixture
def fake_spawn(monkeypatch):
    """Never start a real process: patch Popen and the 0.5 s sleep, isolate the registry."""
    from api.routers import simulation

    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _FakeProc()

    monkeypatch.setattr(simulation, "_demo_script_exists", lambda: True)
    monkeypatch.setattr(simulation.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(simulation.time, "sleep", lambda _s: None)
    with simulation._gui_lock:
        simulation._gui_processes.clear()
    yield calls
    with simulation._gui_lock:
        simulation._gui_processes.clear()


def test_simulation_launch_requires_api_key_when_configured(monkeypatch, fake_spawn):
    """When CITYMIND_API_KEY is set, launch-gui must reject requests without it."""
    monkeypatch.setenv("CITYMIND_API_KEY", "test-secret")
    response = client.post("/api/simulation/launch-gui", params={"controller": "fixed_time"})
    assert response.status_code == 401
    assert fake_spawn == []
    response = client.post(
        "/api/simulation/launch-gui",
        params={"controller": "fixed_time"},
        headers={"X-API-Key": "test-secret"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "launched"
    assert len(fake_spawn) == 1
    assert fake_spawn[0][1]["start_new_session"] is True
