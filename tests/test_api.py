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
    response = client.get("/api/experiments/../../../etc")
    assert response.status_code in (400, 404)


def test_simulation_launch_requires_api_key_when_configured(monkeypatch):
    """When CITYMIND_API_KEY is set, launch-gui must reject requests without it."""
    monkeypatch.setenv("CITYMIND_API_KEY", "test-secret")
    response = client.post("/api/simulation/launch-gui")
    assert response.status_code == 401
    response = client.post(
        "/api/simulation/launch-gui", headers={"X-API-Key": "test-secret"}
    )
    assert response.status_code != 401
