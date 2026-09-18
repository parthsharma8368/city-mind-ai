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
