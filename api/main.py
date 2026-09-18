"""
main.py — FastAPI application setup for CityMind AI.

Run:
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("citymind.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="CityMind AI API",
    description="Centralized RL Traffic Signal Control — research API for the CityMind AI dashboard",
    version="1.0.0",
)

_default_origins = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000"
_cors_origins = [o.strip() for o in os.environ.get("CITYMIND_CORS_ORIGINS", _default_origins).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# Tracks which routers actually loaded, so /api/status never lies about it.
_loaded_endpoints: list[str] = []

# ── Core Routers ──────────────────────────────────────────────────────────────
try:
    from api.routers import network, experiments, replay, comparison, xai
    app.include_router(network.router, prefix="/api/network", tags=["network"])
    app.include_router(experiments.router, prefix="/api/experiments", tags=["experiments"])
    app.include_router(replay.router, prefix="/api/replay", tags=["replay"])
    app.include_router(comparison.router, prefix="/api/comparison", tags=["comparison"])
    app.include_router(xai.router, prefix="/api/xai", tags=["xai"])
    _loaded_endpoints += ["/api/network", "/api/experiments", "/api/replay", "/api/comparison", "/api/xai"]
except ImportError as e:
    logger.warning("could not load core router: %s", e)

# ── Simulation Control Router (XQuartz + WebSocket) ───────────────────────────
try:
    from api.routers import simulation
    app.include_router(simulation.router, prefix="/api/simulation", tags=["simulation"])
    # Mount WebSocket at /ws/simulation
    app.add_api_websocket_route("/ws/simulation", simulation.simulation_ws)
    _loaded_endpoints += ["/api/simulation", "/ws/simulation"]
except ImportError as e:
    logger.warning("simulation router unavailable: %s", e)

# ── Metrics Router ────────────────────────────────────────────────────────────
try:
    from api.routers import metrics
    app.include_router(metrics.router, prefix="/api/metrics", tags=["metrics"])
    _loaded_endpoints.append("/api/metrics")
except ImportError as e:
    logger.warning("metrics router unavailable: %s", e)

if not os.environ.get("CITYMIND_API_KEY"):
    logger.warning(
        "CITYMIND_API_KEY is not set. Simulation-control endpoints (/api/simulation/launch-gui, "
        "stop-gui) are unauthenticated — fine for 127.0.0.1-only use, required before binding "
        "uvicorn to any non-localhost host."
    )


@app.get("/")
def health_check():
    return {"status": "ok", "service": "CityMind AI API", "version": "1.0.0"}


@app.get("/api/health")
def api_health():
    return {"status": "healthy", "service": "CityMind AI", "version": "1.0.0"}


@app.get("/api/status")
def status():
    return {
        "status": "running",
        "version": "1.0.0",
        "endpoints": _loaded_endpoints,
    }
