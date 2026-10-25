"""
main.py — FastAPI application setup for CityMind AI.

Run:
  uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

Run a single worker: the SUMO-GUI process registry lives in process memory.
To expose beyond localhost set CITYMIND_API_KEY (and put TLS/auth in front).
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.security import allowed_origins, configured_key

logger = logging.getLogger("citymind.api")
logging.basicConfig(level=logging.INFO)

_docs_enabled = os.environ.get("CITYMIND_ENABLE_DOCS") == "1"
app = FastAPI(
    title="CityMind AI API",
    description="Centralized RL Traffic Signal Control — research API for the CityMind AI dashboard",
    version="1.0.0",
    docs_url="/docs" if _docs_enabled else None,
    redoc_url=None,
    openapi_url="/openapi.json" if _docs_enabled else None,
)

# No cookies/credentials are used, so allow_credentials stays False; wildcard
# origins in CITYMIND_CORS_ORIGINS are rejected by parse_origins().
_cors_origins = allowed_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.middleware("http")
async def _security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


# Tracks which routers actually loaded, so /api/status never lies about it.
_loaded_endpoints: list[str] = []

# ── Core Routers ──────────────────────────────────────────────────────────────
try:
    from api.routers import network, experiments, replay, xai
    app.include_router(network.router, prefix="/api/network", tags=["network"])
    app.include_router(experiments.router, prefix="/api/experiments", tags=["experiments"])
    app.include_router(replay.router, prefix="/api/replay", tags=["replay"])
    app.include_router(xai.router, prefix="/api/xai", tags=["xai"])
    _loaded_endpoints += ["/api/network", "/api/experiments", "/api/replay", "/api/xai"]
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

if not configured_key():
    logger.warning(
        "CITYMIND_API_KEY is not set: simulation-control endpoints (launch-gui, stop-gui, "
        "WebSocket) accept loopback clients only. Set CITYMIND_API_KEY before exposing the API."
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
