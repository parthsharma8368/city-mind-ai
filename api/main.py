"""
main.py — FastAPI application setup for CityMind AI.

Run:
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="CityMind AI API",
    description="Centralized RL Traffic Signal Control — Production API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Core Routers ──────────────────────────────────────────────────────────────
try:
    from api.routers import network, experiments, replay, comparison, xai
    app.include_router(network.router, prefix="/api/network", tags=["network"])
    app.include_router(experiments.router, prefix="/api/experiments", tags=["experiments"])
    app.include_router(replay.router, prefix="/api/replay", tags=["replay"])
    app.include_router(comparison.router, prefix="/api/comparison", tags=["comparison"])
    app.include_router(xai.router, prefix="/api/xai", tags=["xai"])
except ImportError as e:
    print(f"[main] Warning: could not load core router: {e}")

# ── Simulation Control Router (XQuartz + WebSocket) ───────────────────────────
try:
    from api.routers import simulation
    app.include_router(simulation.router, prefix="/api/simulation", tags=["simulation"])
    # Mount WebSocket at /ws/simulation
    app.add_api_websocket_route("/ws/simulation", simulation.simulation_ws)
except ImportError as e:
    print(f"[main] Warning: simulation router unavailable: {e}")

# ── Metrics Router ────────────────────────────────────────────────────────────
try:
    from api.routers import metrics
    app.include_router(metrics.router, prefix="/api/metrics", tags=["metrics"])
except ImportError as e:
    print(f"[main] Warning: metrics router unavailable: {e}")


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
        "endpoints": [
            "/api/network",
            "/api/experiments",
            "/api/replay",
            "/api/comparison",
            "/api/xai",
            "/api/simulation",
            "/api/metrics",
            "/ws/simulation",
        ],
    }
