"""
simulation.py — Simulation control endpoints for CityMind AI.

Provides:
  - POST /api/simulation/launch-gui  → launch sumo-gui via XQuartz
  - GET  /api/simulation/status      → process alive/idle status
  - GET  /api/simulation/latest-model → auto-discover latest model zip
  - WebSocket /ws/simulation         → stream live metrics
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

router = APIRouter()

_gui_process: Optional[subprocess.Popen] = None
_PROJECT_ROOT = Path(__file__).parent.parent.parent


class LaunchResponse(BaseModel):
    status: str
    pid: Optional[int] = None
    model: Optional[str] = None
    message: str


class StatusResponse(BaseModel):
    gui_running: bool
    gui_pid: Optional[int] = None
    latest_model: Optional[str] = None
    xquartz_display: Optional[str] = None


def _find_latest_model() -> Optional[Path]:
    experiments_dir = _PROJECT_ROOT / "artifacts" / "experiments"
    candidates = list(experiments_dir.glob("**/model_final.zip"))
    if not candidates:
        candidates = list(experiments_dir.glob("**/*.zip"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _get_display() -> str:
    return os.environ.get("DISPLAY", ":0")


def _is_gui_running() -> bool:
    global _gui_process
    if _gui_process is None:
        return False
    if _gui_process.poll() is not None:
        _gui_process = None
        return False
    return True


@router.post("/launch-gui", response_model=LaunchResponse)
def launch_gui(model: Optional[str] = None, delay: int = 80, duration: int = 3600):
    """Launch SUMO-GUI visual simulation via XQuartz."""
    global _gui_process

    if _is_gui_running():
        return LaunchResponse(
            status="already_running",
            pid=_gui_process.pid,
            message="SUMO-GUI already running.",
        )

    model_path: Optional[Path] = None
    if model:
        model_path = Path(model)
        if not model_path.exists():
            raise HTTPException(404, f"Model not found: {model}")
    else:
        model_path = _find_latest_model()
        if model_path is None:
            raise HTTPException(404, "No trained model found in artifacts/experiments/.")

    python_bin = sys.executable
    demo_script = _PROJECT_ROOT / "scripts" / "run_visual_demo.py"

    cmd = [python_bin, str(demo_script), "--model", str(model_path),
           "--delay", str(delay), "--duration", str(duration)]

    env = os.environ.copy()
    if "DISPLAY" not in env:
        env["DISPLAY"] = ":0"

    try:
        _gui_process = subprocess.Popen(
            cmd, cwd=str(_PROJECT_ROOT), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        raise HTTPException(500, f"Failed to launch SUMO-GUI: {exc}") from exc

    time.sleep(0.5)
    if _gui_process.poll() is not None:
        raise HTTPException(500, "SUMO-GUI process exited immediately.")

    return LaunchResponse(
        status="launched",
        pid=_gui_process.pid,
        model=str(model_path),
        message=f"SUMO-GUI launched (PID={_gui_process.pid}). XQuartz window opening. DISPLAY={env.get('DISPLAY')}",
    )


@router.post("/stop-gui")
def stop_gui():
    """Terminate the running SUMO-GUI process."""
    global _gui_process
    if not _is_gui_running():
        return {"status": "not_running", "message": "No SUMO-GUI process is running."}
    pid = _gui_process.pid
    try:
        _gui_process.terminate()
        _gui_process.wait(timeout=5)
    except Exception:
        try:
            _gui_process.kill()
        except Exception:
            pass
    _gui_process = None
    return {"status": "stopped", "pid": pid, "message": f"SUMO-GUI process {pid} terminated."}


@router.get("/status", response_model=StatusResponse)
def get_status():
    """Return current simulation status."""
    model = _find_latest_model()
    return StatusResponse(
        gui_running=_is_gui_running(),
        gui_pid=_gui_process.pid if _is_gui_running() else None,
        latest_model=str(model) if model else None,
        xquartz_display=_get_display(),
    )


@router.get("/latest-model")
def get_latest_model():
    """Return the path to the most recently trained model."""
    model = _find_latest_model()
    if model is None:
        raise HTTPException(404, "No trained model found.")
    run_dir = model.parent
    config: dict[str, Any] = {}
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        try:
            import yaml
            config = yaml.safe_load(cfg_path.read_text()) or {}
        except Exception:
            pass
    return {
        "model_path": str(model),
        "run_id": run_dir.name,
        "modified": model.stat().st_mtime,
        "config": config,
    }


@router.websocket("/ws")
async def simulation_ws(websocket: WebSocket):
    """WebSocket: broadcast live status + metrics every 2s."""
    await websocket.accept()
    try:
        while True:
            metrics_dir = _PROJECT_ROOT / "artifacts" / "metrics"
            metrics: dict[str, Any] = {}
            for json_file in metrics_dir.glob("*.json"):
                try:
                    data = json.loads(json_file.read_text())
                    ctrl = data.get("controller", json_file.stem)
                    metrics[ctrl] = {
                        "mean_waiting_time": data.get("mean_waiting_time", 0),
                        "mean_queue": data.get("mean_queue", 0),
                        "throughput": data.get("throughput", 0),
                        "mean_speed": data.get("mean_speed", 0),
                        "total_arrived": data.get("total_arrived", 0),
                        "cumulative_reward": data.get("cumulative_reward", 0),
                    }
                except Exception:
                    pass
            payload = {
                "type": "status",
                "timestamp": time.time(),
                "gui_running": _is_gui_running(),
                "gui_pid": _gui_process.pid if _is_gui_running() else None,
                "metrics": metrics,
            }
            await websocket.send_json(payload)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
