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
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from api.security import require_api_key

router = APIRouter()

# Keyed by controller name ("fixed_time" | "max_pressure" | "ppo" | "ippo") so
# multiple controllers can run as separate SUMO-GUI windows at once — the
# whole point is comparing a specific intersection live between two windows.
_gui_processes: dict[str, subprocess.Popen] = {}
_gui_lock = threading.Lock()
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_ARTIFACTS_DIR = (_PROJECT_ROOT / "artifacts").resolve()
_CONTROLLERS = ("fixed_time", "sumo_actuated", "max_pressure", "ppo", "ippo")
_MODEL_BASED_CONTROLLERS = ("ppo", "ippo")


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
    running_controllers: list[str] = []


def _find_latest_model(experiments_subdir: str = "") -> Optional[Path]:
    experiments_dir = _PROJECT_ROOT / "artifacts" / "experiments" / experiments_subdir
    candidates = list(experiments_dir.glob("**/model_final.zip"))
    if not candidates:
        candidates = list(experiments_dir.glob("**/*.zip"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _get_display() -> str:
    return os.environ.get("DISPLAY", ":0")


def _is_running(controller: str) -> bool:
    proc = _gui_processes.get(controller)
    if proc is None:
        return False
    if proc.poll() is not None:
        del _gui_processes[controller]
        return False
    return True


def _resolve_model_path(model: Optional[str], controller: str) -> Path:
    """Resolve a requested model path and require it to live under artifacts/."""
    if not model:
        found = _find_latest_model(controller)
        if found is None:
            raise HTTPException(404, f"No trained {controller} model found in artifacts/experiments/{controller}/.")
        return found

    candidate = (_PROJECT_ROOT / model).resolve() if not Path(model).is_absolute() else Path(model).resolve()
    if _ARTIFACTS_DIR not in candidate.parents and candidate != _ARTIFACTS_DIR:
        raise HTTPException(400, "Model path must be inside artifacts/.")
    if candidate.suffix != ".zip" or not candidate.exists():
        raise HTTPException(404, f"Model not found: {model}")
    return candidate


@router.post("/launch-gui", response_model=LaunchResponse, dependencies=[Depends(require_api_key)])
def launch_gui(
    controller: str = "ppo",
    model: Optional[str] = None,
    delay: int = 80,
    duration: int = 3600,
):
    """Launch a SUMO-GUI window for one controller. Call again with a different
    controller to open a second window and compare them live side by side."""
    if controller not in _CONTROLLERS:
        raise HTTPException(400, f"controller must be one of {_CONTROLLERS}")

    with _gui_lock:
        if _is_running(controller):
            proc = _gui_processes[controller]
            return LaunchResponse(
                status="already_running", pid=proc.pid,
                message=f"SUMO-GUI for '{controller}' already running.",
            )

        cmd = [sys.executable, str(_PROJECT_ROOT / "scripts" / "run_visual_demo.py"),
               "--controller", controller, "--delay", str(delay), "--duration", str(duration)]

        model_path: Optional[Path] = None
        if controller in _MODEL_BASED_CONTROLLERS:
            model_path = _resolve_model_path(model, controller)
            cmd += ["--model", str(model_path)]

        env = os.environ.copy()
        if "DISPLAY" not in env:
            env["DISPLAY"] = ":0"

        try:
            proc = subprocess.Popen(
                cmd, cwd=str(_PROJECT_ROOT), env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            raise HTTPException(500, f"Failed to launch SUMO-GUI: {exc}") from exc

        time.sleep(0.5)
        if proc.poll() is not None:
            raise HTTPException(500, f"SUMO-GUI for '{controller}' exited immediately.")

        _gui_processes[controller] = proc
        return LaunchResponse(
            status="launched",
            pid=proc.pid,
            model=str(model_path) if model_path else None,
            message=f"SUMO-GUI for '{controller}' launched (PID={proc.pid}). XQuartz window opening.",
        )


@router.post("/stop-gui", dependencies=[Depends(require_api_key)])
def stop_gui(controller: Optional[str] = None):
    """Stop one controller's SUMO-GUI window, or every running window if controller is omitted."""
    with _gui_lock:
        targets = [controller] if controller else list(_gui_processes.keys())
        stopped = []
        for name in targets:
            if not _is_running(name):
                continue
            proc = _gui_processes[name]
            pid = proc.pid
            # proc.terminate() only signals this Python wrapper — the actual
            # sumo-gui window it spawned via TraCI is a separate child process
            # that keeps running as an orphan. launch_gui uses
            # start_new_session=True, which makes this process its own
            # process-group leader, so killing the *group* (negative pid)
            # takes the sumo-gui window down with it. Verified: without this,
            # "stop" left the window open indefinitely.
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGTERM)
                proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except Exception:
                    pass
            del _gui_processes[name]
            stopped.append({"controller": name, "pid": pid})
        if not stopped:
            return {"status": "not_running", "message": "No matching SUMO-GUI process is running."}
        return {"status": "stopped", "stopped": stopped, "message": f"Stopped {len(stopped)} SUMO-GUI process(es)."}


@router.get("/status", response_model=StatusResponse)
def get_status():
    """Return current simulation status."""
    running = [c for c in _CONTROLLERS if _is_running(c)]
    model = _find_latest_model()
    first_pid = _gui_processes[running[0]].pid if running else None
    return StatusResponse(
        gui_running=bool(running),
        gui_pid=first_pid,
        latest_model=str(model) if model else None,
        xquartz_display=_get_display(),
        running_controllers=running,
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
            running = [c for c in _CONTROLLERS if _is_running(c)]
            payload = {
                "type": "status",
                "timestamp": time.time(),
                "gui_running": bool(running),
                "gui_pid": _gui_processes[running[0]].pid if running else None,
                "running_controllers": running,
                "metrics": metrics,
            }
            await websocket.send_json(payload)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
