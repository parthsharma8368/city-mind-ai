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
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from api.routers.experiments import _load_yaml, _sanitize_json
from api.security import require_api_key, websocket_allowed
from api.util import read_json_file, sanitize_nan

logger = logging.getLogger("citymind.api.simulation")

router = APIRouter()

# Keyed by controller name ("fixed_time" | "max_pressure" | "ppo" | "ippo") so
# multiple controllers can run as separate SUMO-GUI windows at once — the
# whole point is comparing a specific intersection live between two windows.
_gui_processes: dict[str, subprocess.Popen] = {}
# Reentrant: _is_running is called both standalone and from inside locked sections.
# The registry is per-process, so run a single uvicorn worker.
_gui_lock = threading.RLock()
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ARTIFACTS_DIR = (_PROJECT_ROOT / "artifacts").resolve()
_EXPERIMENTS_DIR = (_PROJECT_ROOT / "artifacts" / "experiments").resolve()
_MAX_WS_CONNECTIONS = 20
_ws_connections = 0
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


def _rel(path: Path) -> str:
    """Repo-root-relative path; never expose absolute server paths to clients."""
    try:
        return path.resolve().relative_to(_PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _is_running(controller: str) -> bool:
    with _gui_lock:
        proc = _gui_processes.get(controller)
        if proc is None:
            return False
        if proc.poll() is not None:
            _gui_processes.pop(controller, None)
            return False
        return True


def _running_snapshot() -> tuple[list[str], Optional[int]]:
    """Running controllers and the first PID, read atomically under the lock."""
    with _gui_lock:
        running = [c for c in _CONTROLLERS if _is_running(c)]
        pid = _gui_processes[running[0]].pid if running else None
    return running, pid


def _resolve_model_path(model: Optional[str], controller: str) -> Path:
    """Resolve a requested model path and require it to be a .zip under artifacts/experiments/.

    SECURITY: SB3 model zips contain pickled data executed on PPO.load - treat them as code.
    """
    if not model:
        found = _find_latest_model(controller)
        if found is None:
            raise HTTPException(404, f"No trained {controller} model found in artifacts/experiments/{controller}/.")
        return found

    candidate = (_PROJECT_ROOT / model).resolve() if not Path(model).is_absolute() else Path(model).resolve()
    if _EXPERIMENTS_DIR not in candidate.parents:
        raise HTTPException(400, "Model path must be inside artifacts/experiments/.")
    if candidate.suffix != ".zip" or not candidate.is_file():
        raise HTTPException(404, f"Model not found: {model}")
    return candidate


@router.post("/launch-gui", response_model=LaunchResponse, dependencies=[Depends(require_api_key)])
def launch_gui(
    controller: str = "ppo",
    model: Optional[str] = None,
    delay: int = Query(80, ge=0, le=1000),
    duration: int = Query(3600, ge=60, le=86400),
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
            model=_rel(model_path) if model_path else None,
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
            _gui_processes.pop(name, None)
            stopped.append({"controller": name, "pid": pid})
        if not stopped:
            return {"status": "not_running", "message": "No matching SUMO-GUI process is running."}
        return {"status": "stopped", "stopped": stopped, "message": f"Stopped {len(stopped)} SUMO-GUI process(es)."}


@router.get("/status", response_model=StatusResponse)
def get_status():
    """Return current simulation status."""
    running, first_pid = _running_snapshot()
    model = _find_latest_model()
    return StatusResponse(
        gui_running=bool(running),
        gui_pid=first_pid,
        latest_model=_rel(model) if model else None,
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
            config = _sanitize_json(_load_yaml(cfg_path.read_text()))
        except OSError as exc:
            logger.warning("could not read %s: %s", cfg_path.name, exc)
    return {
        "model_path": _rel(model),
        "run_id": run_dir.name,
        "modified": model.stat().st_mtime,
        "config": config,
    }


def _collect_metrics() -> dict[str, Any]:
    """Blocking file I/O; run via asyncio.to_thread so the event loop is never stalled."""
    metrics_dir = _PROJECT_ROOT / "artifacts" / "metrics"
    metrics: dict[str, Any] = {}
    for json_file in metrics_dir.glob("*.json"):
        data = read_json_file(json_file, retries=1)
        if not isinstance(data, dict):
            continue
        ctrl = data.get("controller", json_file.stem)
        metrics[ctrl] = {
            k: data.get(k, 0)
            for k in ("mean_waiting_time", "mean_queue", "throughput", "mean_speed",
                      "total_arrived", "cumulative_reward")
        }
    return sanitize_nan(metrics)


@router.websocket("/ws")
async def simulation_ws(websocket: WebSocket):
    """WebSocket: broadcast live status + metrics every 2s.

    The handshake is rejected (before accept) unless the Origin is allow-listed and
    the client is authorised (API key when configured, loopback otherwise).
    """
    global _ws_connections
    if not websocket_allowed(websocket) or _ws_connections >= _MAX_WS_CONNECTIONS:
        await websocket.close(code=1008)
        return
    subprotocol = next(
        (p.strip() for p in (websocket.headers.get("sec-websocket-protocol") or "").split(",")
         if p.strip().startswith("citymind-key.")),
        None,
    )
    _ws_connections += 1
    try:
        await websocket.accept(subprotocol=subprotocol)
        while True:
            metrics = await asyncio.to_thread(_collect_metrics)
            running, pid = _running_snapshot()
            payload = {
                "type": "status",
                "timestamp": time.time(),
                "gui_running": bool(running),
                "gui_pid": pid,
                "running_controllers": running,
                "metrics": metrics,
            }
            await websocket.send_json(payload)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.info("simulation websocket closed: %s", exc)
    finally:
        _ws_connections -= 1
