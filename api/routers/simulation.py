"""
simulation.py — Simulation control endpoints for CityMind AI.

Provides:
  - POST /api/simulation/launch-gui  → open a live SUMO-GUI window for one controller
                                        (sumo_actuated | fixed_time | ippo_A | ippo_B)
  - POST /api/simulation/stop-gui    → close one (or every) window
  - GET  /api/simulation/status      → which windows are running, which controllers can be launched
  - WebSocket /ws/simulation         → stream live status + stored metrics
The client only ever sends a controller id: the model files are fixed on the server (MODEL_FILES), never taken from a request.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from api import results
from api.security import require_api_key, websocket_allowed
from api.util import sanitize_nan

logger = logging.getLogger("citymind.api.simulation")

router = APIRouter()

# Keyed by controller id so several controllers can run as separate SUMO-GUI windows at once -
# the point is comparing the same junction live between two windows.
_gui_processes: dict[str, subprocess.Popen] = {}
# Reentrant: _is_running is called both standalone and from inside locked sections.
# The registry is per-process, so run a single uvicorn worker.
_gui_lock = threading.RLock()
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOG_DIR = _PROJECT_ROOT / "artifacts" / "demo_logs"
# Why a window stopped by itself (exit code + the end of its log), so the dashboard can say so instead of staying "starting".
_last_exit: dict[str, str] = {}
_MAX_WS_CONNECTIONS = 20
_ws_connections = 0
_CONTROLLERS = tuple(results.CONTROLLERS)   # ("sumo_actuated", "fixed_time", "ippo_A", "ippo_B")
# SECURITY: SB3 model zips are pickles (code on load). The server only ever loads these two files.
MODEL_FILES: dict[str, str] = {
    "ippo_A": "artifacts/models/ippo_final_seed42/model_final.zip",
    "ippo_B": "artifacts/models/ippo_final_seed1042/model_final.zip",
}
_XY = re.compile(r"^\d{1,5},\d{1,5}$")


class LaunchResponse(BaseModel):
    status: str
    pid: Optional[int] = None
    model: Optional[str] = None
    message: str


class StatusResponse(BaseModel):
    gui_running: bool
    gui_pid: Optional[int] = None
    xquartz_display: Optional[str] = None
    running_controllers: list[str] = []
    controllers: list[dict[str, Any]] = []
    last_errors: dict[str, str] = {}


def _get_display() -> str:
    return os.environ.get("DISPLAY", ":0")


def _rel(path: Path) -> str:
    """Repo-root-relative path; never expose absolute server paths to clients."""
    try:
        return path.resolve().relative_to(_PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _log_path(controller: str) -> Path:
    return _LOG_DIR / f"{controller}.log"


def _log_tail(controller: str, limit: int = 400) -> str:
    try:
        lines = [ln.strip() for ln in _log_path(controller).read_text(errors="replace").splitlines() if ln.strip()]
    except OSError:
        return ""
    # the last line that is not TraCI's connection-retry chatter
    useful = [ln for ln in lines if "Retrying in" not in ln and "Could not connect to TraCI" not in ln]
    return (useful[-1] if useful else (lines[-1] if lines else ""))[-limit:]


def _is_running(controller: str) -> bool:
    with _gui_lock:
        proc = _gui_processes.get(controller)
        if proc is None:
            return False
        code = proc.poll()
        if code is not None:
            _gui_processes.pop(controller, None)
            if code != 0:
                _last_exit[controller] = f"exited with code {code}: {_log_tail(controller) or 'no output'}"
            return False
        return True


def _running_snapshot() -> tuple[list[str], Optional[int]]:
    """Running controllers and the first PID, read atomically under the lock."""
    with _gui_lock:
        running = [c for c in _CONTROLLERS if _is_running(c)]
        pid = _gui_processes[running[0]].pid if running else None
    return running, pid


def _demo_script_exists() -> bool:
    return (_PROJECT_ROOT / "scripts" / "run_visual_demo.py").is_file()


def _model_file(controller: str) -> Optional[Path]:
    rel = MODEL_FILES.get(controller)
    return (_PROJECT_ROOT / rel) if rel else None


@router.post("/launch-gui", response_model=LaunchResponse, dependencies=[Depends(require_api_key)])
def launch_gui(
    controller: str = "sumo_actuated",
    delay: int = Query(80, ge=0, le=1000),
    duration: int = Query(3600, ge=60, le=86400),
    pos: Optional[str] = Query(None, description="window position X,Y (to tile several windows)"),
    size: Optional[str] = Query(None, description="window size W,H"),
):
    """Launch a SUMO-GUI window for one controller. Call again with a different controller to open a second window
    and compare them live side by side."""
    if controller not in _CONTROLLERS:
        raise HTTPException(400, f"controller must be one of {_CONTROLLERS}")
    for name, value in (("pos", pos), ("size", size)):
        if value is not None and not _XY.match(value):
            raise HTTPException(422, f"{name} must look like 120,80")
    model_path = _model_file(controller)
    if model_path is not None and not model_path.is_file():
        raise HTTPException(404, f"Model file {MODEL_FILES[controller]} is missing (see docs/MODELS.md).")
    if not _demo_script_exists():
        raise HTTPException(501, "scripts/run_visual_demo.py is missing; the live SUMO-GUI launcher is unavailable.")

    with _gui_lock:
        if _is_running(controller):
            proc = _gui_processes[controller]
            return LaunchResponse(
                status="already_running", pid=proc.pid,
                message=f"SUMO-GUI for '{controller}' already running.",
            )

        cmd = [sys.executable, str(_PROJECT_ROOT / "scripts" / "run_visual_demo.py"),
               "--controller", controller, "--delay", str(delay), "--duration", str(duration)]
        if pos:
            cmd += ["--window-pos", pos]
        if size:
            cmd += ["--window-size", size]

        env = os.environ.copy()
        env.setdefault("DISPLAY", ":0")
        _last_exit.pop(controller, None)
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(_log_path(controller), "wb") as log:
                proc = subprocess.Popen(
                    cmd, cwd=str(_PROJECT_ROOT), env=env,
                    stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except Exception as exc:
            raise HTTPException(500, f"Failed to launch SUMO-GUI: {exc}") from exc

        time.sleep(0.5)
        if proc.poll() is not None:
            raise HTTPException(500, f"SUMO-GUI for '{controller}' exited immediately: {_log_tail(controller) or 'see artifacts/demo_logs'}")

        _gui_processes[controller] = proc
        return LaunchResponse(
            status="launched",
            pid=proc.pid,
            model=MODEL_FILES.get(controller),
            message=f"SUMO-GUI for '{results.label(controller)}' is starting (PID={proc.pid}); the window opens in about half a minute.",
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


def _controller_info() -> list[dict[str, Any]]:
    out = []
    for c in _CONTROLLERS:
        model = _model_file(c)
        out.append({
            "id": c, "label": results.label(c), "kind": "model" if model else "baseline",
            "available": model is None or model.is_file(),
        })
    return out


@router.get("/status", response_model=StatusResponse)
def get_status():
    """Current windows and the controllers that can be launched."""
    running, first_pid = _running_snapshot()
    return StatusResponse(
        gui_running=bool(running),
        gui_pid=first_pid,
        xquartz_display=_get_display(),
        running_controllers=running,
        controllers=_controller_info(),
        last_errors=dict(_last_exit),
    )


def _collect_metrics() -> dict[str, Any]:
    """Seed-mean headline numbers of every controller (blocking file I/O; run via asyncio.to_thread)."""
    try:
        table = results.per_seed_table()
    except Exception as exc:  # noqa: BLE001 - a half-written results file must not kill the stream
        logger.info("could not read results table: %s", exc)
        return {}
    if table is None:
        return {}
    means = table.groupby("name").mean(numeric_only=True)
    keys = {"mean_waiting_time": "mean_waiting_time", "mean_queue": "mean_queue", "throughput": "throughput",
            "mean_speed": "mean_speed", "total_arrived": "trips_completed"}
    return sanitize_nan({
        name: {out: float(row[col]) for out, col in keys.items() if col in row} | {"cumulative_reward": 0.0}
        for name, row in means.iterrows() if name in results.CONTROLLERS
    })


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
