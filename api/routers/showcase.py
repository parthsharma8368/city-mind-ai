"""showcase.py — Data for the dashboard's one-click showcase.

* ``/frames``  a compact, columnar slice of one recorded run (cars + signal colours at every 5 s frame), small enough that
               the four recorded runs can be fetched together and played side by side at 10x.
* ``/metrics/{controller}``  the stored evaluation of one controller: seed means plus the per-seed values.
* ``/config``  the replay window, the playback speed and which controllers have a recording.
* ``/policy/{controller}``  the trained weights of a final model (actor and critic networks), for the "inside the model" figure.

Everything is read from the evaluation results (api/results.py); nothing is simulated here.
"""
from __future__ import annotations

import functools
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response

from api import results
from api.routers.simulation import MODEL_FILES
from api.util import sanitize_nan
from scripts.report_eval import METRICS as MEASURES  # (key, label, "lower"|"higher", format)

logger = logging.getLogger("citymind.api.showcase")
router = APIRouter()

SEED = results.DEFAULT_SEED
WINDOW_START, WINDOW_END = 26400, 27000          # 07:20-07:30: the network is full and the controllers have pulled apart
FRAME_STEP = 5                                    # seconds between recorded frames
PLAYBACK_SPEED = 10                               # the showcase plays 10 simulated seconds per real second
HOUR_START, HOUR_END = 25200, 28800
MAX_WINDOW_SECONDS = 900
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.\-]+$")
# Coordinates are sent relative to this origin (InTAS metres) so the numbers stay short.
ORIGIN_X, ORIGIN_Y = 209000, 446000
_COLUMNS = ["time", "vehicle_id", "x", "y", "speed", "tls_id", "tls_state"]


def _signal_char(state: Any) -> str:
    """g / y / r for a recorded signal-state string, n when there is none. A phase shows yellow while any link is yellow,
    otherwise green when at least as many links are green as red (most phases leave some link green, so "any green" would
    paint every junction green all the time)."""
    s = str(state or "").lower()
    if not s or s == "nan":
        return "n"
    if "y" in s:
        return "y"
    greens, reds = s.count("g"), s.count("r") + s.count("s")
    if greens == 0 and reds == 0:
        return "n"
    return "g" if greens >= reds else "r"


@functools.lru_cache(maxsize=16)
def _frames_bytes(controller: str, seed: int, demand_id: str, start: int, end: int, mtime: float) -> bytes:  # noqa: ARG001
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    path = results.trajectory_path(controller, seed, demand_id)
    names = set(pq.read_schema(path).names)
    cols = [c for c in _COLUMNS if c in names]
    table = pq.read_table(path, columns=cols, filters=[("time", ">=", start), ("time", "<=", end)]).to_pandas()
    if table.empty:
        raise LookupError("no frames in that window")

    times = np.sort(table["time"].unique())
    veh = table[table["vehicle_id"].notna()].copy()
    codes, uniques = pd.factorize(veh["vehicle_id"])
    veh["code"] = codes
    veh = veh.sort_values(["time", "code"], kind="stable")      # per frame, cars in code order
    counts = veh.groupby("time").size().reindex(times, fill_value=0).astype(int)

    tls_ids: list[str] = []
    tls_frames: list[str] = []
    if "tls_id" in table.columns:
        tls = table[table["tls_id"].notna()].copy()
        tls_ids = sorted(tls["tls_id"].unique().tolist())
        tls["kind"] = tls["tls_state"].map(_signal_char) if "tls_state" in tls.columns else "n"
        index = {t: i for i, t in enumerate(tls_ids)}
        by_time = {t: g for t, g in tls.groupby("time")}
        for t in times:
            chars = ["n"] * len(tls_ids)
            g = by_time.get(t)
            if g is not None:
                for tid, kind in zip(g["tls_id"], g["kind"]):
                    chars[index[tid]] = kind
            tls_frames.append("".join(chars))

    payload = {
        "controller": controller, "seed": seed, "start": float(times[0]), "end": float(times[-1]), "step": FRAME_STEP,
        "origin": [ORIGIN_X, ORIGIN_Y],
        "times": [float(t) for t in times],
        "counts": counts.tolist(),
        "vid": veh["code"].astype(int).tolist(),
        "x": (veh["x"] - ORIGIN_X).round(1).tolist(),
        "y": (veh["y"] - ORIGIN_Y).round(1).tolist(),
        "v": veh["speed"].round(1).tolist(),
        "vehicle_count": int(len(uniques)),
        "tls_ids": tls_ids,
        "tls": tls_frames,
    }
    return json.dumps(payload, separators=(",", ":")).encode()


@router.get("/config")
def get_config():
    """Window, speed and the controllers that have a recording."""
    return {
        "seed": SEED, "start": WINDOW_START, "end": WINDOW_END, "step": FRAME_STEP, "speed": PLAYBACK_SPEED,
        "hour": [HOUR_START, HOUR_END], "demand_id": "1.0x",
        "controllers": [
            {
                "id": c, "label": results.label(c), "kind": "model" if c in results.MODELS else "baseline",
                "has_recording": results.trajectory_path(c, SEED).is_file(),
                "has_metrics": bool(results.load_metrics(c, SEED)),
            }
            for c in results.CONTROLLERS
        ],
        "n_seeds": len(results.seeds()),
    }


@router.get("/frames")
def get_frames(
    controller: str = Query(...),
    seed: int = Query(SEED),
    demand_id: str = Query("1.0x"),
    start: int = Query(WINDOW_START, ge=HOUR_START, le=HOUR_END),
    end: int = Query(WINDOW_END, ge=HOUR_START, le=HOUR_END),
):
    """Compact frames of one recorded run: parallel arrays (car index, x, y, speed) with a per-frame count, plus one
    g/y/r/n character per junction per frame."""
    if not _SAFE_TOKEN.match(demand_id):
        raise HTTPException(400, "Invalid demand_id")
    if controller not in results.CONTROLLERS:
        raise HTTPException(404, f"Unknown controller {controller!r}; known: {list(results.CONTROLLERS)}")
    if end <= start or end - start > MAX_WINDOW_SECONDS:
        raise HTTPException(422, f"end must be after start and the window at most {MAX_WINDOW_SECONDS} s.")
    path = results.trajectory_path(controller, seed, demand_id)
    if not path.is_file():
        raise HTTPException(404, f"No recording for {controller} (seed {seed}).")
    try:
        body = _frames_bytes(controller, seed, demand_id, start, end, path.stat().st_mtime)
    except LookupError:
        raise HTTPException(404, "The recording has no frames in that window.") from None
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not build frames for %s: %s", path.name, exc)
        raise HTTPException(500, f"Could not read the recording for {controller}.") from exc
    return Response(content=body, media_type="application/json", headers={"Cache-Control": "private, max-age=300"})


@router.get("/metrics/{controller}")
def get_controller_metrics(controller: str):
    """Stored evaluation of one controller: mean over the evaluated seeds, the per-seed values, and where they come from."""
    if controller not in results.CONTROLLERS:
        raise HTTPException(404, f"Unknown controller {controller!r}; known: {list(results.CONTROLLERS)}")
    table = results.per_seed_table()
    if table is None:
        raise HTTPException(404, "No evaluation results found. Run scripts/evaluate.py and scripts/report_eval.py first.")
    rows = table[table["name"] == controller].sort_values("seed")
    if rows.empty:
        raise HTTPException(404, f"No evaluation results for {controller}.")
    measures = []
    for key, label, better, fmt in MEASURES:
        if key not in rows.columns:
            continue
        values = [float(v) for v in rows[key].tolist()]
        measures.append({
            "key": key, "label": label, "better": better, "format": fmt,
            "mean": float(rows[key].mean()), "min": min(values), "max": max(values), "per_seed": values,
        })
    return sanitize_nan({
        "controller": controller,
        "label": results.label(controller),
        "kind": "model" if controller in results.MODELS else "baseline",
        "seeds": [int(s) for s in rows["seed"].tolist()],
        "n_runs": int(len(rows)),
        "window": "07:00-08:00 (one simulated peak hour per seed)",
        "source": results.RESULTS_ROOT.name,
        "measures": measures,
    })


# ── the networks themselves ─────────────────────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# SB3 stores the policy as a plain torch state dict (policy.pth) inside the zip; only the weights are read, never the pickled
# model object, and only the two fixed files of MODEL_FILES (a request can never name a path).
_ACTOR_KEYS = ("mlp_extractor.policy_net.0", "mlp_extractor.policy_net.2", "action_net")
_CRITIC_KEYS = ("mlp_extractor.value_net.0", "mlp_extractor.value_net.2", "value_net")
OBS_NAMES = (
    [f"phase {i + 1} is green" for i in range(6)] + ["yellow / all-red running", "time in phase vs its minimum",
    "time in phase vs its maximum"] + [f"phase {i + 1} demand" for i in range(6)] + ["worst queue", "cars waiting outside",
    "number of phases", "upstream queue 1", "upstream queue 2", "downstream queue 1", "downstream queue 2"]
)


@functools.lru_cache(maxsize=4)
def _policy_payload(controller: str, mtime: float) -> dict[str, Any]:  # noqa: ARG001
    import io

    import torch

    path = _PROJECT_ROOT / MODEL_FILES[controller]
    with zipfile.ZipFile(path) as zf:
        state = torch.load(io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=True)

    def flat(t: Any) -> list[float]:
        return [round(float(x), 4) for x in t.reshape(-1).tolist()]

    def net(keys: tuple[str, str, str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, key in zip(("l1", "l2", "out"), keys, strict=True):
            w, b = state[f"{key}.weight"], state[f"{key}.bias"]
            out[name] = {"shape": [int(w.shape[0]), int(w.shape[1])], "w": flat(w), "b": flat(b)}
        return out

    actor, critic = net(_ACTOR_KEYS), net(_CRITIC_KEYS)
    if actor["l1"]["shape"][1] != len(OBS_NAMES) or actor["out"]["shape"][0] != 2 or critic["out"]["shape"][0] != 1:
        raise ValueError(f"{path.name} does not have the expected 22 -> 256 -> 256 -> 2 / 1 shape")
    return {
        "controller": controller,
        "label": results.label(controller),
        "obs_names": OBS_NAMES,
        "actions": ["keep this green", "advance to the next green"],
        "activation": "tanh",
        "actor": actor,
        "critic": critic,
    }


@router.get("/policy/{controller}")
def get_policy(controller: str):
    """Weights of a final model's two networks: actor 22 -> 256 -> 256 -> 2 (keep / advance) and critic 22 -> 256 -> 256 -> 1."""
    if controller not in MODEL_FILES:
        raise HTTPException(404, f"No trained model {controller!r}; known: {list(MODEL_FILES)}")
    path = _PROJECT_ROOT / MODEL_FILES[controller]
    if not path.is_file():
        raise HTTPException(404, f"Model file {MODEL_FILES[controller]} is missing (see docs/MODELS.md).")
    try:
        payload = _policy_payload(controller, path.stat().st_mtime)
    except Exception as exc:  # noqa: BLE001 - corrupt zip / unexpected layout: say which model, keep the cause in the log
        logger.warning("could not read the weights of %s: %s", controller, exc)
        raise HTTPException(500, f"Could not read the weights of {controller}.") from exc
    return Response(content=json.dumps(payload, separators=(",", ":")), media_type="application/json")
