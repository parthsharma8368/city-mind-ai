"""replay.py — Frame-by-frame trajectory serving for visualization."""
from __future__ import annotations
import re
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
import pandas as pd

router = APIRouter()
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAJECTORY_DIR = _PROJECT_ROOT / "artifacts" / "trajectories"

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _validate_token(value: str, field: str) -> None:
    """Reject anything that isn't a plain filename-safe token (blocks path traversal)."""
    if not _SAFE_TOKEN.match(value):
        raise HTTPException(400, f"Invalid {field}: {value!r}")


@router.get("/frames")
def get_trajectory_frames(
    controller: str = Query(...),
    seed: int = Query(42),
    demand_id: str = Query("1.0x"),
    start_time: float = Query(0.0),
    end_time: float = Query(60.0),
):
    """Return trajectory frames for the specified controller/seed/demand between start and end time."""
    _validate_token(controller, "controller")
    _validate_token(demand_id, "demand_id")

    parquet_path = TRAJECTORY_DIR / f"{controller}_seed{seed}_{demand_id}.parquet"
    csv_path = TRAJECTORY_DIR / f"{controller}_seed{seed}_{demand_id}.csv"

    try:
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            raise HTTPException(404, f"Trajectory not found for {controller} (seed={seed}, demand={demand_id})")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Could not read trajectory for {controller}: {exc}") from exc

    mask = (df["time"] >= start_time) & (df["time"] <= end_time)
    subset = df[mask]

    vehicle_frames = {}
    tls_by_time: dict[str, list[dict]] = {}
    has_tls_columns = {"tls_id", "phase", "tls_state"}.issubset(subset.columns)

    for time_val, group in subset.groupby("time"):
        key = str(time_val)
        vehicle_frames[key] = (
            group.dropna(subset=["vehicle_id"])[["vehicle_id", "x", "y", "speed"]].to_dict("records")
        )
        if has_tls_columns:
            tls_rows = (
                group.dropna(subset=["tls_id"])[["tls_id", "phase", "tls_state"]]
                .drop_duplicates(subset=["tls_id"])
                .rename(columns={"tls_state": "state"})
                .to_dict("records")
            )
            tls_by_time[key] = tls_rows

    return {
        "frames": vehicle_frames,
        "tls_frames": tls_by_time,
        "frame_count": len(vehicle_frames),
    }

@router.get("/available")
def list_trajectories():
    """List all available trajectory files."""
    if not TRAJECTORY_DIR.exists():
        return {"trajectories": []}
    files = sorted(
        {f.stem for f in TRAJECTORY_DIR.glob("*.parquet")}
        | {f.stem for f in TRAJECTORY_DIR.glob("*.csv")}
    )
    return {"trajectories": files}
