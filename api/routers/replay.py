"""replay.py — Frame-by-frame trajectory serving for visualization."""
from __future__ import annotations
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
import pandas as pd

router = APIRouter()
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TRAJECTORY_DIR = _PROJECT_ROOT / "artifacts" / "trajectories"

@router.get("/frames")
def get_trajectory_frames(
    controller: str = Query(...),
    seed: int = Query(42),
    demand_id: str = Query("1.0x"),
    start_time: float = Query(0.0),
    end_time: float = Query(60.0),
    trajectory_dir: str | None = None,
):
    """Return trajectory frames for the specified controller/seed/demand between start and end time."""
    base = Path(trajectory_dir) if trajectory_dir else DEFAULT_TRAJECTORY_DIR
    parquet_path = base / f"{controller}_seed{seed}_{demand_id}.parquet"
    csv_path = base / f"{controller}_seed{seed}_{demand_id}.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise HTTPException(404, f"Trajectory not found for {controller} (seed={seed}, demand={demand_id})")

    mask = (df["time"] >= start_time) & (df["time"] <= end_time)
    subset = df[mask]

    frames = {}
    for time_val, group in subset.groupby("time"):
        frames[str(time_val)] = group[["vehicle_id", "x", "y", "speed"]].to_dict("records")

    return {"frames": frames, "frame_count": len(frames)}

@router.get("/available")
def list_trajectories(trajectory_dir: str | None = None):
    """List all available trajectory files."""
    base = Path(trajectory_dir) if trajectory_dir else DEFAULT_TRAJECTORY_DIR
    if not base.exists():
        return {"trajectories": []}
    files = sorted(list({f.stem for f in base.glob("*.parquet")} | {f.stem for f in base.glob("*.csv")}))
    return {"trajectories": files}

