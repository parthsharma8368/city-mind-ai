"""replay.py — Frame-by-frame trajectory serving for visualization."""
from __future__ import annotations
import logging
import re
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
import pandas as pd

logger = logging.getLogger("citymind.api.replay")
router = APIRouter()
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAJECTORY_DIR = _PROJECT_ROOT / "artifacts" / "trajectories"

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _validate_token(value: str, field: str) -> None:
    """Reject anything that isn't a plain filename-safe token (blocks path traversal)."""
    if not _SAFE_TOKEN.match(value):
        raise HTTPException(400, f"Invalid {field}: {value!r}")


MAX_WINDOW_SECONDS = 600.0  # frontend asks for 300 s; leave headroom, hard-cap anything larger
MAX_ROWS = 1_500_000        # backstop on rows materialised per response
_COLUMNS = ["time", "vehicle_id", "x", "y", "speed", "tls_id", "phase", "tls_state"]


def _read_window(parquet_path: Path, csv_path: Path, start: float, end: float) -> pd.DataFrame:
    """Read only the rows in [start, end] (filter pushed into the parquet scan)."""
    if parquet_path.exists():
        import pyarrow.parquet as pq

        names = set(pq.read_schema(parquet_path).names)
        cols = [c for c in _COLUMNS if c in names]
        table = pq.read_table(
            parquet_path, columns=cols,
            filters=[("time", ">=", start), ("time", "<=", end)],
        )
        return table.to_pandas()
    if csv_path.exists():
        header = pd.read_csv(csv_path, nrows=0).columns
        cols = [c for c in _COLUMNS if c in header]
        parts = []
        for chunk in pd.read_csv(csv_path, usecols=cols, chunksize=200_000):
            parts.append(chunk[(chunk["time"] >= start) & (chunk["time"] <= end)])
        return pd.concat(parts) if parts else pd.DataFrame(columns=cols)
    raise FileNotFoundError(parquet_path.name)


@router.get("/frames")
def get_trajectory_frames(
    controller: str = Query(...),
    seed: int = Query(42),
    demand_id: str = Query("1.0x"),
    start_time: float = Query(0.0, ge=0, allow_inf_nan=False),
    end_time: float = Query(60.0, ge=0, allow_inf_nan=False),
):
    """Return trajectory frames for the specified controller/seed/demand between start and end time."""
    _validate_token(controller, "controller")
    _validate_token(demand_id, "demand_id")
    if end_time < start_time or end_time - start_time > MAX_WINDOW_SECONDS:
        raise HTTPException(
            422, f"end_time must be >= start_time and the window at most {MAX_WINDOW_SECONDS:.0f} s."
        )

    parquet_path = TRAJECTORY_DIR / f"{controller}_seed{seed}_{demand_id}.parquet"
    csv_path = TRAJECTORY_DIR / f"{controller}_seed{seed}_{demand_id}.csv"

    try:
        subset = _read_window(parquet_path, csv_path, start_time, end_time)
    except FileNotFoundError:
        raise HTTPException(
            404, f"Trajectory not found for {controller} (seed={seed}, demand={demand_id})"
        ) from None
    except Exception as exc:
        logger.warning("could not read trajectory %s: %s", parquet_path.name, exc)
        raise HTTPException(500, f"Could not read trajectory for {controller}.") from exc
    if len(subset) > MAX_ROWS:
        raise HTTPException(413, "Requested window contains too many rows; use a smaller window.")

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
