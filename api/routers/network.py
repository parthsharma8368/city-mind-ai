"""network.py — Serve GeoJSON + TLS metadata for the frontend map."""
from __future__ import annotations
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

router = APIRouter()
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METADATA_DIR = _PROJECT_ROOT / "artifacts" / "scenario"
GEOJSON_PATH = _PROJECT_ROOT / "data" / "processed" / "network.geojson"


def _load_json_or_404(path: Path, missing_detail: str):
    if not path.exists():
        raise HTTPException(404, missing_detail)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"{path.name} is not valid JSON: {exc}") from exc


@router.get("/tls_metadata")
def get_tls_metadata():
    path = METADATA_DIR / "tls_metadata.json"
    return _load_json_or_404(path, "TLS metadata not found. Run inspect_intas.py first.")

@router.get("/network_metadata")
def get_network_metadata():
    path = METADATA_DIR / "network_metadata.json"
    return _load_json_or_404(path, "Network metadata not found.")

@router.get("/summary")
def get_scenario_summary():
    path = METADATA_DIR / "scenario_summary.json"
    return _load_json_or_404(path, "Scenario summary not found.")

@router.get("/geojson")
def get_network_geojson():
    return _load_json_or_404(GEOJSON_PATH, "Network GeoJSON not found. Run inspect_intas.py first.")
