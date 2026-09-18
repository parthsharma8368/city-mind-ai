"""network.py — Serve GeoJSON + TLS metadata for the frontend map."""
from __future__ import annotations
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

router = APIRouter()
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METADATA_DIR = _PROJECT_ROOT / "artifacts" / "scenario"
GEOJSON_PATH = _PROJECT_ROOT / "data" / "processed" / "network.geojson"


@router.get("/tls_metadata")
def get_tls_metadata():
    path = METADATA_DIR / "tls_metadata.json"
    if not path.exists():
        raise HTTPException(404, "TLS metadata not found. Run inspect_intas.py first.")
    return json.loads(path.read_text())

@router.get("/network_metadata")
def get_network_metadata():
    path = METADATA_DIR / "network_metadata.json"
    if not path.exists():
        raise HTTPException(404, "Network metadata not found.")
    return json.loads(path.read_text())

@router.get("/summary")
def get_scenario_summary():
    path = METADATA_DIR / "scenario_summary.json"
    if not path.exists():
        raise HTTPException(404, "Scenario summary not found.")
    return json.loads(path.read_text())

@router.get("/geojson")
def get_network_geojson():
    if not GEOJSON_PATH.exists():
        raise HTTPException(404, "Network GeoJSON not found. Run inspect_intas.py first.")
    return json.loads(GEOJSON_PATH.read_text())
