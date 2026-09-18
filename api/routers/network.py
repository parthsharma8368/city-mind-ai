"""network.py — Serve GeoJSON + TLS metadata for the frontend map."""
from __future__ import annotations
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

router = APIRouter()
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_METADATA_DIR = _PROJECT_ROOT / "artifacts" / "scenario"
DEFAULT_GEOJSON_PATH = _PROJECT_ROOT / "data" / "processed" / "network.geojson"

@router.get("/tls_metadata")
def get_tls_metadata(metadata_dir: str | None = None):
    base = Path(metadata_dir) if metadata_dir else DEFAULT_METADATA_DIR
    path = base / "tls_metadata.json"
    if not path.exists():
        raise HTTPException(404, "TLS metadata not found. Run inspect_intas.py first.")
    return json.loads(path.read_text())

@router.get("/network_metadata")
def get_network_metadata(metadata_dir: str | None = None):
    base = Path(metadata_dir) if metadata_dir else DEFAULT_METADATA_DIR
    path = base / "network_metadata.json"
    if not path.exists():
        raise HTTPException(404, "Network metadata not found.")
    return json.loads(path.read_text())

@router.get("/summary")
def get_scenario_summary(metadata_dir: str | None = None):
    base = Path(metadata_dir) if metadata_dir else DEFAULT_METADATA_DIR
    path = base / "scenario_summary.json"
    if not path.exists():
        raise HTTPException(404, "Scenario summary not found.")
    return json.loads(path.read_text())

@router.get("/geojson")
def get_network_geojson(geojson_path: str | None = None):
    path = Path(geojson_path) if geojson_path else DEFAULT_GEOJSON_PATH
    if not path.exists():
        # Fallback to citymind-ai path
        path = _PROJECT_ROOT / "citymind-ai" / "data" / "processed" / "network.geojson"
    if not path.exists():
        raise HTTPException(404, "Network GeoJSON not found.")
    return json.loads(path.read_text())

