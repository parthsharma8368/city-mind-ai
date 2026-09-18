"""
xai.py — FastAPI router for Explainable AI (XAI) queries and visualizations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
XAI_DIR = _PROJECT_ROOT / "artifacts" / "reports" / "xai"
from fastapi.responses import FileResponse


class FeatureItem(BaseModel):
    feature_index: int
    name: str
    group: str
    tls_id: str
    lane_id: str | None = None
    value: float
    attribution: float


class JunctionExplanation(BaseModel):
    tls_id: str
    tls_index: int
    action: int
    switch_probability: float
    narrative: str
    local_influence_pct: float
    neighbor_influence_pct: float
    group_breakdown: Dict[str, float]
    top_positive_features: List[FeatureItem]
    top_negative_features: List[FeatureItem]


@router.get("/summary", response_model=List[JunctionExplanation])
def get_xai_summary():
    """Retrieve the latest precomputed XAI explanations."""
    json_path = XAI_DIR / "xai_summary.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="XAI summary not found. Run scripts/run_xai.py first.")
    with open(json_path) as f:
        data = json.load(f)
    return data


@router.get("/explain/{tls_id}", response_model=JunctionExplanation)
def get_junction_explanation(tls_id: str):
    """Retrieve explanation for a specific traffic light junction."""
    json_path = XAI_DIR / "xai_summary.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="XAI summary not found.")
    with open(json_path) as f:
        data = json.load(f)
    for entry in data:
        if entry["tls_id"] == tls_id:
            return entry
    raise HTTPException(status_code=404, detail=f"TLS ID '{tls_id}' not found in latest XAI report.")


@router.get("/coordination-heatmap")
def get_coordination_heatmap():
    """Serve the precomputed coordination heatmap image."""
    img_path = XAI_DIR / "coordination_heatmap.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Coordination heatmap image not found.")
    return FileResponse(str(img_path), media_type="image/png")


@router.get("/driver-plot/{tls_id}")
def get_driver_plot(tls_id: str):
    """Serve the feature attribution driver plot for a specific junction."""
    img_path = XAI_DIR / f"drivers_{tls_id}.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Driver plot for junction '{tls_id}' not found.")
    return FileResponse(str(img_path), media_type="image/png")

