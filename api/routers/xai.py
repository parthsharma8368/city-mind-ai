"""
xai.py — FastAPI router for Explainable AI (XAI) queries and visualizations.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
XAI_DIR = _PROJECT_ROOT / "artifacts" / "reports" / "xai"
from fastapi.responses import FileResponse

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.\-]+$")


_CONTROLLERS = {"ppo", "ippo"}


def _validate_tls_id(tls_id: str) -> None:
    """Reject anything that isn't a plain filename-safe token (blocks path traversal)."""
    if not _SAFE_TOKEN.match(tls_id):
        raise HTTPException(400, f"Invalid tls_id: {tls_id!r}")


def _xai_dir(controller: str) -> Path:
    if controller not in _CONTROLLERS:
        raise HTTPException(400, f"controller must be one of {sorted(_CONTROLLERS)}")
    # IPPO's artifacts live in a subdirectory (see scripts/run_xai.py) so
    # they never collide with PPO's - they're a structurally different kind
    # of explanation (per-junction local attribution + an empirical
    # spillback-correlation heatmap, not a model-attribution coordination
    # matrix - see src/evaluation/xai.py's module docstring).
    return XAI_DIR / "ippo" if controller == "ippo" else XAI_DIR


def _load_xai_summary(controller: str) -> list[dict]:
    json_path = _xai_dir(controller) / "xai_summary.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"XAI summary not found for {controller}. The XAI analysis was retired with the legacy agents.")
    try:
        return json.loads(json_path.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"xai_summary.json is not valid JSON: {exc}") from exc


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
def get_xai_summary(controller: str = "ppo"):
    """Retrieve the latest precomputed XAI explanations for one controller."""
    return _load_xai_summary(controller)


@router.get("/explain/{tls_id}", response_model=JunctionExplanation)
def get_junction_explanation(tls_id: str, controller: str = "ppo"):
    """Retrieve explanation for a specific traffic light junction."""
    _validate_tls_id(tls_id)
    data = _load_xai_summary(controller)
    for entry in data:
        if entry.get("tls_id") == tls_id:
            return entry
    raise HTTPException(status_code=404, detail=f"TLS ID '{tls_id}' not found in latest {controller} XAI report.")


@router.get("/coordination-heatmap")
def get_coordination_heatmap(controller: str = "ppo"):
    """Serve the precomputed coordination heatmap image (PPO: model-attribution matrix; IPPO: empirical spillback correlation — see /meta)."""
    img_path = _xai_dir(controller) / "coordination_heatmap.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Coordination heatmap image not found for {controller}.")
    return FileResponse(str(img_path), media_type="image/png")


@router.get("/driver-plot/{tls_id}")
def get_driver_plot(tls_id: str, controller: str = "ppo"):
    """Serve the feature attribution driver plot for a specific junction."""
    _validate_tls_id(tls_id)
    img_path = _xai_dir(controller) / f"drivers_{tls_id}.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Driver plot for junction '{tls_id}' not found for {controller}.")
    return FileResponse(str(img_path), media_type="image/png")


@router.get("/meta")
def get_xai_meta(controller: str = "ppo"):
    """
    Describes what *kind* of evidence this controller's coordination heatmap
    is. PPO's is a gradient-based model attribution; IPPO's architecture has
    no cross-agent gradient path (see src/evaluation/xai.py's module
    docstring), so its heatmap is instead an empirical correlation of
    recorded outcomes — a different, still-real, but not equivalent kind of
    evidence. The frontend uses this to label each one honestly rather than
    presenting them as the same computation.
    """
    if controller not in _CONTROLLERS:
        raise HTTPException(400, f"controller must be one of {sorted(_CONTROLLERS)}")
    if controller == "ppo":
        return {
            "controller": "ppo",
            "explains": "Centralized PPO — one policy jointly observing all 98 intersections.",
            "coordination_evidence_type": "model_attribution",
            "coordination_evidence_label": "Model Attribution Matrix",
            "coordination_evidence_description": "Integrated-Gradients attribution of each junction's decision back across the full joint observation, bucketed by which other junction's features each attributed value came from.",
        }
    return {
        "controller": "ippo",
        "explains": "IPPO — one shared policy invoked independently per intersection, each seeing only its own local observation.",
        "coordination_evidence_type": "empirical_correlation",
        "coordination_evidence_label": "Empirical Spillback Correlation",
        "coordination_evidence_description": "Pearson correlation of recorded phase-switch timing between geographically-nearby intersections over an evaluated episode — not a model attribution, since IPPO's architecture has no gradient path between one intersection's decision and a specific neighbor's state.",
    }

