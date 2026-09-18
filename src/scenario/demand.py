"""
demand.py
=========
Demand file utilities for InTAS.

Week 1: Reads the original InTAS demand at 1.0× scale.
Post-week: Add demand scaling via route departure time manipulation.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_demand_metadata(json_path: Path) -> dict:
    """Load demand metadata artifact from inspect_intas.py output."""
    with open(json_path) as f:
        return json.load(f)


def get_route_files(demand_meta: dict) -> list[str]:
    """Return list of route file paths from demand metadata."""
    return demand_meta.get("route_files", [])


def get_additional_files(demand_meta: dict) -> list[str]:
    """Return list of additional file paths (TLS programs, etc.)."""
    return demand_meta.get("additional_files", [])
