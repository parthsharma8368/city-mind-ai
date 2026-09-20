"""
models.py — Pydantic schemas for the FastAPI backend.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel


class ExperimentRun(BaseModel):
    run_id: str
    algorithm: str
    seed: int
    total_timesteps: int
    model_path: Optional[str] = None
    config: dict[str, Any] = {}
