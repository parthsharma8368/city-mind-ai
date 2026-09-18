"""
models.py — Pydantic schemas for the FastAPI backend.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel


class TLSStateFrame(BaseModel):
    id: str
    phase: int
    state: str  # SUMO signal state string, e.g. "GGrr"


class VehicleFrame(BaseModel):
    id: str
    x: float
    y: float
    speed: float


class TrajectoryFrame(BaseModel):
    time: float
    vehicles: list[VehicleFrame]
    tls: list[TLSStateFrame]


class MetricsSummary(BaseModel):
    controller: str
    seed: int
    demand_id: str
    mean_waiting_time: float
    mean_queue: float
    throughput: float
    total_arrived: int
    total_departed: int
    total_switches: int
    cumulative_reward: float = 0.0


class ExperimentRun(BaseModel):
    run_id: str
    algorithm: str
    seed: int
    total_timesteps: int
    model_path: Optional[str] = None
    config: dict[str, Any] = {}


class ComparisonRequest(BaseModel):
    controller_a: str
    controller_b: str
    demand_id: str = "1.0x"
    metric: str = "mean_waiting_time"


class ComparisonResult(BaseModel):
    controller_a: str
    controller_b: str
    metric: str
    demand_id: str
    a_mean: float
    a_std: float
    b_mean: float
    b_std: float
    improvement_pct: float
