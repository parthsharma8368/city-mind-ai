"""
observation.py
==============
Global observation builder for the centralized RL environment.

Observation structure (concatenated flat vector):
  For each TLS (sorted by tls_id):
    [ one_hot_phase..., phase_elapsed_norm,
      queue_lane0_norm, queue_lane1_norm, ...,
      vehicles_lane0_norm, vehicles_lane1_norm, ...,
      speed_lane0_norm, speed_lane1_norm, ...,
      density_lane0_norm, density_lane1_norm, ... ]
  Global:
    [ total_vehicles_norm, total_waiting_time_norm ]

All values normalized to [0, 1] where possible.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

from src.environment.sumo_connection import traci

from src.scenario.tls_metadata import TLSRuntimeState, TLSSpec


@dataclass
class ObservationConfig:
    """Configuration for what to include in the global observation."""
    include_queue: bool = True
    include_vehicle_count: bool = True
    include_mean_speed: bool = True
    include_density: bool = True
    include_global: bool = True
    normalize: bool = True
    max_lane_vehicles: float = 50.0
    max_speed: float = 13.9          # ~50 km/h in m/s
    max_waiting_time: float = 3600.0 # 1 hour
    max_total_vehicles: float = 2000.0
    max_phase_elapsed: float = 120.0 # 2 minutes


class ObservationBuilder:
    """
    Extracts the global observation vector from SUMO via TraCI.

    The builder is stateless — it reads fresh values each time
    build() is called and writes them into the shared runtime states.
    """

    def __init__(
        self,
        tls_specs: list[TLSSpec],
        config: ObservationConfig | None = None,
    ) -> None:
        self.specs: list[TLSSpec] = sorted(tls_specs, key=lambda s: s.tls_id)
        self.cfg = config or ObservationConfig()
        self._obs_size: int | None = None

    # ──────────────────────────────────────────────────────────────────────────
    # Observation size computation
    # ──────────────────────────────────────────────────────────────────────────

    def compute_obs_size(self) -> int:
        """Compute the size of the flat observation vector."""
        size = 0
        for spec in self.specs:
            n_lanes = len(spec.controlled_lanes)
            n_phases = len(spec.phases)
            # phase one-hot + elapsed
            size += n_phases + 1
            if self.cfg.include_queue:
                size += n_lanes
            if self.cfg.include_vehicle_count:
                size += n_lanes
            if self.cfg.include_mean_speed:
                size += n_lanes
            if self.cfg.include_density:
                size += n_lanes
        if self.cfg.include_global:
            size += 2  # total_vehicles, total_waiting_time
        self._obs_size = size
        return size

    # ──────────────────────────────────────────────────────────────────────────
    # Main API
    # ──────────────────────────────────────────────────────────────────────────

    def build(self, states: dict[str, TLSRuntimeState]) -> np.ndarray:
        """
        Construct the global observation vector.

        Also updates lane-level fields in TLSRuntimeState objects
        so the reward/metrics modules can reuse the same TraCI reads.
        """
        parts: list[float] = []
        cfg = self.cfg

        lane_cache: dict[str, tuple[float, float, float, float]] = {}

        for spec in self.specs:
            state = states[spec.tls_id]
            n_phases = len(spec.phases)

            # ── Phase one-hot + elapsed ──────────────────────────────────────
            one_hot = [0.0] * n_phases
            if 0 <= state.current_phase < n_phases:
                one_hot[state.current_phase] = 1.0
            elapsed_norm = (
                min(state.phase_elapsed / cfg.max_phase_elapsed, 1.0)
                if cfg.normalize else state.phase_elapsed
            )
            parts.extend(one_hot)
            parts.append(elapsed_norm)

            # ── Lane features ────────────────────────────────────────────────
            queue_vals: list[float] = []
            vcount_vals: list[float] = []
            speed_vals: list[float] = []
            density_vals: list[float] = []

            for lane_id in spec.controlled_lanes:
                if lane_id in lane_cache:
                    halt, vcnt, spd, occ = lane_cache[lane_id]
                else:
                    try:
                        halt = float(traci.lane.getLastStepHaltingNumber(lane_id))
                        vcnt = float(traci.lane.getLastStepVehicleNumber(lane_id))
                        spd  = float(traci.lane.getLastStepMeanSpeed(lane_id))
                        occ  = float(traci.lane.getLastStepOccupancy(lane_id)) / 100.0
                    except traci.exceptions.TraCIException:
                        halt, vcnt, spd, occ = 0.0, 0.0, 0.0, 0.0
                    lane_cache[lane_id] = (halt, vcnt, spd, occ)

                if cfg.normalize:
                    queue_vals.append(min(halt / cfg.max_lane_vehicles, 1.0))
                    vcount_vals.append(min(vcnt / cfg.max_lane_vehicles, 1.0))
                    speed_vals.append(min(max(spd, 0.0) / cfg.max_speed, 1.0))
                    density_vals.append(min(occ, 1.0))
                else:
                    queue_vals.append(halt)
                    vcount_vals.append(vcnt)
                    speed_vals.append(spd)
                    density_vals.append(occ)

            # Write back to shared state for reward/metrics
            state.lane_queue = queue_vals
            state.lane_vehicles = vcount_vals
            state.lane_speed = speed_vals
            state.lane_density = density_vals

            if cfg.include_queue:
                parts.extend(queue_vals)
            if cfg.include_vehicle_count:
                parts.extend(vcount_vals)
            if cfg.include_mean_speed:
                parts.extend(speed_vals)
            if cfg.include_density:
                parts.extend(density_vals)

        # ── Global features ──────────────────────────────────────────────────
        vehicle_ids = []
        waiting_times = []
        total_v = 0.0
        total_w = 0.0
        avg_w = 0.0
        try:
            vehicle_ids = traci.vehicle.getIDList()
            total_v = float(len(vehicle_ids))
            waiting_times = [traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids]
            total_w = float(sum(waiting_times))
            avg_w = float(np.mean(waiting_times)) if waiting_times else 0.0
        except traci.exceptions.TraCIException:
            pass

        self.last_step_cache = {
            "vehicle_ids": vehicle_ids,
            "waiting_times": waiting_times,
            "total_vehicles": total_v,
            "total_waiting": total_w,
            "avg_waiting": avg_w,
        }

        if cfg.include_global:
            if cfg.normalize:
                parts.append(min(total_v / cfg.max_total_vehicles, 1.0))
                parts.append(min(total_w / cfg.max_waiting_time, 1.0))
            else:
                parts.append(total_v)
                parts.append(total_w)

        obs = np.array(parts, dtype=np.float32)
        return obs

    # ──────────────────────────────────────────────────────────────────────────
    # Gymnasium spaces helper
    # ──────────────────────────────────────────────────────────────────────────

    def observation_space(self):
        """Return a gymnasium.spaces.Box for the observation."""
        import gymnasium as gym  # lazy import

        size = self._obs_size or self.compute_obs_size()
        return gym.spaces.Box(
            low=0.0,
            high=1.0 if self.cfg.normalize else np.inf,
            shape=(size,),
            dtype=np.float32,
        )
