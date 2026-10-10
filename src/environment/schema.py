"""
schema.py
=========
Single place for the version numbers that make saved artifacts auditable.

METRICS_SCHEMA_VERSION
    Version of the metric DEFINITIONS (not the JSON layout). Result JSONs
    written before this field existed are implicitly version 1 and are NOT
    numerically comparable with version >= 2 for mean_speed, mean_queue,
    max_queue and backlog, nor for total_switches of the SUMO-native
    baselines (they were hard-coded to 0).

    v2 (2026-10): ONE definition for every controller and env
        mean_speed  = mean per-vehicle getSpeed over all vehicles in the net
        mean_queue  = mean getLastStepHaltingNumber over UNIQUE tls-controlled lanes
        max_queue   = max of the same
        backlog     = vehicles in network + pending (insertion-blocked) vehicles
        total_switches = logical green-to-green switches for ALL controllers
        sample grid = after every `decision_interval`-th simulationStep, labelled
                      with post-step sim time (identical for baselines and RL)

OBS_SCHEMA_VERSION
    Version of the observation/feature semantics a model was trained on.
    Loaders refuse a model whose recorded version differs from the code's.
    v2: per-phase pressure uses the correct signal->lane mapping
        (TLSSpec.signal_lanes) and the env freezes SUMO's native program at
        reset, so phase_elapsed / phase one-hot describe the displayed state.
"""
from __future__ import annotations

METRICS_SCHEMA_VERSION = 2
OBS_SCHEMA_VERSION = 2
