"""
sumo_connection.py
==================
Unified SUMO TraCI / libsumo interface for CityMind AI.

Import `traci` from this module across all controllers, environment modules,
and evaluation scripts so the entire codebase uses the active backend:
    from src.environment.sumo_connection import traci, USING_LIBSUMO
"""
from __future__ import annotations

import os
import sys

SUMO_HOME = os.environ.get("SUMO_HOME")
if SUMO_HOME is None:
    # Default macOS path for EclipseSUMO framework
    default_mac_path = "/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo"
    if os.path.exists(default_mac_path):
        SUMO_HOME = default_mac_path
        os.environ["SUMO_HOME"] = default_mac_path
    else:
        raise EnvironmentError(
            "SUMO_HOME is not set.\n"
            "Please export SUMO_HOME to your SUMO installation share/sumo directory."
        )

tools_dir = os.path.join(SUMO_HOME, "tools")
if tools_dir not in sys.path:
    sys.path.insert(0, tools_dir)

try:
    import libsumo as traci
    USING_LIBSUMO = True
except ImportError:
    import traci
    USING_LIBSUMO = False

try:
    import pandas as pd
    pd.set_option("io.parquet.engine", "pyarrow")
except Exception:
    pass

__all__ = ["traci", "USING_LIBSUMO"]
