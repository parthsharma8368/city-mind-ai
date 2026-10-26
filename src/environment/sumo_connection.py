"""
sumo_connection.py
==================
Unified SUMO TraCI / libsumo interface for CityMind AI.

Import `traci` from this module across all controllers, environment modules,
and evaluation scripts so the entire codebase uses the active backend:
    from src.environment.sumo_connection import traci, USING_LIBSUMO

IMPORT-SAFE WITHOUT SUMO (tests-reliability-008): importing this module never
touches SUMO. ``traci`` is a lazy proxy that loads the real backend (libsumo
if installed, else the socket ``traci``) on first attribute access and raises
``SumoNotAvailableError`` with install guidance if SUMO cannot be found.
Unit tests that fully mock ``traci`` (``patch("src.environment.x.traci")``)
therefore import and run on machines without SUMO, and tests that really need
SUMO are skipped through the ``sumo`` marker (see tests/conftest.py).
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import sys

_MAC_DEFAULT_SUMO_HOME = (
    "/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo"
)


class SumoNotAvailableError(EnvironmentError):
    """Raised on first real use of ``traci`` when SUMO cannot be located."""


def resolve_sumo_home() -> str | None:
    """Return a usable SUMO_HOME (and export it), or None if SUMO is absent."""
    home = os.environ.get("SUMO_HOME")
    if home and os.path.isdir(home):
        return home
    if os.path.exists(_MAC_DEFAULT_SUMO_HOME):
        os.environ["SUMO_HOME"] = _MAC_DEFAULT_SUMO_HOME
        return _MAC_DEFAULT_SUMO_HOME
    binary = shutil.which("sumo")
    if binary:  # <prefix>/bin/sumo -> <prefix>/share/sumo
        candidate = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(binary))), "share", "sumo")
        if os.path.isdir(candidate):
            os.environ["SUMO_HOME"] = candidate
            return candidate
    return home or None


def sumo_available() -> bool:
    """True if a ``sumo`` binary can be found (PATH or SUMO_HOME/bin)."""
    if shutil.which("sumo"):
        return True
    home = resolve_sumo_home()
    return bool(home and os.path.exists(os.path.join(home, "bin", "sumo")))


def _prepare_tools_path() -> str | None:
    home = resolve_sumo_home()
    if home:
        tools_dir = os.path.join(home, "tools")
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
    return home


_prepare_tools_path()
# Decided at import time WITHOUT importing libsumo (importing it may need SUMO).
# CITYMIND_USE_TRACI=1 forces the socket ``traci`` binding: libsumo runs SUMO inside the Python process and cannot drive
# the sumo-gui window, so the live-GUI demo (scripts/run_visual_demo.py) sets it.
USING_LIBSUMO: bool = importlib.util.find_spec("libsumo") is not None and os.environ.get("CITYMIND_USE_TRACI") != "1"


class _LazyTraci:
    """Module-like proxy: resolves the real traci/libsumo on first use."""

    _module = None

    def _load(self):
        if _LazyTraci._module is None:
            if resolve_sumo_home() is None and not shutil.which("sumo"):
                raise SumoNotAvailableError(
                    "SUMO is not available: SUMO_HOME is not set and no `sumo` binary was found.\n"
                    "Install SUMO and export SUMO_HOME to its share/sumo directory."
                )
            _prepare_tools_path()
            try:
                if USING_LIBSUMO:
                    _LazyTraci._module = importlib.import_module("libsumo")
                else:
                    _LazyTraci._module = importlib.import_module("traci")
            except ImportError as exc:  # pragma: no cover - environment specific
                raise SumoNotAvailableError(
                    f"Could not import the SUMO python bindings ({exc}). "
                    "Install `traci`/`libsumo` or fix SUMO_HOME."
                ) from exc
        return _LazyTraci._module

    def __getattr__(self, name: str):
        return getattr(self._load(), name)

    def __repr__(self) -> str:
        return "<lazy traci proxy>" + ("" if _LazyTraci._module is None else f" -> {_LazyTraci._module.__name__}")


traci = _LazyTraci()

try:  # optional nicety, never fatal
    import pandas as pd
    pd.set_option("io.parquet.engine", "pyarrow")
except Exception:
    pass

__all__ = [
    "traci", "USING_LIBSUMO", "SumoNotAvailableError", "sumo_available", "resolve_sumo_home",
]
