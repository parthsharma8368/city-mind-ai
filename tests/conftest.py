"""Shared pytest configuration.

* Registers the ``sumo`` and ``slow`` markers (pyproject.toml is not ours to edit).
* Skips every ``@pytest.mark.sumo`` test automatically when no SUMO binary is
  available, so the mocked unit tests still run (and import) on a machine
  without SUMO -- src.environment.sumo_connection is import-safe.
* Offers a ``FakeTraci`` factory fixture: a scriptable in-memory stand-in for the
  traci API used by the environments/controllers (see tests/fake_traci.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCENARIO_CFG = ROOT / "data" / "raw" / "intas" / "scenario" / "InTAS_train.sumocfg"
TLS_METADATA = ROOT / "artifacts" / "scenario" / "tls_metadata.json"


def pytest_configure(config):
    config.addinivalue_line("markers", "sumo: needs a real SUMO binary and the InTAS scenario (auto-skipped otherwise)")
    config.addinivalue_line("markers", "slow: slower than ~10 s")


def _sumo_unavailable_reason() -> str | None:
    from src.environment.sumo_connection import sumo_available

    if not sumo_available():
        return "SUMO binary not found (set SUMO_HOME or put `sumo` on PATH)"
    if not SCENARIO_CFG.exists():
        return f"InTAS scenario not present: {SCENARIO_CFG}"
    if not TLS_METADATA.exists():
        return f"TLS metadata not present: {TLS_METADATA}"
    return None


def pytest_collection_modifyitems(config, items):
    reason = None
    computed = False
    for item in items:
        if "sumo" in item.keywords:
            if not computed:
                reason, computed = _sumo_unavailable_reason(), True
            if reason:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session")
def real_tls_specs():
    """All 98 real InTAS TLSSpecs (skips if the metadata artifact is missing)."""
    if not TLS_METADATA.exists():
        pytest.skip("artifacts/scenario/tls_metadata.json not present")
    from src.scenario.tls_metadata import load_tls_specs

    return load_tls_specs(TLS_METADATA)
