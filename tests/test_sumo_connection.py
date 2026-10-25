"""sumo_connection must be import-safe without SUMO (tests-reliability-008)."""
from __future__ import annotations

import importlib
import os
import subprocess
import sys

import pytest

import src.environment.sumo_connection as sc


def test_import_works_with_no_sumo_anywhere_and_fails_clearly_on_first_use():
    code = (
        "import os, sys, shutil\n"
        "os.environ.pop('SUMO_HOME', None)\n"
        "import src.environment.sumo_connection as sc\n"
        "sc.shutil.which = lambda *_a, **_k: None\n"
        "sc._MAC_DEFAULT_SUMO_HOME = '/nonexistent'\n"
        "sc._LazyTraci._module = None\n"
        "os.environ.pop('SUMO_HOME', None)\n"
        "import src.environment.safety, src.environment.metrics, src.environment.decentralized_sumo_env\n"
        "try:\n"
        "    sc.traci.trafficlight\n"
        "except sc.SumoNotAvailableError as e:\n"
        "    assert 'SUMO_HOME' in str(e); print('OK-lazy')\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "SUMO_HOME"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         cwd=os.path.dirname(os.path.dirname(__file__)))
    assert "OK-lazy" in out.stdout, out.stdout + out.stderr


def test_sumo_available_reflects_binary_presence(monkeypatch):
    monkeypatch.setattr(sc.shutil, "which", lambda *_a, **_k: None)
    monkeypatch.setattr(sc, "resolve_sumo_home", lambda: None)
    assert sc.sumo_available() is False
    monkeypatch.setattr(sc.shutil, "which", lambda *_a, **_k: "/usr/bin/sumo")
    assert sc.sumo_available() is True


@pytest.mark.sumo
def test_lazy_proxy_resolves_to_the_real_backend():
    importlib.reload(sc)
    assert sc.traci.trafficlight is not None
    assert hasattr(sc.traci, "start")
