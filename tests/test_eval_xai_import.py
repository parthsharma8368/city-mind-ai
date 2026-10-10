"""xai.py failed at import when the local observation grew 12 -> 18 dims (module-level assert)."""
from __future__ import annotations

from src.environment.local_observation import LOCAL_OBS_SIZE
from src.scenario.tls_metadata import MAX_GREEN_PHASES


def test_xai_imports_and_feature_specs_match_observation_size():
    from src.evaluation import xai

    assert len(xai.LOCAL_FEATURE_SPECS) == LOCAL_OBS_SIZE
    names = [n for n, _ in xai.LOCAL_FEATURE_SPECS]
    assert names[12:] == [f"phase_pressure_norm_{i}" for i in range(MAX_GREEN_PHASES)]   # appended last, in order
    assert names[0] == "phase_elapsed_norm" and len(set(names)) == len(names)
    descriptors = xai.build_local_feature_descriptors("T1", 0)
    assert len(descriptors) == LOCAL_OBS_SIZE
