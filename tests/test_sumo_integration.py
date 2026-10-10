"""Real-SUMO integration tests (auto-skipped when SUMO / the InTAS scenario is absent).

Short windows only (<= 100 s of simulated time each); run one at a time.
"""
from __future__ import annotations

import psutil
import pytest

from src.environment.sumo_args import build_sumo_command, command_signature
from src.environment.sumo_connection import traci
from tests.conftest import SCENARIO_CFG, TLS_METADATA

pytestmark = [pytest.mark.sumo, pytest.mark.slow]

CFG = str(SCENARIO_CFG)
META = str(TLS_METADATA)
BEGIN = 25200
EXTRA = ["--begin", str(BEGIN), "--end", str(BEGIN + 100)]


def _sumo_children() -> list[int]:
    out = []
    for p in psutil.Process().children(recursive=True):
        try:
            if "sumo" in p.name().lower():
                out.append(p.pid)
        except psutil.NoSuchProcess:
            pass
    return out


def _hold(env):
    acts = {}
    for tid, st in env._runtime_states.items():
        gi = env.specs_by_id[tid].green_phase_indices
        acts[tid] = gi.index(st.current_phase) if st.current_phase in gi else 0
    return acts


def test_decentralized_env_smoke_reset_20_steps_close_without_orphans():
    from src.environment.decentralized_sumo_env import DecentralizedSumoEnv
    from src.environment.local_observation import LOCAL_OBS_SIZE

    assert _sumo_children() == []
    env = DecentralizedSumoEnv(CFG, META, episode_duration=BEGIN + 100, sumo_extra_args=EXTRA)
    try:
        obs, infos = env.reset()
        assert len(obs) == 98 and all(o.shape == (LOCAL_OBS_SIZE,) for o in obs.values())
        assert all(infos[t]["action_mask"].shape == (6,) for t in infos)
        for _ in range(20):
            obs, rew, te, tr, inf = env.step(_hold(env))
        assert env.agents == []                                  # 20 x 5 s == the 100 s window
        assert all(tr.values()) and not any(te.values())         # time limit => truncated only
        assert all(inf[t]["TimeLimit.truncated"] for t in inf)
        assert all(0.0 <= v <= 1.0 for o in obs.values() for v in o)
        assert all(abs(r) < 1e4 for r in rew.values())
        summary = env.metrics.summarize()
        assert summary.episode_steps == 20 and summary.metrics_schema_version == 2
        assert summary.sumo_command and "--begin" in summary.sumo_command
    finally:
        env.close()
    assert _sumo_children() == [], "orphan sumo process after env.close()"


def test_hold_leaves_the_displayed_state_equal_to_the_believed_phase():
    """tests-reliability-005: with the native program frozen at reset, 'hold' keeps
    SUMO's displayed state == the safety layer's believed state for every TLS."""
    from src.environment.decentralized_sumo_env import DecentralizedSumoEnv

    env = DecentralizedSumoEnv(CFG, META, episode_duration=BEGIN + 100, sumo_extra_args=EXTRA)
    try:
        env.reset()
        for _ in range(8):                                       # 40 s, before any max-green force-advance
            env.step(_hold(env))
            for tid in env._runtime_states:
                assert traci.trafficlight.getRedYellowGreenState(tid) == env.safety.displayed_state(tid), tid
    finally:
        env.close()


def test_env_and_native_baseline_start_in_identical_conditions():
    """rl-core-003/004: same command => same initial signal state on all 98 TLS and the
    same traffic for the first 25 s (until the frozen RL signals differ from the native
    actuated program)."""
    from src.environment.decentralized_sumo_env import DecentralizedSumoEnv

    cmd = build_sumo_command("sumo", CFG, seed=42, extra_args=EXTRA)
    traci.start(cmd, label="native_parity")
    try:
        native_init = {t: traci.trafficlight.getRedYellowGreenState(t) for t in traci.trafficlight.getIDList()}
        native_series = []
        for _ in range(25):
            traci.simulationStep()
            native_series.append((traci.simulation.getDepartedNumber(), len(traci.vehicle.getIDList())))
    finally:
        traci.close()

    env = DecentralizedSumoEnv(CFG, META, seed=42, episode_duration=BEGIN + 100, sumo_extra_args=EXTRA)
    try:
        env.reset()
        assert command_signature(env.last_sumo_command) == command_signature(cmd)
        assert {t: env.safety.displayed_state(t) for t in env._runtime_states} == native_init
        env_series = []
        for _ in range(5):
            env.step(_hold(env))
            rec = env.metrics._records[-1]
            env_series.append((rec.departed_vehicles, rec.total_vehicles))
    finally:
        env.close()
    native_5 = [(sum(d for d, _ in native_series[i * 5:(i + 1) * 5]), native_series[i * 5 + 4][1]) for i in range(5)]
    assert env_series == native_5


def test_baselines_and_unified_metrics_smoke():
    from src.controllers.fixed_time import FixedTimeController
    from src.controllers.max_pressure import MaxPressureController
    from src.controllers.sumo_actuated import SumoActuatedController

    results = {}
    for ctrl in (SumoActuatedController, FixedTimeController, MaxPressureController):
        c = ctrl(episode_duration=BEGIN + 100, sumo_extra_args=EXTRA)
        results[c.name] = c.run_episode(CFG, META, seed=42)
        assert _sumo_children() == [], f"{c.name} left a sumo process behind"
    for name, r in results.items():
        assert r.total_steps == 20 and r.episode_duration == 95.0, name       # same sampling grid
        assert r.metrics_schema_version == 2 and r.sumo_command
        assert 0.0 <= r.mean_speed <= 20.0 and r.max_queue >= 0.0
    assert results["sumo_actuated"].total_switches > 0 and results["fixed_time"].total_switches > 0
    assert any("static_programs.add.xml" in a for a in results["fixed_time"].sumo_command)
    assert command_signature(results["sumo_actuated"].sumo_command) == command_signature(results["max_pressure"].sumo_command)
