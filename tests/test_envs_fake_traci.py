"""Environment reset/step loops against an in-memory fake traci (no SUMO needed).

Covers: command recording, seed modes (rl-core-005), takeover at reset
(rl-core-003), time-limit => truncated only + the SuperSuit/SB3 chain
(rl-core-006), displayed-state trajectories (rl-core-008), and SUMO is closed
on failure paths (rl-core-011).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.environment.decentralized_sumo_env import DecentralizedSumoEnv
from src.environment.local_observation import OBS_SIZE
from src.environment.sumo_args import get_option
from src.environment.sumo_session import episode_end_flags, mark_time_limit
from tests.fake_traci import FakeTraci, install

STATES = ["GGrr", "yyrr", "rrGG", "rryy"]


@pytest.fixture
def world(tmp_path, monkeypatch):
    fake = FakeTraci({"T0": list(STATES), "T1": list(STATES)}, end_time=1e9, begin=100.0)
    install(monkeypatch, fake)
    entries = [{
        "tls_id": tid, "program_id": "0", "type": "actuated",
        "phases": [{"index": i, "state": s, "duration": 10.0 if "G" in s else 3.0,
                    "min_dur": 3.0, "max_dur": 40.0 if "G" in s else 3.0}
                   for i, s in enumerate(STATES)],
        "controlled_lanes": ["l0", "l1", "l0", "l1"], "signal_lanes": [["l0"], ["l0"], ["l1"], ["l1"]],
    } for tid in ("T0", "T1")]
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps(entries))
    cfg = tmp_path / "s.sumocfg"
    cfg.write_text("<configuration/>")
    return fake, str(cfg), str(meta)


def _dec(world, **kw):
    fake, cfg, meta = world
    kw.setdefault("episode_duration", 120.0)
    kw.setdefault("sumo_extra_args", ["--begin", "100", "--end", "120"])
    return DecentralizedSumoEnv(cfg, meta, **kw)


def _hold(env):
    return {tid: (env.specs_by_id[tid].green_phase_indices.index(s.current_phase)
                  if s.current_phase in env.specs_by_id[tid].green_phase_indices else 0)
            for tid, s in env._runtime_states.items()}


def test_reset_shapes_command_and_takeover(world):
    fake, cfg, meta = world
    env = _dec(world, seed=11)
    obs, infos = env.reset()
    assert set(obs) == {"T0", "T1"} and all(o.shape == (OBS_SIZE,) for o in obs.values())
    cmd = fake.started[0]
    assert get_option(cmd, "--seed") == "11" and get_option(cmd, "--begin") == "100"
    assert infos["T0"]["sumo_command"] == cmd == env.last_sumo_command
    # takeover: believed state written at reset, displayed == believed
    assert {(t, s) for _, t, s in fake.written} == {("T0", "GGrr"), ("T1", "GGrr")}
    for tid in ("T0", "T1"):
        assert fake.trafficlight.getRedYellowGreenState(tid) == env.safety.displayed_state(tid)
    env.close()
    assert fake.closed == 1


def test_hold_keeps_displayed_equal_to_believed_every_step(world):
    fake, *_ = world
    env = _dec(world, seed=1, episode_duration=10_000.0)
    env.reset()
    for _ in range(10):
        env.step(_hold(env))
        for tid in ("T0", "T1"):
            assert fake.trafficlight.getRedYellowGreenState(tid) == env.safety.displayed_state(tid)
    env.close()


def test_without_takeover_native_program_runs_on_and_diverges(world):
    """Documents the pre-fix behaviour (freeze_native_program=False) the takeover removes."""
    fake, *_ = world
    env = _dec(world, seed=1, episode_duration=10_000.0, freeze_native_program=False)
    env.reset()
    for _ in range(3):
        env.step(_hold(env))
    assert any(fake.trafficlight.getRedYellowGreenState(t) != env.safety.displayed_state(t)
               for t in ("T0", "T1"))
    env.close()


def test_seed_modes(world):
    fake, *_ = world
    fixed = _dec(world, seed=42)
    for _ in range(3):
        fixed.reset()
    assert [get_option(c, "--seed") for c in fake.started] == ["42", "42", "42"]
    fake.started.clear()
    per = _dec(world, seed=42, seed_mode="per_episode")
    for _ in range(3):
        per.reset()
    assert [get_option(c, "--seed") for c in fake.started] == ["42", "43", "44"]
    per.reset(seed=1000)                                   # an explicit seed re-bases the sequence
    per.reset()
    assert [get_option(c, "--seed") for c in fake.started[-2:]] == ["1000", "1001"]
    with pytest.raises(ValueError, match="seed_mode"):
        _dec(world, seed_mode="bogus")


def test_warm_start_runs_native_signals_before_the_episode(world):
    fake, *_ = world
    env = _dec(world, seed=1, warm_start_seconds=30.0)
    env.reset()
    assert get_option(fake.started[0], "--begin") == "70"       # 100 - 30
    assert fake.steps == 30 and fake.time == 100.0
    # takeover happens AFTER the warm start, at t = begin
    assert all(t == 100.0 for t, *_ in fake.written)


def test_time_limit_is_truncated_not_terminated_and_flagged_for_bootstrapping(world):
    fake, *_ = world
    env = _dec(world, seed=1, episode_duration=120.0)
    env.reset()
    flags = []
    for _ in range(4):                                         # 100 -> 120
        _, _, te, tr, inf = env.step(_hold(env))
        flags.append((te["T0"], tr["T0"], inf["T0"]["TimeLimit.truncated"]))
    assert flags[:3] == [(False, False, False)] * 3
    assert flags[3] == (False, True, True)
    assert env.agents == []


def test_natural_end_is_terminated_and_not_a_timelimit(world):
    fake, *_ = world
    fake.end_time = 110.0                                      # nothing left to simulate at t=110
    env = _dec(world, seed=1, episode_duration=10_000.0)
    env.reset()
    env.step(_hold(env))
    _, _, te, tr, inf = env.step(_hold(env))
    assert (te["T0"], tr["T0"], inf["T0"]["TimeLimit.truncated"]) == (True, False, False)


def test_end_flag_helper_truth_table():
    assert episode_end_flags(5, 10, 3) == (False, False)
    assert episode_end_flags(5, 10, 0) == (True, False)
    assert episode_end_flags(10, 10, 3) == (False, True)
    assert episode_end_flags(10, 10, 0) == (False, True)      # time limit wins: still a truncation
    assert mark_time_limit({}, False, True)["TimeLimit.truncated"] is True
    assert mark_time_limit({}, True, False)["TimeLimit.truncated"] is False


def test_dead_sumo_is_not_reported_as_a_normal_episode_end(world):
    from tests.fake_traci import FatalTraCIError, TraCIException

    fake, *_ = world
    env = _dec(world, seed=1, episode_duration=10_000.0)
    env.reset()
    fake.fail_on["getMinExpectedNumber"] = TraCIException("odd")
    assert env._episode_end_flags(105.0) == (True, False)       # expected error: logged, ends the episode
    fake.fail_on["getMinExpectedNumber"] = FatalTraCIError("closed")
    with pytest.raises(FatalTraCIError):
        env._episode_end_flags(105.0)


def test_sumo_is_closed_when_reset_fails_after_start(world):
    fake, *_ = world
    fake.fail_on["setRedYellowGreenState"] = RuntimeError("takeover failed")
    env = _dec(world, seed=1)
    with pytest.raises(RuntimeError):
        env.reset()
    assert fake.closed == 1 and env._sim_running is False


def test_supersuit_sb3_chain_passes_timelimit_truncated_to_the_bootstrap_hook(world):
    """SB3's on-policy loop bootstraps only when done AND infos[i]['TimeLimit.truncated']
    AND infos[i]['terminal_observation']. Verify the real SuperSuit wrapper chain delivers both."""
    import supersuit as ss

    env = _dec(world, seed=1, episode_duration=120.0)
    vec = ss.pettingzoo_env_to_vec_env_v1(env)
    vec = ss.concat_vec_envs_v1(vec, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
    vec.reset()
    for k in range(4):
        obs, rew, dones, infos = vec.step(np.zeros(2, dtype=np.int64))
    assert dones.tolist() == [True, True]
    for inf in infos:
        assert inf["TimeLimit.truncated"] is True
        assert inf["terminal_observation"].shape == (OBS_SIZE,)
    vec.close()


def test_action_and_observation_spaces_and_unique_lane_bookkeeping(world):
    """Discrete(2) keep / advance; a 22-number signed observation; lanes that serve several links are counted once."""
    env = _dec(world)
    assert env.action_space("T0").n == 2
    box = env.observation_space("T0")
    assert box.shape == (OBS_SIZE,) and box.low.min() == -1.0 and box.high.max() == 1.0
    assert env._unique_lane_idx["T0"] == [0, 1]            # controlled_lanes = [l0, l1, l0, l1]
    assert env._edge_to_tls == {"l0": "T0", "l1": "T0"}    # topology is derived from the metadata, not written by hand
    obs, _ = env.reset()
    assert obs["T0"][0] == 1.0 and obs["T0"][6] == 0.0     # slot 0 active, not in transition
    obs, rew, *_ = env.step({t: 1 for t in env.agents})    # advance
    assert all(np.isfinite(o).all() for o in obs.values()) and set(rew) == {"T0", "T1"}
    assert all(r <= 0.0 for r in rew.values())             # every reward term is a cost
    env.close()


def test_road_neighbours_follow_the_road_graph_not_straight_line_distance(tmp_path, monkeypatch):
    """T0 -> (unsignalised eB, 100 m) -> T1 are neighbours; T0 -> 700 m of unsignalised road -> T2 are not."""
    import json

    from src.environment.decentralized_sumo_env import DecentralizedSumoEnv
    from src.environment.local_observation import LocalObservationConfig
    from tests.fake_traci import FakeTraci, install

    states = ["GGrr", "yyrr", "rrGG", "rryy"]
    fake = FakeTraci({t: list(states) for t in ("T0", "T1", "T2")}, end_time=1e9, begin=100.0)
    install(monkeypatch, fake)
    lane_of = {"T0": "eA_0", "T1": "eC_0", "T2": "eE_0"}
    meta = [{
        "tls_id": t, "program_id": "0", "type": "actuated",
        "phases": [{"index": i, "state": s, "duration": 10.0 if "G" in s else 3.0, "min_dur": 3.0,
                    "max_dur": 40.0 if "G" in s else 3.0} for i, s in enumerate(states)],
        "controlled_lanes": [ln] * 4, "signal_lanes": [[ln]] * 4,
    } for t, ln in lane_of.items()]
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "s.sumocfg").write_text("<configuration/>")
    fake.halting.clear()
    fake.halting.update({"eA_0": 3, "eB_0": 0, "eC_0": 8, "eG_0": 0, "eE_0": 0})
    links = {"eA_0": ["eB_0", "eG_0"], "eB_0": ["eC_0"], "eG_0": ["eE_0"]}
    lengths = {"eB_0": 100.0, "eG_0": 700.0}
    monkeypatch.setattr(fake.lane, "getLinks", staticmethod(lambda lid: [(x, True, True, False, "M", "s", 1.0)
                                                                           for x in links.get(lid, [])]), raising=False)
    monkeypatch.setattr(fake.lane, "getLength", staticmethod(lambda lid: lengths.get(lid, 100.0)), raising=False)

    env = DecentralizedSumoEnv(str(tmp_path / "s.sumocfg"), str(tmp_path / "meta.json"),
                               obs_config=LocalObservationConfig(), episode_duration=120.0,
                               sumo_extra_args=["--begin", "100", "--end", "120"])
    assert env.observation_space("T0").shape == (OBS_SIZE,)
    obs, _ = env.reset()
    assert env._road_down["T0"] == ["T1"] and env._road_up["T1"] == ["T0"]
    assert env._road_down["T1"] == [] and env._road_up["T2"] == [] and env._road_down["T2"] == []
    assert obs["T0"][20] == pytest.approx(8 / 16)           # T0 sees T1's queue (8 cars) downstream
    assert obs["T1"][18] == pytest.approx(3 / 11)           # T1 sees T0's queue (3 cars) upstream
    assert obs["T2"][18:].tolist() == [0.0, 0.0, 0.0, 0.0]  # unreachable within 500 m -> no neighbours
    env.close()




def test_env_queue_reach_extends_lane_queue_and_reward_level_without_double_counting(tmp_path, monkeypatch):
    import json

    from src.environment.decentralized_sumo_env import DecentralizedSumoEnv
    from src.environment.local_observation import LocalObservationConfig
    from tests.fake_traci import FakeTraci, install

    states = ["GGrr", "yyrr", "rrGG", "rryy"]
    fake = FakeTraci({"T0": list(states)}, end_time=1e9, begin=100.0)
    install(monkeypatch, fake)
    meta = [{"tls_id": "T0", "program_id": "0", "type": "actuated",
             "phases": [{"index": i, "state": st, "duration": 10.0 if "G" in st else 3.0, "min_dur": 3.0,
                         "max_dur": 40.0 if "G" in st else 3.0} for i, st in enumerate(states)],
             "controlled_lanes": ["eX_0", "eX_0", "eX_1", "eX_1"], "signal_lanes": [["eX_0"], ["eX_0"], ["eX_1"], ["eX_1"]]}]
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "s.sumocfg").write_text("<configuration/>")
    fake.halting.clear()
    fake.halting.update({"eX_0": 2, "eX_1": 1, "eF_0": 10, "eF_1": 5})
    links = {"eF_0": ["eX_0", "eX_1"], "eF_1": ["eX_1"]}      # eF_0 feeds BOTH stop-line lanes
    monkeypatch.setattr(fake.lane, "getLinks", staticmethod(lambda lid: [(x, True, True, False, "M", "s", 1.0)
                                                                           for x in links.get(lid, [])]), raising=False)
    monkeypatch.setattr(fake.lane, "getLength", staticmethod(lambda lid: 100.0), raising=False)
    env = DecentralizedSumoEnv(str(tmp_path / "s.sumocfg"), str(tmp_path / "meta.json"),
                               obs_config=LocalObservationConfig(queue_reach_m=300.0), episode_duration=120.0,
                               sumo_extra_args=["--begin", "100", "--end", "120"])
    env.reset()
    st = env._runtime_states["T0"]
    assert st.lane_queue == [2 + 10, 2 + 10, 1 + 10 + 5, 1 + 10 + 5]      # each stop-line lane + its feeders
    assert env._queue_level["T0"] == (2 + 1) + (10 + 5)                    # own unique lanes + UNION of feeders
    env.close()
