"""Unit tests for the per-TLS local reward (parameter-shared IPPO)."""
from __future__ import annotations

from unittest.mock import patch

from src.environment.local_reward import LocalRewardCalculator, LocalRewardConfig
from src.scenario.tls_metadata import PhaseSpec, TLSSpec


def make_spec(tls_id: str, lanes: list[str]) -> TLSSpec:
    return TLSSpec(
        tls_id=tls_id, program_id="0", type="static", junction_id=tls_id,
        phases=[PhaseSpec(index=0, state="G", duration=30.0)], controlled_lanes=lanes,
    )


@patch("src.environment.local_reward.traci")
def test_reward_only_reflects_own_lanes(mock_traci):
    """TLS A's reward must not change when only TLS B's lane waiting time changes."""
    mock_traci.exceptions.TraCIException = Exception
    spec_a = make_spec("A", ["a_lane"])
    spec_b = make_spec("B", ["b_lane"])

    calc = LocalRewardCalculator(LocalRewardConfig(waiting_delta_weight=1.0, switching_penalty=0.0))
    calc.reset()

    mock_traci.vehicle.getIDList.return_value = ["v1", "v2"]
    mock_traci.vehicle.getLaneID.side_effect = lambda vid: {"v1": "a_lane", "v2": "b_lane"}[vid]
    mock_traci.vehicle.getWaitingTime.side_effect = lambda vid: {"v1": 10.0, "v2": 10.0}[vid]
    calc.refresh_lane_waiting()
    calc.compute("A", spec_a, queue_sum=0.0, switched=False)  # baseline for A
    calc.compute("B", spec_b, queue_sum=0.0, switched=False)  # baseline for B

    # Only B's lane gets worse; A's own waiting is unchanged.
    mock_traci.vehicle.getWaitingTime.side_effect = lambda vid: {"v1": 10.0, "v2": 100.0}[vid]
    calc.refresh_lane_waiting()
    reward_a, _ = calc.compute("A", spec_a, queue_sum=0.0, switched=False)
    reward_b, _ = calc.compute("B", spec_b, queue_sum=0.0, switched=False)

    assert reward_a == 0.0
    assert reward_b < 0.0


@patch("src.environment.local_reward.traci")
def test_switching_penalty_applies_only_when_this_agent_switched(mock_traci):
    mock_traci.exceptions.TraCIException = Exception
    spec = make_spec("A", ["a_lane"])
    calc = LocalRewardCalculator(LocalRewardConfig(waiting_delta_weight=0.0, switching_penalty=0.3))
    calc.reset()
    mock_traci.vehicle.getIDList.return_value = []
    calc.refresh_lane_waiting()

    reward_hold, _ = calc.compute("A", spec, queue_sum=0.0, switched=False)
    reward_switch, _ = calc.compute("A", spec, queue_sum=0.0, switched=True)

    assert reward_hold == 0.0
    assert reward_switch == -0.3


@patch("src.environment.local_reward.traci")
def test_spillback_penalty_only_fires_on_downstream_queue_increase(mock_traci):
    """An increasing downstream queue should be penalized; a decreasing or flat one should not
    (only the *increase* in spillback is a signal worth discouraging)."""
    mock_traci.exceptions.TraCIException = Exception
    spec = make_spec("A", ["a_lane"])
    calc = LocalRewardCalculator(LocalRewardConfig(
        waiting_delta_weight=0.0, switching_penalty=0.0, spillback_weight=0.5,
    ))
    calc.reset()
    mock_traci.vehicle.getIDList.return_value = []
    calc.refresh_lane_waiting()

    reward_baseline, _ = calc.compute("A", spec, queue_sum=0.0, switched=False, downstream_queue_sum=2.0)
    assert reward_baseline == 0.0  # first call has no prior value to compare against

    reward_worse, info_worse = calc.compute("A", spec, queue_sum=0.0, switched=False, downstream_queue_sum=6.0)
    assert reward_worse == -0.5 * 4.0  # delta of +4.0 vehicles * spillback_weight
    assert info_worse["reward/spillback_component"] == -2.0

    reward_better, info_better = calc.compute("A", spec, queue_sum=0.0, switched=False, downstream_queue_sum=1.0)
    assert reward_better == 0.0  # a decrease is not penalized (only increases are)
    assert info_better["reward/spillback_component"] == 0.0


@patch("src.environment.local_reward.traci")
def test_pressure_penalty_fires_symmetrically_on_magnitude(mock_traci):
    """The pressure term penalizes |pressure| in either direction — both a
    congested-upstream imbalance (positive pressure) and a spillback-risk
    imbalance (negative pressure) should be penalized, since both are
    departures from a balanced upstream/downstream state. Disabled
    (weight=0.0) must be a complete no-op — the run's current default."""
    mock_traci.exceptions.TraCIException = Exception
    spec = make_spec("A", ["a_lane"])
    calc = LocalRewardCalculator(LocalRewardConfig(
        waiting_delta_weight=0.0, switching_penalty=0.0, spillback_weight=0.0, pressure_weight=0.4,
    ))
    calc.reset()
    mock_traci.vehicle.getIDList.return_value = []
    calc.refresh_lane_waiting()

    reward_pos, info_pos = calc.compute("A", spec, queue_sum=0.0, switched=False, pressure=5.0)
    assert reward_pos == -0.4 * 5.0
    assert info_pos["reward/pressure_component"] == -2.0

    reward_neg, info_neg = calc.compute("A", spec, queue_sum=0.0, switched=False, pressure=-5.0)
    assert reward_neg == -0.4 * 5.0  # magnitude, not sign
    assert info_neg["reward/pressure_component"] == -2.0

    reward_balanced, _ = calc.compute("A", spec, queue_sum=0.0, switched=False, pressure=0.0)
    assert reward_balanced == 0.0


@patch("src.environment.local_reward.traci")
def test_pressure_disabled_by_default_is_a_no_op(mock_traci):
    mock_traci.exceptions.TraCIException = Exception
    spec = make_spec("A", ["a_lane"])
    calc = LocalRewardCalculator(LocalRewardConfig(waiting_delta_weight=0.0, switching_penalty=0.0))
    calc.reset()
    mock_traci.vehicle.getIDList.return_value = []
    calc.refresh_lane_waiting()

    reward, info = calc.compute("A", spec, queue_sum=0.0, switched=False, pressure=999.0)
    assert reward == 0.0
    assert info["reward/pressure_component"] == 0.0


@patch("src.environment.local_reward.traci")
def test_neglected_pressure_penalizes_ignoring_a_congested_other_phase(mock_traci):
    """RL-strategy Phase 2 fix (2026-09-24): unlike pressure_weight (which
    uses aggregate, action-independent pressure), neglected_pressure_weight
    must fire on a positive neglected_pressure value regardless of the
    other components — this is what's supposed to make ignoring a growing
    queue on a phase the agent isn't currently serving costly."""
    mock_traci.exceptions.TraCIException = Exception
    spec = make_spec("A", ["a_lane"])
    calc = LocalRewardCalculator(LocalRewardConfig(
        waiting_delta_weight=0.0, switching_penalty=0.0, spillback_weight=0.0,
        pressure_weight=0.0, neglected_pressure_weight=1.0,
    ))
    calc.reset()
    mock_traci.vehicle.getIDList.return_value = []
    calc.refresh_lane_waiting()

    reward, info = calc.compute("A", spec, queue_sum=0.0, switched=False, neglected_pressure=9.0)
    assert reward == -9.0
    assert info["reward/neglected_pressure_component"] == -9.0


@patch("src.environment.local_reward.traci")
def test_neglected_pressure_does_not_penalize_negative_values(mock_traci):
    """A negative neglected_pressure means every other phase is
    under-saturated (spillback risk elsewhere, not backlog) — not something
    to punish the same way as a genuinely congested neglected phase."""
    mock_traci.exceptions.TraCIException = Exception
    spec = make_spec("A", ["a_lane"])
    calc = LocalRewardCalculator(LocalRewardConfig(
        waiting_delta_weight=0.0, switching_penalty=0.0, spillback_weight=0.0,
        pressure_weight=0.0, neglected_pressure_weight=1.0,
    ))
    calc.reset()
    mock_traci.vehicle.getIDList.return_value = []
    calc.refresh_lane_waiting()

    reward, info = calc.compute("A", spec, queue_sum=0.0, switched=False, neglected_pressure=-9.0)
    assert reward == 0.0
    assert info["reward/neglected_pressure_component"] == 0.0


@patch("src.environment.local_reward.traci")
def test_neglected_pressure_disabled_by_default_is_a_no_op(mock_traci):
    mock_traci.exceptions.TraCIException = Exception
    spec = make_spec("A", ["a_lane"])
    calc = LocalRewardCalculator(LocalRewardConfig(waiting_delta_weight=0.0, switching_penalty=0.0))
    calc.reset()
    mock_traci.vehicle.getIDList.return_value = []
    calc.refresh_lane_waiting()

    reward, info = calc.compute("A", spec, queue_sum=0.0, switched=False, neglected_pressure=999.0)
    assert reward == 0.0
    assert info["reward/neglected_pressure_component"] == 0.0


@patch("src.environment.local_reward.traci")
def test_reward_clip_bounds_both_directions(mock_traci):
    import traci.exceptions as texc

    mock_traci.exceptions.TraCIException = texc.TraCIException
    spec = make_spec("A", ["a_lane"])
    calc = LocalRewardCalculator(LocalRewardConfig(
        waiting_delta_weight=0.0, switching_penalty=0.0, spillback_weight=0.0,
        neglected_pressure_weight=1.0, reward_clip=10.0))
    calc.reset()
    mock_traci.vehicle.getIDList.return_value = []
    calc.refresh_lane_waiting()
    reward, _ = calc.compute("A", spec, queue_sum=0.0, switched=False, neglected_pressure=500.0)
    assert reward == -10.0
    unclipped = LocalRewardCalculator(LocalRewardConfig(
        waiting_delta_weight=0.0, switching_penalty=0.0, spillback_weight=0.0,
        neglected_pressure_weight=1.0, reward_clip=None))
    unclipped.reset()
    unclipped.refresh_lane_waiting()
    assert unclipped.compute("A", spec, 0.0, False, neglected_pressure=500.0)[0] == -500.0


@patch("src.environment.local_reward.traci")
def test_refresh_lane_waiting_sums_every_vehicle_on_a_lane(mock_traci):
    import traci.exceptions as texc

    mock_traci.exceptions.TraCIException = texc.TraCIException
    mock_traci.vehicle.getIDList.return_value = ["v1", "v2", "v3"]
    mock_traci.vehicle.getLaneID.side_effect = lambda v: {"v1": "L", "v2": "L", "v3": "M"}[v]
    mock_traci.vehicle.getWaitingTime.side_effect = lambda v: {"v1": 4.0, "v2": 6.0, "v3": 1.0}[v]
    calc = LocalRewardCalculator()
    calc.refresh_lane_waiting()
    assert calc._lane_waiting == {"L": 10.0, "M": 1.0}      # not just the last vehicle per lane
    calc.reset()
    spec = make_spec("A", ["L", "M"])
    calc.refresh_lane_waiting()
    calc.compute("A", spec, 0.0, False)                       # baseline
    mock_traci.vehicle.getWaitingTime.side_effect = lambda v: {"v1": 14.0, "v2": 6.0, "v3": 1.0}[v]
    calc.refresh_lane_waiting()
    reward, info = calc.compute("A", spec, 0.0, False)
    assert info["metrics/local_waiting"] == 21.0
    assert reward == pytest.approx(-10.0 * 1.0)               # -(21 - 11) * waiting_delta_weight


@patch("src.environment.local_reward.traci")
def test_queue_weight_term_uses_queue_sum(mock_traci):
    import traci.exceptions as texc

    mock_traci.exceptions.TraCIException = texc.TraCIException
    mock_traci.vehicle.getIDList.return_value = []
    spec = make_spec("A", ["a"])
    calc = LocalRewardCalculator(LocalRewardConfig(
        waiting_delta_weight=0.0, queue_weight=0.5, switching_penalty=0.0, spillback_weight=0.0))
    calc.refresh_lane_waiting()
    reward, info = calc.compute("A", spec, queue_sum=8.0, switched=False)
    assert reward == -4.0 and info["reward/queue_component"] == -4.0


import pytest  # noqa: E402
