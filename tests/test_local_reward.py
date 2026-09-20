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
