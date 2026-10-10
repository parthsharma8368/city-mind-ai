"""
test_reward.py
==============
Unit tests for RewardCalculator.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch, MagicMock
from src.scenario.tls_metadata import TLSRuntimeState
from src.environment.reward import RewardCalculator, RewardConfig


def make_states() -> dict[str, TLSRuntimeState]:
    s = TLSRuntimeState(tls_id="TLS_1")
    s.lane_queue = [5.0, 3.0]
    return {"TLS_1": s}


@patch("src.environment.reward.traci")
def test_reward_decreases_when_waiting_improves(mock_traci):
    """When waiting time decreases, reward should be positive."""
    mock_traci.vehicle.getIDList.return_value = ["v1", "v2"]
    # Step 1: 100s waiting
    mock_traci.vehicle.getWaitingTime.return_value = 50.0  # 2 vehicles * 50 = 100
    mock_traci.exceptions.TraCIException = Exception

    calc = RewardCalculator(RewardConfig(waiting_delta_weight=1.0))
    calc.reset()
    states = make_states()

    # First call sets baseline: delta from the initial 0 to 100 -> r1 = -100
    r1, _ = calc.compute(states)
    assert r1 == -100.0

    # Step 2: waiting decreases to 50s
    mock_traci.vehicle.getWaitingTime.return_value = 25.0  # 50 total
    r2, _ = calc.compute(states)

    assert r2 == 50.0, f"Expected +50 (waiting fell 100 -> 50), got {r2}"


@patch("src.environment.reward.traci")
def test_reward_negative_when_congestion_grows(mock_traci):
    """When waiting time increases, reward should be negative."""
    mock_traci.vehicle.getIDList.return_value = ["v1"]
    mock_traci.vehicle.getWaitingTime.return_value = 10.0
    mock_traci.exceptions.TraCIException = Exception

    calc = RewardCalculator()
    calc.reset()
    states = make_states()

    calc.compute(states)  # baseline

    mock_traci.vehicle.getWaitingTime.return_value = 100.0  # big increase
    r, info = calc.compute(states)
    assert r == -90.0, f"Expected -(100-10), got {r}"
    assert info["reward/waiting_component"] == -90.0
    assert info["metrics/total_queue"] == 8.0


@patch("src.environment.reward.traci")
def test_reward_clipping(mock_traci):
    mock_traci.vehicle.getIDList.return_value = ["v1"] * 1000
    mock_traci.vehicle.getWaitingTime.return_value = 1000.0
    mock_traci.exceptions.TraCIException = Exception

    calc = RewardCalculator(RewardConfig(reward_clip=50.0))
    calc.reset()
    states = make_states()
    calc.compute(states)

    mock_traci.vehicle.getWaitingTime.return_value = 0.001  # huge improvement
    r, _ = calc.compute(states)
    assert abs(r) <= 50.0, f"Reward not clipped: {r}"


@patch("src.environment.reward.traci")
def test_global_switching_and_queue_terms(mock_traci):
    mock_traci.vehicle.getIDList.return_value = []
    mock_traci.exceptions.TraCIException = Exception
    calc = RewardCalculator(RewardConfig(waiting_delta_weight=0.0, queue_weight=0.5, switching_penalty=2.0))
    calc.reset()
    calc.set_switch_count(3)
    r, info = calc.compute(make_states())          # lane_queue = [5, 3] -> total queue 8
    assert info["reward/queue_component"] == -4.0
    assert info["reward/switch_component"] == -6.0
    assert r == -10.0 and info["metrics/switch_count"] == 3
