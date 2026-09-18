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

    # First call sets baseline
    r1, _ = calc.compute(states)
    # r1 should be negative (delta from 0 to 100)

    # Step 2: waiting decreases to 50s
    mock_traci.vehicle.getWaitingTime.return_value = 25.0  # 50 total
    r2, _ = calc.compute(states)

    assert r2 > 0, f"Expected positive reward when waiting decreases, got {r2}"


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
    assert r < 0, f"Expected negative reward when waiting grows, got {r}"


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
