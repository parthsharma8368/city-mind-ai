"""
action.py
=========
Action space definition and decoder for the centralized environment.

Action space: MultiBinary(num_tls)
  action[i] = 0  →  TLS i keeps its current phase
  action[i] = 1  →  TLS i requests advance to next green phase

The safety layer enforces all timing constraints.
"""
from __future__ import annotations

import numpy as np
from src.scenario.tls_metadata import TLSSpec


class ActionDecoder:
    """
    Decodes a MultiBinary action vector into a per-TLS action dict.

    The ordering matches the sorted tls_id list used in ObservationBuilder.
    """

    def __init__(self, tls_specs: list[TLSSpec]) -> None:
        self.tls_ids: list[str] = sorted(s.tls_id for s in tls_specs)
        self.num_tls: int = len(self.tls_ids)

    def decode(self, action: np.ndarray) -> dict[str, int]:
        """
        Convert a flat MultiBinary array to {tls_id: action_int}.

        Parameters
        ----------
        action : np.ndarray of shape (num_tls,) with values in {0, 1}

        Returns
        -------
        dict[str, int] — one action value per TLS id
        """
        if len(action) != self.num_tls:
            raise ValueError(
                f"Action length mismatch: expected {self.num_tls}, got {len(action)}"
            )
        return {tid: int(action[i]) for i, tid in enumerate(self.tls_ids)}

    def action_space(self):
        """Return a gymnasium.spaces.MultiBinary action space."""
        import gymnasium as gym  # lazy import
        return gym.spaces.MultiBinary(self.num_tls)

    def __repr__(self) -> str:
        return f"ActionDecoder(num_tls={self.num_tls}, tls_ids={self.tls_ids})"
