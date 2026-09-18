"""
xai.py — Explainable AI (XAI) engine for CityMind AI PPO Policies.
Provides feature attribution, junction-level decision rationales, and
network-level traffic signal coordination analysis using Integrated Gradients.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.environment.observation import ObservationConfig
from src.scenario.tls_metadata import TLSSpec, load_tls_specs


@dataclass
class FeatureDescriptor:
    index: int
    tls_id: str
    tls_index: int
    group: str  # "phase", "elapsed", "queue", "vehicle_count", "speed", "density", "global"
    lane_id: Optional[str]
    name: str


def build_feature_descriptors(
    tls_specs: List[TLSSpec],
    cfg: Optional[ObservationConfig] = None,
) -> List[FeatureDescriptor]:
    """
    Build complete index-to-semantic descriptor mapping matching ObservationBuilder.
    """
    cfg = cfg or ObservationConfig()
    specs = sorted(tls_specs, key=lambda s: s.tls_id)
    descriptors: List[FeatureDescriptor] = []
    idx = 0

    for tls_idx, spec in enumerate(specs):
        # 1. Phase one-hot
        n_phases = len(spec.phases)
        for p in range(n_phases):
            descriptors.append(FeatureDescriptor(
                index=idx,
                tls_id=spec.tls_id,
                tls_index=tls_idx,
                group="phase",
                lane_id=None,
                name=f"{spec.tls_id}:phase_{p}",
            ))
            idx += 1

        # 2. Phase elapsed
        descriptors.append(FeatureDescriptor(
            index=idx,
            tls_id=spec.tls_id,
            tls_index=tls_idx,
            group="elapsed",
            lane_id=None,
            name=f"{spec.tls_id}:elapsed",
        ))
        idx += 1

        # 3. Queue
        if cfg.include_queue:
            for lane in spec.controlled_lanes:
                descriptors.append(FeatureDescriptor(
                    index=idx,
                    tls_id=spec.tls_id,
                    tls_index=tls_idx,
                    group="queue",
                    lane_id=lane,
                    name=f"{spec.tls_id}:{lane}:queue",
                ))
                idx += 1

        # 4. Vehicle count
        if cfg.include_vehicle_count:
            for lane in spec.controlled_lanes:
                descriptors.append(FeatureDescriptor(
                    index=idx,
                    tls_id=spec.tls_id,
                    tls_index=tls_idx,
                    group="vehicle_count",
                    lane_id=lane,
                    name=f"{spec.tls_id}:{lane}:vcount",
                ))
                idx += 1

        # 5. Speed
        if cfg.include_mean_speed:
            for lane in spec.controlled_lanes:
                descriptors.append(FeatureDescriptor(
                    index=idx,
                    tls_id=spec.tls_id,
                    tls_index=tls_idx,
                    group="speed",
                    lane_id=lane,
                    name=f"{spec.tls_id}:{lane}:speed",
                ))
                idx += 1

        # 6. Density / Occupancy
        if cfg.include_density:
            for lane in spec.controlled_lanes:
                descriptors.append(FeatureDescriptor(
                    index=idx,
                    tls_id=spec.tls_id,
                    tls_index=tls_idx,
                    group="density",
                    lane_id=lane,
                    name=f"{spec.tls_id}:{lane}:density",
                ))
                idx += 1

    # Global features
    if cfg.include_global:
        descriptors.append(FeatureDescriptor(
            index=idx,
            tls_id="GLOBAL",
            tls_index=-1,
            group="global",
            lane_id=None,
            name="global:total_vehicles",
        ))
        idx += 1
        descriptors.append(FeatureDescriptor(
            index=idx,
            tls_id="GLOBAL",
            tls_index=-1,
            group="global",
            lane_id=None,
            name="global:total_waiting_time",
        ))
        idx += 1

    return descriptors


class XAIExplainer:
    """
    Explainability engine for CityMind AI PPO models.
    Supports gradient-based attribution and Integrated Gradients.
    """

    def __init__(
        self,
        model_path: str,
        tls_json_path: str,
        obs_config: Optional[ObservationConfig] = None,
        device: str = "cpu",
    ) -> None:
        from stable_baselines3 import PPO

        self.device = torch.device(device)
        self.model = PPO.load(model_path, device=self.device)
        self.specs = load_tls_specs(Path(tls_json_path))
        self.tls_ids = [s.tls_id for s in self.specs]
        self.tls_id_to_idx = {tid: i for i, tid in enumerate(self.tls_ids)}

        self.obs_config = obs_config or ObservationConfig()
        self.descriptors = build_feature_descriptors(self.specs, self.obs_config)
        self.num_features = len(self.descriptors)

    def _get_logits(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass extracting action logits from PPO Actor network."""
        features = self.model.policy.extract_features(obs_tensor)
        latent_pi, _ = self.model.policy.mlp_extractor(features)
        distribution = self.model.policy._get_action_dist_from_latent(latent_pi)
        return distribution.distribution.logits

    def attribute(
        self,
        obs: np.ndarray,
        tls_idx: int,
        method: str = "integrated_gradients",
        steps: int = 20,
    ) -> np.ndarray:
        """
        Compute feature attribution for a single traffic light's switch decision.
        
        Args:
            obs: Observation vector of shape (num_features,)
            tls_idx: Index of the traffic light in action space [0..num_tls-1]
            method: 'integrated_gradients' or 'input_x_gradient'
            steps: Number of integration steps for Integrated Gradients

        Returns:
            Attribution vector of shape (num_features,)
        """
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        if method == "input_x_gradient":
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            obs_t.requires_grad_(True)
            logits = self._get_logits(obs_t)
            logits[0, tls_idx].backward()
            grad = obs_t.grad.detach().cpu().numpy()[0]
            return obs * grad

        # Integrated Gradients
        baseline = np.zeros_like(obs)
        grads = []
        for step in range(1, steps + 1):
            interpolated = baseline + (float(step) / steps) * (obs - baseline)
            obs_t = torch.tensor(interpolated, dtype=torch.float32, device=self.device).unsqueeze(0)
            obs_t.requires_grad_(True)
            logits = self._get_logits(obs_t)
            logits[0, tls_idx].backward()
            grads.append(obs_t.grad.detach().cpu().numpy()[0])

        avg_grads = np.mean(grads, axis=0)
        return (obs - baseline) * avg_grads

    def explain_junction(
        self,
        obs: np.ndarray,
        tls_id: str,
        top_k: int = 5,
        method: str = "integrated_gradients",
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive, structured explanation for a specific junction.
        """
        if tls_id not in self.tls_id_to_idx:
            raise ValueError(f"Unknown TLS ID: {tls_id}")
        tls_idx = self.tls_id_to_idx[tls_id]

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self._get_logits(obs_t)[0]
            prob = torch.sigmoid(logits[tls_idx]).item()
            action = int(prob >= 0.5)

        attributions = self.attribute(obs, tls_idx, method=method)

        # Ranked features
        sorted_indices = np.argsort(-attributions)
        top_positive_indices = [idx for idx in sorted_indices if attributions[idx] > 0][:top_k]
        top_negative_indices = [idx for idx in reversed(sorted_indices) if attributions[idx] < 0][:top_k]

        def _format_item(i: int):
            d = self.descriptors[i]
            return {
                "feature_index": int(i),
                "name": d.name,
                "group": d.group,
                "tls_id": d.tls_id,
                "lane_id": d.lane_id,
                "value": float(obs[i]),
                "attribution": float(attributions[i]),
            }

        top_positive = [_format_item(i) for i in top_positive_indices]
        top_negative = [_format_item(i) for i in top_negative_indices]

        # Group attribution totals
        group_totals: Dict[str, float] = {}
        for i, val in enumerate(attributions):
            grp = self.descriptors[i].group
            group_totals[grp] = group_totals.get(grp, 0.0) + abs(float(val))

        # Local vs Neighbor influence
        local_attr = 0.0
        neighbor_attr = 0.0
        global_attr = 0.0
        for i, val in enumerate(attributions):
            d = self.descriptors[i]
            if d.tls_id == tls_id:
                local_attr += abs(float(val))
            elif d.tls_id == "GLOBAL":
                global_attr += abs(float(val))
            else:
                neighbor_attr += abs(float(val))

        total_influence = max(local_attr + neighbor_attr + global_attr, 1e-6)

        # Generate readable natural-language summary
        action_verb = "SWITCH to the next phase" if action == 1 else "HOLD the current green phase"
        confidence = prob if action == 1 else (1.0 - prob)
        
        primary_drivers = []
        if top_positive:
            d0 = top_positive[0]
            primary_drivers.append(f"{d0['group']} on {d0['lane_id'] or d0['name']} (score: +{d0['attribution']:.3f})")
        if len(top_positive) > 1:
            d1 = top_positive[1]
            primary_drivers.append(f"{d1['group']} on {d1['lane_id'] or d1['name']} (score: +{d1['attribution']:.3f})")

        drivers_str = " and ".join(primary_drivers) if primary_drivers else "baseline phase duration"
        narrative = (
            f"Junction '{tls_id}' decided to {action_verb} with {confidence*100:.1f}% confidence. "
            f"This decision was primarily influenced by {drivers_str}. "
            f"Local traffic conditions contributed {local_attr/total_influence*100:.1f}% of decision weight, "
            f"while neighboring junctions contributed {neighbor_attr/total_influence*100:.1f}%."
        )

        return {
            "tls_id": tls_id,
            "tls_index": tls_idx,
            "action": action,
            "switch_probability": float(prob),
            "narrative": narrative,
            "local_influence_pct": float(round(local_attr / total_influence * 100, 1)),
            "neighbor_influence_pct": float(round(neighbor_attr / total_influence * 100, 1)),
            "group_breakdown": {k: float(round(v, 4)) for k, v in group_totals.items()},
            "top_positive_features": top_positive,
            "top_negative_features": top_negative,
        }

    def compute_coordination_matrix(
        self,
        obs: np.ndarray,
        method: str = "input_x_gradient",
    ) -> np.ndarray:
        """
        Compute the (num_tls, num_tls) interaction matrix.
        Row i represents the influence that all TLS j have on TLS i's decision.
        """
        n_tls = len(self.tls_ids)
        matrix = np.zeros((n_tls, n_tls), dtype=np.float32)

        for i in range(n_tls):
            attr = self.attribute(obs, i, method=method)
            for feat_idx, val in enumerate(attr):
                d = self.descriptors[feat_idx]
                if d.tls_index >= 0:
                    matrix[i, d.tls_index] += abs(float(val))

        # Row-normalize
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return matrix / row_sums

    def plot_top_drivers(
        self,
        explanation: Dict[str, Any],
        output_path: str,
    ) -> None:
        """Plot horizontal bar chart of top positive and negative drivers."""
        items = explanation["top_positive_features"][:8] + explanation["top_negative_features"][:8]
        if not items:
            return

        names = [item["name"].split(":")[-2] + " " + item["name"].split(":")[-1] for item in items]
        values = [item["attribution"] for item in items]
        colors = ["#10b981" if v > 0 else "#ef4444" for v in values]

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color=colors, align="center", edgecolor="none")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Integrated Gradients Attribution Score", fontsize=10)
        ax.set_title(f"Decision Drivers for Junction {explanation['tls_id']}\nAction: {'SWITCH' if explanation['action'] == 1 else 'HOLD'}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()

    def plot_coordination_heatmap(
        self,
        coordination_matrix: np.ndarray,
        output_path: str,
        top_k: int = 20,
    ) -> None:
        """Plot coordination heatmap of top K interconnected junctions."""
        # Find top K junctions with highest non-diagonal interaction
        diag_mask = np.eye(len(coordination_matrix), dtype=bool)
        off_diag = coordination_matrix.copy()
        off_diag[diag_mask] = 0.0
        top_indices = np.argsort(-off_diag.sum(axis=1))[:top_k]

        sub_matrix = coordination_matrix[np.ix_(top_indices, top_indices)]
        labels = [self.tls_ids[i][:15] for i in top_indices]

        fig, ax = plt.subplots(figsize=(11, 9))
        cax = ax.matshow(sub_matrix, cmap="magma", interpolation="nearest")
        fig.colorbar(cax, fraction=0.046, pad=0.04, label="Decision Influence Weight")

        ax.set_xticks(range(top_k))
        ax.set_yticks(range(top_k))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"Top {top_k} Inter-Junction Coordination Influence Matrix", fontsize=12, fontweight="bold", pad=20)
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
