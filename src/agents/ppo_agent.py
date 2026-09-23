"""
ppo_agent.py
============
PPO agent configuration for centralized traffic signal control.

Uses Stable-Baselines3 PPO with:
  - MultiBinary action space (compatible with SB3 PPO)
  - MLP policy (256x256 default)
  - Apple Silicon MPS acceleration where available
  - Artifact saving: model, config, tensorboard logs
"""
from __future__ import annotations

import os
import time
import json
from pathlib import Path
from typing import Any

import torch
import yaml
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from src.environment.centralized_sumo_env import CentralizedSumoEnv
from src.environment.observation import ObservationConfig
from src.environment.reward import RewardConfig
from src.environment.safety import SafetyConfig


def make_env(
    sumo_cfg: str,
    tls_metadata_json: str,
    seed: int = 42,
    demand_id: str = "1.0x",
    obs_config: ObservationConfig | None = None,
    reward_config: RewardConfig | None = None,
    safety_config: SafetyConfig | None = None,
    decision_interval: int = 5,
    episode_duration: int = 3600,
    begin_time: int = 0,
    record_trajectories: bool = False,
    tripinfo_path: str | None = None,
) -> Monitor:
    """Factory for a single monitored CentralizedSumoEnv.

    episode_duration is the ABSOLUTE simulation end-time, not a duration
    relative to begin_time (matches make_ippo_env's convention).
    """
    sumo_extra_args = ["--begin", str(begin_time), "--end", str(episode_duration)]
    if tripinfo_path:
        from pathlib import Path as _Path
        _Path(tripinfo_path).parent.mkdir(parents=True, exist_ok=True)
        sumo_extra_args += ["--tripinfo-output", str(tripinfo_path)]
    env = CentralizedSumoEnv(
        sumo_cfg=sumo_cfg,
        tls_metadata_json=tls_metadata_json,
        sumo_binary="sumo",
        decision_interval=decision_interval,
        controller="rl",
        seed=seed,
        demand_id=demand_id,
        obs_config=obs_config,
        reward_config=reward_config,
        safety_config=safety_config,
        record_trajectories=record_trajectories,
        episode_duration=episode_duration,
        sumo_extra_args=sumo_extra_args,
    )
    return Monitor(env)


class PPOAgent:
    """
    Wrapper around Stable-Baselines3 PPO for CityMind AI.

    Handles:
      - Environment creation
      - PPO configuration from YAML
      - Device selection (MPS > CPU)
      - Training with callbacks
      - Artifact saving
    """

    def __init__(
        self,
        sumo_cfg: str,
        tls_metadata_json: str,
        ppo_config_path: str = "configs/ppo.yaml",
        env_config_path: str = "configs/environment.yaml",
        seed: int = 42,
        demand_id: str = "1.0x",
        output_dir: str = "artifacts/experiments/ppo",
    ) -> None:
        self.sumo_cfg = sumo_cfg
        self.tls_metadata_json = tls_metadata_json
        self.seed = seed
        self.demand_id = demand_id
        self.output_dir = Path(output_dir)

        # Load configs
        with open(ppo_config_path) as f:
            self.ppo_cfg: dict = yaml.safe_load(f)
        with open(env_config_path) as f:
            self.env_cfg: dict = yaml.safe_load(f)

        # Select compute device
        if self.ppo_cfg.get("device", "auto") == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
                print("[PPOAgent] Using Apple Silicon MPS backend")
            else:
                self.device = "cpu"
                print("[PPOAgent] Using CPU")
        else:
            self.device = self.ppo_cfg["device"]

        self.model: PPO | None = None
        self._run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_seed{seed}"
        self.run_dir = self.output_dir / self._run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _build_sub_configs(self):
        """Build typed config objects from YAML dicts."""
        env_cfg = self.env_cfg
        obs_cfg = ObservationConfig(
            normalize=env_cfg["observation"].get("normalize", True),
            max_lane_vehicles=env_cfg["observation"].get("max_lane_vehicles", 50.0),
            max_speed=env_cfg["observation"].get("max_speed", 13.9),
        )
        rwd_cfg = RewardConfig(
            waiting_delta_weight=env_cfg["reward"].get("waiting_delta_weight", 1.0),
            queue_weight=env_cfg["reward"].get("queue_weight", 0.0),
            switching_penalty=env_cfg["reward"].get("switching_penalty", 0.0),
        )
        saf_cfg = SafetyConfig(
            min_green=env_cfg["safety"].get("min_green", 5.0),
            max_green=env_cfg["safety"].get("max_green", 60.0),
            yellow_duration=env_cfg["safety"].get("yellow_duration", 3.0),
            all_red_duration=env_cfg["safety"].get("all_red_duration", 1.0),
        )
        return obs_cfg, rwd_cfg, saf_cfg

    def build_model(self) -> PPO:
        """Instantiate the PPO model with the configured environment."""
        set_random_seed(self.seed)
        obs_cfg, rwd_cfg, saf_cfg = self._build_sub_configs()

        sumo_cfg_dict = self.env_cfg.get("sumo", {})
        env = make_env(
            sumo_cfg=self.sumo_cfg,
            tls_metadata_json=self.tls_metadata_json,
            seed=self.seed,
            demand_id=self.demand_id,
            obs_config=obs_cfg,
            reward_config=rwd_cfg,
            safety_config=saf_cfg,
            decision_interval=self.env_cfg.get("decision_interval", 5),
            episode_duration=int(sumo_cfg_dict.get("end", 3600)),
            begin_time=int(sumo_cfg_dict.get("begin", 0)),
        )

        ppo = self.ppo_cfg
        # Copy, don't mutate: self.ppo_cfg is dumped verbatim to config.yaml
        # later, and a raw torch.nn class object isn't yaml.safe_load-able.
        policy_kwargs = dict(ppo.get("policy_kwargs", {}))
        if "activation_fn" in policy_kwargs:
            import torch.nn as nn
            activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU}
            policy_kwargs["activation_fn"] = activation_map.get(
                policy_kwargs["activation_fn"], nn.Tanh
            )

        self.model = PPO(
            policy=ppo.get("policy", "MlpPolicy"),
            env=env,
            learning_rate=float(ppo.get("learning_rate", 3e-4)),
            n_steps=int(ppo.get("n_steps", 2048)),
            batch_size=int(ppo.get("batch_size", 64)),
            n_epochs=int(ppo.get("n_epochs", 10)),
            gamma=float(ppo.get("gamma", 0.99)),
            gae_lambda=float(ppo.get("gae_lambda", 0.95)),
            clip_range=float(ppo.get("clip_range", 0.2)),
            ent_coef=float(ppo.get("ent_coef", 0.01)),
            vf_coef=float(ppo.get("vf_coef", 0.5)),
            max_grad_norm=float(ppo.get("max_grad_norm", 0.5)),
            normalize_advantage=bool(ppo.get("normalize_advantage", True)),
            tensorboard_log=str(self.run_dir / "tb_logs"),
            device=self.device,
            verbose=1,
            seed=self.seed,
            policy_kwargs=policy_kwargs if policy_kwargs else None,
        )
        print(f"[PPOAgent] Model built. Obs={self.model.observation_space.shape}, "
              f"Act={self.model.action_space}")
        return self.model

    def train(self) -> PPO:
        """Run training with callbacks and save artifacts."""
        if self.model is None:
            self.build_model()

        total_timesteps = int(self.ppo_cfg.get("total_timesteps", 500_000))
        checkpoint_freq = int(self.ppo_cfg.get("checkpoint_freq", 100_000))

        # Callbacks
        checkpoint_cb = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(self.run_dir / "checkpoints"),
            name_prefix="ppo_citymind",
        )
        callbacks = CallbackList([checkpoint_cb])

        print(f"\n[PPOAgent] Starting training — {total_timesteps:,} timesteps")
        print(f"[PPOAgent] Run dir: {self.run_dir}\n")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name="ppo",
            progress_bar=True,
        )

        # Save final model
        model_path = self.run_dir / "model_final.zip"
        self.model.save(str(model_path))
        print(f"\n[PPOAgent] Model saved: {model_path}")

        # Save config artifact
        self._save_config_artifact(total_timesteps)
        return self.model

    def _save_config_artifact(self, total_timesteps: int) -> None:
        """Write config.yaml + metadata to run_dir."""
        artifact = {
            "algorithm": "PPO",
            "seed": self.seed,
            "total_timesteps": total_timesteps,
            "device": self.device,
            "demand_id": self.demand_id,
            "ppo_config": self.ppo_cfg,
            "env_config": self.env_cfg,
        }
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.dump(artifact, f, default_flow_style=False)

        # git commit hash
        try:
            import subprocess
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            commit = "unknown"
        (self.run_dir / "git_commit.txt").write_text(commit)

        print(f"[PPOAgent] Config artifact saved: {self.run_dir / 'config.yaml'}")

    def load(self, model_path: str) -> PPO:
        """Load a saved model for evaluation."""
        self.model = PPO.load(model_path, device=self.device)
        return self.model

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """Predict action from observation."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call build_model() or load() first.")
        return self.model.predict(obs, deterministic=deterministic)
