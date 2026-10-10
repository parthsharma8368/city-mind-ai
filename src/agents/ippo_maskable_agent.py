"""
ippo_maskable_agent.py
=======================
Parameter-shared IPPO training for the phase-selection action space
(RL-strategy Phase 2), using sb3-contrib's MaskablePPO so the policy never
places probability mass on a phase an intersection doesn't have.

Integration gap (read before changing num_vec_envs): MaskablePPO's
get_action_masks(env) calls env.env_method("action_masks") on the training
VecEnv every rollout step. None of SuperSuit's VecEnv wrappers implement
env_method at all (verified against the installed supersuit==3.11.0
source, not assumed) — not MarkovVectorEnv (pettingzoo_env_to_vec_env_v1's
output), not ConcatVecEnv, not ProcConcatVec (concat_vec_envs_v1's output
for num_cpus>1), not the SB3VecEnvWrapper compatibility shim
(base_class="stable_baselines3") that only patches env_is_wrapped/get_attr.

MaskableActionWrapper below closes the gap by reaching directly into
ConcatVecEnv.vec_envs[i].par_env (the live DecentralizedSumoEnv instances)
and calling .action_masks() — which only works because those instances are
directly addressable Python objects in THIS process. That requires staying
on ConcatVecEnv (concat_vec_envs_v1's num_cpus=0-or-1 path — see
constructors.py), never ProcConcatVec, whose sub-environments live in
separate subprocesses reachable only through its own pipe protocol
(async_loop), which has no "env_method" instruction and would need a
supersuit patch to gain one — a separate task, not attempted here.

Independently of masking, this project's SUMO connection is libsumo — a
single-connection-per-PROCESS binding (see ippo_agent.py's make_ippo_env
docstring) — so ConcatVecEnv (all instances sharing one process) can only
safely hold ONE live SUMO connection at a time regardless of masking.
Together these two constraints are why make_ippo_maskable_env requires
num_vec_envs=1: it is not an arbitrary restriction to lift by writing more
wrapper code, it is the ceiling of what ConcatVecEnv + libsumo can run
safely, until ProcConcatVec itself gains cross-process env_method support
(and, separately, libsumo's one-connection-per-process constraint is
addressed for whatever replaces it). See configs/ippo_phase_select.yaml's
comment for the resulting wall-clock estimate.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import supersuit as ss
import torch
import yaml
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor

from src.agents.ippo_agent import _close_env_with_timeout
from src.agents.model_contract import build_artifact, validate_model_contract, write_run_artifacts
from src.environment.decentralized_sumo_env import DecentralizedSumoEnv
from src.environment.local_observation import LOCAL_OBS_SIZE, LocalObservationConfig
from src.environment.local_reward import LocalRewardConfig
from src.environment.safety import SafetyConfig
from src.scenario.tls_metadata import MAX_GREEN_PHASES


class MaskableActionWrapper(VecEnvWrapper):
    """Exposes DecentralizedSumoEnv.action_masks() through the VecEnv
    env_method interface MaskablePPO requires. self.venv is an
    SB3VecEnvWrapper wrapping a ConcatVecEnv (concat_vec_envs_v1's
    num_cpus<=1 path — see module docstring for why it must stay there);
    ConcatVecEnv.vec_envs holds the live MarkovVectorEnv instance(s), each
    with a .par_env that is the actual DecentralizedSumoEnv."""

    def env_method(self, method_name, *args, indices=None, **kwargs):
        if method_name == "action_masks":
            masks: list[np.ndarray] = []
            for sub_env in self.venv.venv.vec_envs:  # SB3VecEnvWrapper.venv -> ConcatVecEnv.vec_envs
                par_env = sub_env.par_env
                masks_by_agent = par_env.action_masks()
                masks.extend(masks_by_agent[agent] for agent in par_env.possible_agents)
            return masks
        return super().env_method(method_name, *args, indices=indices, **kwargs)

    def has_attr(self, attr_name: str) -> bool:
        # MaskablePPO's is_masking_supported() probes this before ever
        # calling env_method — ConcatVecEnv (a bare gymnasium.vector.VectorEnv)
        # has no has_attr at all, so the default passthrough would raise
        # here rather than answer, for the one attribute this wrapper exists
        # to provide.
        if attr_name == "action_masks":
            return True
        return super().has_attr(attr_name)

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()


def make_ippo_maskable_env(
    sumo_cfg: str,
    tls_metadata_json: str,
    seed: int = 42,
    demand_id: str = "1.0x",
    obs_config: LocalObservationConfig | None = None,
    reward_config: LocalRewardConfig | None = None,
    safety_config: SafetyConfig | None = None,
    decision_interval: int = 5,
    episode_duration: int = 3600,
    begin_time: int = 0,
    num_vec_envs: int = 1,
    seed_mode: str = "fixed",
    warm_start_seconds: float = 0.0,
):
    """Build DecentralizedSumoEnv (phase-selection action space) and wrap it
    for MaskablePPO. num_vec_envs must be 1 — see module docstring."""
    if num_vec_envs != 1:
        raise NotImplementedError(
            f"make_ippo_maskable_env got num_vec_envs={num_vec_envs}, but masked "
            "training only supports 1 (see this module's docstring: ConcatVecEnv, "
            "the only base that exposes live sub-env objects for action_masks(), "
            "cannot safely hold more than one libsumo connection per process)."
        )

    env = DecentralizedSumoEnv(
        sumo_cfg=sumo_cfg,
        tls_metadata_json=tls_metadata_json,
        sumo_binary="sumo",
        decision_interval=decision_interval,
        seed=seed,
        demand_id=demand_id,
        obs_config=obs_config,
        reward_config=reward_config,
        safety_config=safety_config,
        episode_duration=episode_duration,
        sumo_extra_args=["--begin", str(begin_time), "--end", str(episode_duration)],
        seed_mode=seed_mode,
        warm_start_seconds=warm_start_seconds,
    )
    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    vec_env = ss.concat_vec_envs_v1(
        vec_env, num_vec_envs=num_vec_envs, num_cpus=1, base_class="stable_baselines3"
    )
    vec_env = MaskableActionWrapper(vec_env)
    vec_env = VecMonitor(vec_env)
    return vec_env


class IPPOMaskableAgent:
    """Mirrors IPPOAgent's config-driven interface for the phase-selection,
    MaskablePPO-trained variant. Reads environment_phase_select.yaml /
    ippo_phase_select.yaml by default (NOT environment.yaml/ippo.yaml —
    those stay exactly as IPPOAgent/PPOAgent left them; see
    environment_phase_select.yaml's header comment for why this needed a
    separate file rather than editing the shared one in place)."""

    def __init__(
        self,
        sumo_cfg: str,
        tls_metadata_json: str,
        ppo_config_path: str = "configs/ippo_phase_select.yaml",
        env_config_path: str = "configs/environment_phase_select.yaml",
        seed: int = 42,
        demand_id: str = "1.0x",
        output_dir: str = "artifacts/experiments/ippo_phase_select",
    ) -> None:
        self.sumo_cfg = sumo_cfg
        self.tls_metadata_json = tls_metadata_json
        self.seed = seed
        self.demand_id = demand_id
        self.output_dir = Path(output_dir)

        with open(ppo_config_path) as f:
            self.ppo_cfg: dict = yaml.safe_load(f)
        with open(env_config_path) as f:
            self.env_cfg: dict = yaml.safe_load(f)

        if self.ppo_cfg.get("device", "auto") == "auto":
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = self.ppo_cfg["device"]

        self.model: MaskablePPO | None = None
        self._run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_seed{seed}"
        # Created lazily (build_model / resume): loading a model for evaluation
        # must not leave an empty experiment run behind.
        self.run_dir = self.output_dir / self._run_id

    def build_model(self) -> MaskablePPO:
        set_random_seed(self.seed)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        env = self._build_env()

        ppo = self.ppo_cfg
        policy_kwargs = dict(ppo.get("policy_kwargs", {}))
        if "activation_fn" in policy_kwargs:
            import torch.nn as nn
            activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU}
            policy_kwargs["activation_fn"] = activation_map.get(policy_kwargs["activation_fn"], nn.Tanh)

        self.model = MaskablePPO(
            policy=ppo.get("policy", "MlpPolicy"),
            env=env,
            learning_rate=float(ppo.get("learning_rate", 3e-4)),
            n_steps=int(ppo.get("n_steps", 720)),
            batch_size=int(ppo.get("batch_size", 8820)),
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
            policy_kwargs=policy_kwargs if policy_kwargs else None,
        )
        print(f"[IPPOMaskableAgent] Model built. Obs={self.model.observation_space.shape}, "
              f"Act={self.model.action_space}, num_agents(envs)={self.model.n_envs}")
        return self.model

    def _build_env(self):
        """Env construction shared by build_model() and resume_training() —
        pulled out so a resumed run gets an env built from the exact same
        config-derived parameters as the original, not a hand-copied subset."""
        reward_cfg = LocalRewardConfig(
            waiting_delta_weight=self.env_cfg["reward"].get("waiting_delta_weight", 0.3),
            queue_weight=self.env_cfg["reward"].get("queue_weight", 0.0),
            switching_penalty=self.env_cfg["reward"].get("switching_penalty", 0.05),
            spillback_weight=self.env_cfg["reward"].get("spillback_weight", 0.2),
            pressure_weight=self.env_cfg["reward"].get("pressure_weight", 0.3),
            neglected_pressure_weight=self.env_cfg["reward"].get("neglected_pressure_weight", 1.0),
        )
        safety_cfg = SafetyConfig(
            min_green=self.env_cfg["safety"].get("min_green", 5.0),
            max_green=self.env_cfg["safety"].get("max_green", 60.0),
            yellow_duration=self.env_cfg["safety"].get("yellow_duration", 3.0),
            all_red_duration=self.env_cfg["safety"].get("all_red_duration", 1.0),
        )
        sumo_cfg_dict = self.env_cfg.get("sumo", {})
        begin_time = int(sumo_cfg_dict.get("begin", 0))
        end_time = int(sumo_cfg_dict.get("end", 3600))
        num_vec_envs = int(self.ppo_cfg.get("num_vec_envs", 1))
        return make_ippo_maskable_env(
            sumo_cfg=self.sumo_cfg,
            tls_metadata_json=self.tls_metadata_json,
            seed=self.seed,
            demand_id=self.demand_id,
            reward_config=reward_cfg,
            safety_config=safety_cfg,
            decision_interval=self.env_cfg.get("decision_interval", 5),
            episode_duration=end_time,
            begin_time=begin_time,
            num_vec_envs=num_vec_envs,
            seed_mode=self.env_cfg.get("seed_mode", "per_episode"),
            warm_start_seconds=float(sumo_cfg_dict.get("warm_start_seconds", 0.0)),
        )

    def resume_training(self, checkpoint_path: str, completed_timesteps: int | None = None) -> MaskablePPO:
        """Continue training from a saved checkpoint after an interruption
        (e.g. the process was killed by something outside this run, like a
        machine restart) — not a substitute for train() on a fresh run.

        completed_timesteps defaults to parsing it from the checkpoint's own
        filename (SB3's CheckpointCallback names them
        "<prefix>_<num_timesteps>_steps.zip"), so total_timesteps in the
        config is honored as the real target rather than restarting the
        countdown from the checkpoint.
        """
        if completed_timesteps is None:
            stem = Path(checkpoint_path).stem  # e.g. "ippo_phase_select_2822400_steps"
            completed_timesteps = int(stem.rsplit("_", 2)[-2])

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model = self.load(checkpoint_path)
        env = self._build_env()
        self.model.set_env(env)
        print(f"[IPPOMaskableAgent] Resumed from {checkpoint_path} at {completed_timesteps:,} timesteps.")

        total_timesteps = int(self.ppo_cfg.get("total_timesteps", 7_056_000))
        remaining = total_timesteps - completed_timesteps
        if remaining <= 0:
            raise ValueError(
                f"Checkpoint already has {completed_timesteps:,} timesteps >= "
                f"target {total_timesteps:,} — nothing to resume."
            )
        return self._run_learn(remaining, reset_num_timesteps=False)

    def train(self) -> MaskablePPO:
        if self.model is None:
            self.build_model()

        total_timesteps = int(self.ppo_cfg.get("total_timesteps", 7_056_000))
        return self._run_learn(total_timesteps, reset_num_timesteps=True)

    def _run_learn(self, timesteps: int, reset_num_timesteps: bool) -> MaskablePPO:
        checkpoint_freq = int(self.ppo_cfg.get("checkpoint_freq", 705_600))

        save_freq = max(checkpoint_freq // self.model.n_envs, 1)
        checkpoint_cb = CheckpointCallback(
            save_freq=save_freq,
            save_path=str(self.run_dir / "checkpoints"),
            name_prefix="ippo_phase_select",
        )
        callbacks = CallbackList([checkpoint_cb])

        print(f"\n[IPPOMaskableAgent] Training for {timesteps:,} more timesteps "
              f"({timesteps // self.model.n_envs:,} real decision steps across {self.model.n_envs} agents)")
        self._save_config_artifact(timesteps, status="started")
        status = "failed"
        try:
            self.model.learn(
                total_timesteps=timesteps, callback=callbacks, tb_log_name="ippo_phase_select",
                reset_num_timesteps=reset_num_timesteps, progress_bar=True,
            )
            status = "finished"
            model_path = self.run_dir / "model_final.zip"
            self.model.save(str(model_path))
            print(f"\n[IPPOMaskableAgent] Model saved: {model_path}")
        finally:
            self._save_config_artifact(timesteps, status=status)
            _close_env_with_timeout(self.model.env, timeout=30.0)
        return self.model

    def _save_config_artifact(self, requested_timesteps: int, status: str = "finished") -> None:
        model = self.model
        artifact = build_artifact(
            algorithm="MaskablePPO (parameter-shared, phase-selection)",
            seed=self.seed,
            device=self.device,
            demand_id=self.demand_id,
            ppo_cfg=self.ppo_cfg,
            env_cfg=self.env_cfg,
            obs_size=int(model.observation_space.shape[0]) if model is not None else LOCAL_OBS_SIZE,
            action_space=str(model.action_space) if model is not None else f"Discrete({MAX_GREEN_PHASES})",
            requested_timesteps=requested_timesteps,
            num_timesteps=int(model.num_timesteps) if model is not None else 0,
            status=status,
            extra={"local_obs_size": LOCAL_OBS_SIZE, "max_green_phases": MAX_GREEN_PHASES,
                   "sumo_cfg": self.sumo_cfg, "tls_metadata_json": self.tls_metadata_json,
                   "seed_mode": self.env_cfg.get("seed_mode", "per_episode")},
        )
        write_run_artifacts(self.run_dir, artifact)

    def load(self, model_path: str) -> MaskablePPO:
        """Load a phase-selection checkpoint; raises ModelContractError on an
        obs-size / action-space mismatch."""
        model = MaskablePPO.load(model_path, device=self.device)
        validate_model_contract(
            model, LOCAL_OBS_SIZE, model_path=model_path, expected_action_n=MAX_GREEN_PHASES
        )
        self.model = model
        return self.model

    def predict(self, obs, action_masks: np.ndarray | None = None, deterministic: bool = True):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call build_model() or load() first.")
        return self.model.predict(obs, action_masks=action_masks, deterministic=deterministic)
