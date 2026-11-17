"""Pipeline de treinamento PPO com PBRS, Frame Stacking e Curriculum Learning."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize

from cartpole_sim.config import DriveMode, SimConfig
from cartpole_sim.curriculum import CurriculumManager, DEFAULT_STAGES
from cartpole_sim.env import CartPoleStepperEnv


def _make_single_env(
    cfg: SimConfig,
    seed: int,
    rank: int,
) -> Callable[[], Monitor]:
    def _init() -> Monitor:
        env_cfg = SimConfig(
            physics=dataclasses.replace(cfg.physics),
            sensor=dataclasses.replace(cfg.sensor),
            disturbance=dataclasses.replace(cfg.disturbance),
            reward=dataclasses.replace(cfg.reward),
            max_episode_steps=cfg.max_episode_steps,
            eval_mode=cfg.eval_mode,
            randomize_disturbances=cfg.randomize_disturbances,
        )
        env = CartPoleStepperEnv(config=env_cfg)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _init


def build_vec_env(
    cfg: SimConfig,
    n_envs: int,
    seed: int,
    *,
    normalize_reward: bool = True,
    training: bool = True,
) -> VecEnv:
    """Cria VecEnv (paralelo em SubprocVecEnv se n_envs > 1) com VecNormalize consistente."""
    fns = [_make_single_env(cfg, seed=seed, rank=i) for i in range(n_envs)]
    vec: VecEnv = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    # Observações já são normalizadas fisicamente em CartPoleStepperEnv (norm_obs=False)
    # garantindo deploy direto do modelo .zip sem estatísticas móveis externas.
    # Envolvemos também o eval_env em VecNormalize(norm_reward=False) para manter
    # compatibilidade com EvalCallback.sync_envs_normalization preservando retornos brutos.
    return VecNormalize(
        vec,
        training=training,
        norm_obs=False,
        norm_reward=normalize_reward,
        clip_reward=10.0,
        gamma=cfg.reward.gamma,
    )


class CurriculumEvalCallback(EvalCallback):
    """EvalCallback que sincroniza e avança estágios de dificuldade do curriculum."""

    def __init__(
        self,
        eval_env: VecEnv,
        curriculum: CurriculumManager,
        output_dir: Path,
        *,
        eval_freq: int,
        n_eval_episodes: int = 10,
        verbose: int = 1,
    ) -> None:
        super().__init__(
            eval_env=eval_env,
            best_model_save_path=str(output_dir),
            log_path=str(output_dir / "eval_logs"),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=verbose,
        )
        self.curriculum = curriculum
        self.output_dir = output_dir
        self._last_check_timesteps = 0

    def _apply_stage_to_envs(self) -> None:
        dist = self.curriculum.current_stage.disturbance
        self.training_env.env_method("set_disturbance_config", dist)
        self.eval_env.env_method("set_disturbance_config", dist)

    def _on_training_start(self) -> None:
        super()._on_training_start()
        self._apply_stage_to_envs()
        self._last_check_timesteps = self.num_timesteps
        if self.verbose > 0:
            print(f"[Curriculum] Iniciando em: {self.curriculum.current_stage.name}")

    def _on_step(self) -> bool:
        prev_eval_count = self.evaluations_timesteps[-1] if self.evaluations_timesteps else -1
        continue_training = super()._on_step()

        ran_eval = (
            len(self.evaluations_timesteps) > 0
            and self.evaluations_timesteps[-1] != prev_eval_count
        )
        if ran_eval:
            delta_steps = self.num_timesteps - self._last_check_timesteps
            self._last_check_timesteps = self.num_timesteps
            mean_rew = float(self.last_mean_reward)
            self.logger.record("curriculum/stage", self.curriculum.current_idx)

            old_stage = self.curriculum.current_stage
            if self.curriculum.maybe_advance(mean_rew, delta_steps):
                if self.model is not None:
                    self.model.save(self.output_dir / f"stage_{old_stage.stage_id}_mastered.zip")
                # Reinicia referência de melhor recompensa pois o novo estágio é mais difícil
                self.best_mean_reward = -np.inf
                self._apply_stage_to_envs()
                if self.verbose > 0:
                    print(
                        f"\n[Curriculum] Promovido! "
                        f"(reward={mean_rew:.1f} >= {old_stage.promote_mean_reward:.1f}) -> "
                        f"{self.curriculum.current_stage.name}\n"
                    )
        return continue_training


class TelemetryCallback(BaseCallback):
    """Registra métricas físicas extras (taxa upright, torque e etapa) no logger do SB3."""

    def __init__(self, log_freq: int = 2048) -> None:
        super().__init__()
        self.log_freq = log_freq
        self._upright_buf: list[float] = []
        self._torque_buf: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "upright" in info:
                self._upright_buf.append(1.0 if info["upright"] else 0.0)
            if "torque_utilization" in info:
                self._torque_buf.append(float(info["torque_utilization"]))

        if self.n_calls % self.log_freq == 0 and self._upright_buf:
            self.logger.record("physics/upright_ratio", float(np.mean(self._upright_buf)))
            self.logger.record("physics/mean_torque_util", float(np.mean(self._torque_buf)))
            self._upright_buf.clear()
            self._torque_buf.clear()
        return True


class TimeBudgetCallback(BaseCallback):
    """Encerra o treinamento graciosamente ao atingir o orçamento de tempo."""

    def __init__(self, max_minutes: float, verbose: int = 1) -> None:
        super().__init__(verbose=verbose)
        self.max_seconds = max_minutes * 60.0
        self._start_time = 0.0

    def _on_training_start(self) -> None:
        self._start_time = time.monotonic()

    def _on_step(self) -> bool:
        elapsed = time.monotonic() - self._start_time
        if elapsed >= self.max_seconds:
            if self.verbose > 0:
                print(
                    f"\n[TimeBudget] Tempo limite de {self.max_seconds / 60.0:.1f} min "
                    f"atingido ({self.num_timesteps} passos). Salvando e finalizando..."
                )
            return False
        return True


def train_ppo(
    *,
    output_dir: str | Path = "runs/ppo_cartpole",
    total_timesteps: int = 600_000,
    n_envs: int = 8,
    seed: int = 42,
    drive_mode: DriveMode = DriveMode.BELT_PULLEY,
    use_curriculum: bool = True,
    initial_stage: int = 0,
    use_pbrs: bool = True,
    learning_rate: float = 2.5e-4,
    ent_coef: float = 0.001,
    eval_freq_steps: int = 32_768,
    n_eval_episodes: int = 5,
    max_minutes: float | None = None,
    resume_from: str | Path | None = None,
    device: str = "cpu",
    verbose: int = 1,
) -> Path:
    """Executa o treinamento completo de PPO e retorna o caminho do melhor modelo."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_cfg = SimConfig()
    train_cfg.physics.drive_mode = drive_mode
    train_cfg.reward.use_pbrs = use_pbrs
    train_cfg.eval_mode = False
    train_cfg.randomize_disturbances = True

    curriculum = CurriculumManager(
        stages=DEFAULT_STAGES,
        initial_stage=initial_stage if use_curriculum else len(DEFAULT_STAGES) - 1,
    )
    train_cfg.disturbance = dataclasses.replace(curriculum.current_stage.disturbance)

    eval_cfg = SimConfig(
        physics=dataclasses.replace(train_cfg.physics),
        sensor=dataclasses.replace(train_cfg.sensor),
        disturbance=dataclasses.replace(curriculum.current_stage.disturbance),
        reward=dataclasses.replace(train_cfg.reward),
        max_episode_steps=train_cfg.max_episode_steps,
        eval_mode=True,
        randomize_disturbances=True,
    )

    # Salva configuração utilizada para reprodutibilidade e carga no player/avaliador
    meta = {
        "drive_mode": drive_mode.value,
        "use_pbrs": use_pbrs,
        "gamma": train_cfg.reward.gamma,
        "frame_stack": train_cfg.sensor.frame_stack,
        "control_dt": train_cfg.physics.control_dt,
        "rail_half_length": train_cfg.physics.rail_half_length,
        "rod_length": train_cfg.physics.rod_length,
        "bob_distance": train_cfg.physics.bob_distance,
        "use_curriculum": use_curriculum,
    }
    (out_path / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    resume_path = Path(resume_from) if resume_from is not None else None
    raw_train_fns = [_make_single_env(train_cfg, seed=seed, rank=i) for i in range(n_envs)]
    raw_train_vec: VecEnv = SubprocVecEnv(raw_train_fns) if n_envs > 1 else DummyVecEnv(raw_train_fns)

    vec_norm_candidate = resume_path.parent / "vec_normalize.pkl" if resume_path is not None else None
    if vec_norm_candidate is not None and vec_norm_candidate.exists():
        train_env: VecEnv = VecNormalize.load(str(vec_norm_candidate), raw_train_vec)
        assert isinstance(train_env, VecNormalize)
        train_env.training = True
        train_env.norm_obs = False
        train_env.norm_reward = True
    else:
        train_env = VecNormalize(
            raw_train_vec,
            training=True,
            norm_obs=False,
            norm_reward=True,
            clip_reward=10.0,
            gamma=train_cfg.reward.gamma,
        )

    eval_env = build_vec_env(eval_cfg, n_envs=1, seed=seed + 1000, normalize_reward=False, training=False)

    n_steps_per_env = max(256, 1024 if total_timesteps >= 50_000 else 256)
    batch_size = min(256, n_steps_per_env * n_envs)

    if resume_path is not None:
        model = PPO.load(
            resume_path,
            env=train_env,
            device=device,
            custom_objects={"learning_rate": learning_rate, "ent_coef": ent_coef},
        )
        model.learning_rate = learning_rate
        model.ent_coef = ent_coef
        reset_num_timesteps = False
        if verbose > 0:
            print(f"[Resume] Continuando de {resume_path} (timesteps={model.num_timesteps})")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            n_steps=n_steps_per_env,
            batch_size=batch_size,
            n_epochs=10,
            gamma=train_cfg.reward.gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={
                "net_arch": {"pi": [128, 128], "vf": [128, 128]},
                "activation_fn": nn.Tanh,
                "log_std_init": -0.4,
            },
            tensorboard_log=str(out_path / "tb"),
            seed=seed,
            device=device,
            verbose=verbose,
        )
        reset_num_timesteps = True

    eval_freq_per_env = max(1, eval_freq_steps // max(1, n_envs))
    eval_cb = CurriculumEvalCallback(
        eval_env=eval_env,
        curriculum=curriculum,
        output_dir=out_path,
        eval_freq=eval_freq_per_env,
        n_eval_episodes=n_eval_episodes,
        verbose=verbose,
    )
    telem_cb = TelemetryCallback(log_freq=max(512, n_steps_per_env))
    callbacks: list[BaseCallback] = [eval_cb, telem_cb]
    if max_minutes is not None and max_minutes > 0.0:
        callbacks.append(TimeBudgetCallback(max_minutes=max_minutes, verbose=verbose))

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
        )
    finally:
        model.save(out_path / "last_model.zip")
        if isinstance(train_env, VecNormalize):
            train_env.save(str(out_path / "vec_normalize.pkl"))
        train_env.close()
        eval_env.close()

    best_path = out_path / "best_model.zip"
    return best_path if best_path.exists() else (out_path / "last_model.zip")
