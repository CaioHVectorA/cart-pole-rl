"""Testes de integração Gymnasium, Curriculum, renderização headless e smoke test PPO."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO

from cartpole_sim.config import DriveMode, SimConfig
from cartpole_sim.curriculum import CurriculumManager
from cartpole_sim.env import CartPoleStepperEnv
from cartpole_sim.train import train_ppo


def test_gymnasium_check_env() -> None:
    """Ambiente deve passar integralmente no verificador oficial do Gymnasium."""
    env = CartPoleStepperEnv(config=SimConfig())
    check_env(env, skip_render_check=True)
    env.close()


def test_eval_mode_reset_upright_vs_hanging() -> None:
    """Reset em eval_mode deve respeitar init_mode='upright' vs 'hanging'."""
    env = CartPoleStepperEnv(config=SimConfig(eval_mode=True))
    _, info_up = env.reset(seed=0, options={"init_mode": "upright"})
    assert abs(info_up["theta_true"]) < 1e-6

    _, info_hang = env.reset(seed=0, options={"init_mode": "hanging"})
    assert abs(abs(info_hang["theta_true"]) - np.pi) < 1e-6
    env.close()


def test_curriculum_progression_rules() -> None:
    """Curriculum só promove após cumprir min_steps_in_stage e promote_mean_reward."""
    cm = CurriculumManager()
    assert cm.current_idx == 0
    thresh = cm.current_stage.promote_mean_reward
    min_steps = cm.current_stage.min_steps_in_stage

    # Recompensa suficiente, mas passos insuficientes
    assert not cm.maybe_advance(mean_eval_reward=thresh + 50.0, total_steps_delta=min_steps // 4)
    assert cm.current_idx == 0

    # Passos suficientes, mas recompensa abaixo do limiar
    assert not cm.maybe_advance(mean_eval_reward=thresh - 200.0, total_steps_delta=min_steps)
    assert cm.current_idx == 0

    # Ambos cumpridos -> promove para Estágio 1
    assert cm.maybe_advance(mean_eval_reward=thresh + 50.0, total_steps_delta=5_000)
    assert cm.current_idx == 1


def test_pygame_headless_rgb_array(monkeypatch) -> None:
    """Renderização em modo rgb_array deve gerar frame válido sem janela gráfica."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    env = CartPoleStepperEnv(config=SimConfig(), render_mode="rgb_array")
    env.reset(seed=0)
    env.step(np.array([0.3], dtype=np.float32))
    frame = env.render()
    env.close()

    assert frame is not None
    assert frame.shape == (760, 1280, 3)
    assert frame.dtype == np.uint8
    assert frame.mean() > 0.0


def test_ppo_smoke_train_and_reload(tmp_path: Path) -> None:
    """Smoke test completo: treina PPO por alguns passos, salva, retoma e avalia."""
    out_dir = tmp_path / "smoke_run"
    best_path = train_ppo(
        output_dir=out_dir,
        total_timesteps=2048,
        n_envs=2,
        seed=123,
        drive_mode=DriveMode.BELT_PULLEY,
        use_curriculum=True,
        eval_freq_steps=1024,
        n_eval_episodes=2,
        verbose=0,
    )

    assert best_path.exists()
    assert (out_dir / "last_model.zip").exists()
    assert (out_dir / "run_config.json").exists()

    # Retoma treino com orçamento curto de tempo
    resumed_best = train_ppo(
        output_dir=out_dir,
        total_timesteps=2048,
        n_envs=2,
        seed=123,
        drive_mode=DriveMode.BELT_PULLEY,
        initial_stage=3,
        eval_freq_steps=1024,
        n_eval_episodes=2,
        max_minutes=0.25,
        resume_from=out_dir / "last_model.zip",
        verbose=0,
    )
    assert resumed_best.exists()

    model = PPO.load(resumed_best, device="cpu")
    eval_env = CartPoleStepperEnv(config=SimConfig(eval_mode=True))
    obs, _ = eval_env.reset(seed=99)
    for _ in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = eval_env.step(action)
        assert np.isfinite(reward)
        if term or trunc:
            break
    eval_env.close()
