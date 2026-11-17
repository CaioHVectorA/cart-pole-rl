"""Estúdio interativo Pygame para testar física, IA (PPO) e distúrbios em tempo real."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from cartpole_sim.config import DisturbanceConfig, DriveMode, SimConfig
from cartpole_sim.env import CartPoleStepperEnv
from cartpole_sim.renderer import PygameStudioRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estúdio Interativo Cart-Pole Stepper")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/ppo_cartpole/best_model.zip",
        help="Caminho para modelo PPO (.zip). Se não existir, inicia em modo Manual.",
    )
    parser.add_argument(
        "--drive",
        type=str,
        choices=[m.value for m in DriveMode],
        default=DriveMode.BELT_PULLEY.value,
    )
    parser.add_argument("--start-upright", action="store_true", help="Inicia já equilibrado no topo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = None
    model_path = Path(args.model)
    if model_path.exists():
        from stable_baselines3 import PPO

        model = PPO.load(model_path, device="cpu")
        print(f"Modelo carregado: {model_path}")
    else:
        print(f"Modelo não encontrado em '{model_path}'. Iniciando em modo MANUAL (use A/D ou setas).")

    cfg = SimConfig(
        disturbance=DisturbanceConfig(
            table_tilt_deg=0.0,
            tilt_range_deg=(0.0, 0.0),
            wind_mean_mps=0.0,
            wind_mean_range_mps=(0.0, 0.0),
            wind_gust_std_mps=0.0,
            flick_prob_per_sec=0.0,
            param_randomization_pct=0.0,
        ),
        max_episode_steps=1_000_000,
        eval_mode=True,
        randomize_disturbances=True,
    )
    cfg.physics.drive_mode = DriveMode(args.drive)

    env = CartPoleStepperEnv(config=cfg, render_mode=None)
    renderer = PygameStudioRenderer(mode="human")

    init_opt = {"init_mode": "upright" if args.start_upright else "hanging"}
    obs, info = env.reset(seed=42, options=init_opt)
    renderer.sync_sliders_from_env(env)
    renderer.record_step(info)

    ai_active = model is not None
    paused = False
    last_reset_mode = "upright" if args.start_upright else "hanging"
    wall_alert_timer = 0.0

    try:
        while True:
            actions = renderer.poll_events(env, ai_active=ai_active, paused=paused)
            if actions.quit_requested:
                break

            if actions.toggle_mode and model is not None:
                ai_active = not ai_active
            if actions.toggle_pause:
                paused = not paused
            if actions.toggle_drive:
                new_mode = (
                    DriveMode.LEAD_SCREW
                    if env.physics.cfg.drive_mode == DriveMode.BELT_PULLEY
                    else DriveMode.BELT_PULLEY
                )
                new_phys = dataclasses.replace(env.cfg.physics, drive_mode=new_mode)
                env.set_physics_config(new_phys)

            if actions.reset_hanging or actions.reset_upright:
                last_reset_mode = "upright" if actions.reset_upright else "hanging"
                obs, info = env.reset(options={"init_mode": last_reset_mode})
                renderer.clear_history()
                renderer.record_step(info)

            if abs(actions.flick_impulse) > 0.0:
                env.physics.apply_flick(actions.flick_impulse)

            if not paused:
                if ai_active and model is not None:
                    act, _ = model.predict(obs, deterministic=True)
                else:
                    act = np.array([actions.manual_action], dtype=np.float32)

                obs, _rew, terminated, truncated, info = env.step(act)
                renderer.record_step(info)

                if terminated or truncated:
                    wall_alert_timer = 1.2 if terminated else 0.0
                    obs, info = env.reset(options={"init_mode": last_reset_mode})
                    renderer.clear_history()
                    renderer.record_step(info)

            if wall_alert_timer > 0.0:
                wall_alert_timer = max(0.0, wall_alert_timer - env.physics.cfg.control_dt)

            status = "BATEU NA BORDA! (reset automático)" if wall_alert_timer > 0.0 else ""
            if model is None:
                status = (status + " | Sem modelo carregado -> Modo Manual").strip(" |")
            renderer.render_env(env, extra_status=status, paused=paused)
    finally:
        renderer.close()
        env.close()


if __name__ == "__main__":
    main()
