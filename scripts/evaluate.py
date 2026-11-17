"""Suíte de avaliação comparativa em múltiplos cenários físicos adversos."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from cartpole_sim.config import DisturbanceConfig, DriveMode, SimConfig
from cartpole_sim.env import CartPoleStepperEnv


@dataclass
class EvalScenario:
    name: str
    disturbance: DisturbanceConfig
    drive_mode: DriveMode = DriveMode.BELT_PULLEY
    periodic_flick: float = 0.0  # aplica um peteleco aos 6s e aos 12s se > 0


def build_scenarios(drive_mode: DriveMode) -> list[EvalScenario]:
    return [
        EvalScenario(
            name="1. Plano limpo",
            drive_mode=drive_mode,
            disturbance=DisturbanceConfig(),
        ),
        EvalScenario(
            name="2. Mesa inclinada +10°",
            drive_mode=drive_mode,
            disturbance=DisturbanceConfig(table_tilt_deg=10.0),
        ),
        EvalScenario(
            name="3. Mesa inclinada -10°",
            drive_mode=drive_mode,
            disturbance=DisturbanceConfig(table_tilt_deg=-10.0),
        ),
        EvalScenario(
            name="4. Vento constante +4.5 m/s",
            drive_mode=drive_mode,
            disturbance=DisturbanceConfig(wind_mean_mps=4.5),
        ),
        EvalScenario(
            name="5. Rampa -8° + Vento +4 m/s + Rajadas",
            drive_mode=drive_mode,
            disturbance=DisturbanceConfig(
                table_tilt_deg=-8.0,
                wind_mean_mps=4.0,
                wind_gust_std_mps=1.5,
            ),
        ),
        EvalScenario(
            name="6. Caos + Petelecos determinísticos (6s e 12s)",
            drive_mode=drive_mode,
            disturbance=DisturbanceConfig(
                table_tilt_deg=6.0,
                wind_mean_mps=-3.0,
                wind_gust_std_mps=1.2,
            ),
            periodic_flick=0.09,
        ),
    ]


def evaluate_scenario(
    model: PPO,
    scenario: EvalScenario,
    n_episodes: int = 5,
    seed: int = 100,
) -> dict[str, float]:
    cfg = SimConfig(
        disturbance=scenario.disturbance,
        max_episode_steps=1000,
        eval_mode=True,
        randomize_disturbances=(scenario.disturbance.wind_gust_std_mps > 0.0),
    )
    cfg.physics.drive_mode = scenario.drive_mode
    env = CartPoleStepperEnv(config=cfg)

    rewards = []
    upright_ratios = []
    mean_abs_x = []
    swingup_times = []
    peak_torques = []
    survived = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep, options={"init_mode": "hanging"})
        ep_rew = 0.0
        upright_steps = 0
        steps = 0
        first_upright_s = float("nan")
        x_abs_list = []
        ep_peak_t = 0.0
        done = False

        while not done:
            if scenario.periodic_flick > 0.0 and steps in (300, 600):
                sign = 1.0 if steps == 300 else -1.0
                env.physics.apply_flick(sign * scenario.periodic_flick)

            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_rew += rew
            steps += 1
            x_abs_list.append(abs(info["x_true"]))
            ep_peak_t = max(ep_peak_t, info["torque_utilization"])
            if info["upright"]:
                upright_steps += 1
                if math.isnan(first_upright_s):
                    first_upright_s = info["time_s"]
            if truncated and not terminated:
                survived += 1

        rewards.append(ep_rew)
        upright_ratios.append(upright_steps / max(1, steps))
        mean_abs_x.append(float(np.mean(x_abs_list)))
        peak_torques.append(ep_peak_t)
        if not math.isnan(first_upright_s):
            swingup_times.append(first_upright_s)

    env.close()
    return {
        "survival_pct": 100.0 * survived / n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "upright_pct": 100.0 * float(np.mean(upright_ratios)),
        "mean_abs_x_cm": 100.0 * float(np.mean(mean_abs_x)),
        "swingup_s": float(np.mean(swingup_times)) if swingup_times else float("nan"),
        "peak_torque_util": float(np.mean(peak_torques)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark de robustez do controlador PPO")
    parser.add_argument("--model", type=str, default="runs/ppo_cartpole/best_model.zip")
    parser.add_argument(
        "--drive",
        type=str,
        choices=[m.value for m in DriveMode],
        default=DriveMode.BELT_PULLEY.value,
    )
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    model = PPO.load(model_path, device="cpu")
    scenarios = build_scenarios(DriveMode(args.drive))

    print(f"\nAvaliando modelo: {model_path} | Transmissão: {args.drive}\n")
    header = (
        f"{'Cenário':<46} | {'Sobrev%':>7} | {'Reward':>7} | {'Topo%':>6} | "
        f"{'|x| cm':>6} | {'Swing(s)':>8} | {'Torque%':>7}"
    )
    print(header)
    print("-" * len(header))

    for sc in scenarios:
        res = evaluate_scenario(model, sc, n_episodes=args.episodes)
        sw = f"{res['swingup_s']:8.2f}" if not math.isnan(res["swingup_s"]) else "     N/A"
        print(
            f"{sc.name:<46} | {res['survival_pct']:6.1f}% | {res['mean_reward']:7.1f} | "
            f"{res['upright_pct']:5.1f}% | {res['mean_abs_x_cm']:6.2f} | {sw} | "
            f"{100.0 * res['peak_torque_util']:6.1f}%"
        )


if __name__ == "__main__":
    main()
