"""Script CLI para treinamento do controlador PPO + PBRS."""

from __future__ import annotations

import argparse

from cartpole_sim.config import DriveMode
from cartpole_sim.train import train_ppo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina agente PPO para Cart-Pole com Motor de Passo")
    parser.add_argument("--output-dir", type=str, default="runs/ppo_cartpole")
    parser.add_argument("--total-timesteps", type=int, default=600_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--drive",
        type=str,
        choices=[m.value for m in DriveMode],
        default=DriveMode.BELT_PULLEY.value,
        help="Transmissão: belt_pulley (polia GT2) ou lead_screw (barra roscada TR8x8)",
    )
    parser.add_argument("--no-curriculum", action="store_true", help="Desativa curriculum (inicia já no Estágio 3)")
    parser.add_argument("--initial-stage", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--no-pbrs", action="store_true", help="Desativa shaping PBRS (para ablação)")
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--eval-freq", type=int, default=32_768)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--max-minutes", type=float, default=None, help="Orçamento máximo de tempo em minutos")
    parser.add_argument("--resume", type=str, default=None, help="Caminho para checkpoint .zip para continuar treino")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best_model = train_ppo(
        output_dir=args.output_dir,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        drive_mode=DriveMode(args.drive),
        use_curriculum=not args.no_curriculum,
        initial_stage=args.initial_stage,
        use_pbrs=not args.no_pbrs,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
        eval_freq_steps=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        max_minutes=args.max_minutes,
        resume_from=args.resume,
        verbose=1,
    )
    print(f"\nTreino concluído! Melhor modelo salvo em: {best_model}")


if __name__ == "__main__":
    main()
