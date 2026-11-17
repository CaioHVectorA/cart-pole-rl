"""Gerenciador de Curriculum Learning em 4 estágios (Plano -> Inclinado -> Vento -> Caos)."""

from __future__ import annotations

from dataclasses import dataclass

from cartpole_sim.config import DisturbanceConfig


@dataclass(frozen=True)
class CurriculumStage:
    """Definição de um estágio de dificuldade do curriculum."""

    stage_id: int
    name: str
    disturbance: DisturbanceConfig
    promote_mean_reward: float
    min_steps_in_stage: int


DEFAULT_STAGES: tuple[CurriculumStage, ...] = (
    CurriculumStage(
        stage_id=0,
        name="Estágio 0 - Mesa plana limpa (swing-up + centralização)",
        disturbance=DisturbanceConfig(
            tilt_range_deg=(0.0, 0.0),
            wind_mean_range_mps=(0.0, 0.0),
            wind_gust_std_mps=0.0,
            flick_prob_per_sec=0.0,
            param_randomization_pct=0.0,
        ),
        promote_mean_reward=1050.0,
        min_steps_in_stage=80_000,
    ),
    CurriculumStage(
        stage_id=1,
        name="Estágio 1 - Mesa inclinada (±7°) + randomização leve",
        disturbance=DisturbanceConfig(
            tilt_range_deg=(-7.0, 7.0),
            wind_mean_range_mps=(0.0, 0.0),
            wind_gust_std_mps=0.0,
            flick_prob_per_sec=0.0,
            param_randomization_pct=0.05,
        ),
        promote_mean_reward=980.0,
        min_steps_in_stage=100_000,
    ),
    CurriculumStage(
        stage_id=2,
        name="Estágio 2 - Mesa inclinada (±10°) + vento constante/rajada leve",
        disturbance=DisturbanceConfig(
            tilt_range_deg=(-10.0, 10.0),
            wind_mean_range_mps=(-4.0, 4.0),
            wind_gust_std_mps=0.8,
            flick_prob_per_sec=0.0,
            param_randomization_pct=0.08,
        ),
        promote_mean_reward=900.0,
        min_steps_in_stage=120_000,
    ),
    CurriculumStage(
        stage_id=3,
        name="Estágio 3 - Caos completo (±12° + vento forte + rajadas + petelecos)",
        disturbance=DisturbanceConfig(
            tilt_range_deg=(-12.0, 12.0),
            wind_mean_range_mps=(-5.5, 5.5),
            wind_gust_std_mps=1.8,
            flick_prob_per_sec=0.25,
            flick_impulse_range=(0.04, 0.12),
            flick_duration_s=0.04,
            param_randomization_pct=0.10,
        ),
        promote_mean_reward=float("inf"),
        min_steps_in_stage=0,
    ),
)


class CurriculumManager:
    """Controla a progressão adaptativa de estágios durante o treino."""

    def __init__(
        self,
        stages: tuple[CurriculumStage, ...] = DEFAULT_STAGES,
        initial_stage: int = 0,
    ) -> None:
        if not stages:
            raise ValueError("stages não pode ser vazio.")
        self.stages = stages
        self.current_idx = max(0, min(initial_stage, len(stages) - 1))
        self.steps_in_stage = 0

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.current_idx]

    @property
    def is_final_stage(self) -> bool:
        return self.current_idx >= len(self.stages) - 1

    def maybe_advance(self, mean_eval_reward: float, total_steps_delta: int) -> bool:
        """Avança para o próximo estágio se atingir o critério mínimo de passos e recompensa."""
        self.steps_in_stage += max(0, total_steps_delta)
        if self.is_final_stage:
            return False
        stage = self.current_stage
        if (
            self.steps_in_stage >= stage.min_steps_in_stage
            and mean_eval_reward >= stage.promote_mean_reward
        ):
            self.current_idx += 1
            self.steps_in_stage = 0
            return True
        return False
