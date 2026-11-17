"""Configurações físicas, sensoriais, de distúrbios e de recompensa (PBRS)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class DriveMode(str, Enum):
    """Tipo de transmissão acoplada ao motor de passo NEMA 17."""

    BELT_PULLEY = "belt_pulley"
    LEAD_SCREW = "lead_screw"


@dataclass
class PhysicsConfig:
    """Parâmetros físicos reais da bancada (unidades SI)."""

    g: float = 9.81

    # Trilho linear (metade do curso útil disponível para o carrinho)
    rail_half_length: float = 0.35

    # Carrinho
    cart_mass: float = 0.30
    cart_viscous_friction: float = 0.25

    # Pêndulo físico rígido (haste uniforme + peso posicionável)
    rod_length: float = 0.32
    rod_mass: float = 0.035
    rod_diameter: float = 0.008
    bob_mass: float = 0.085
    bob_distance: float = 0.22
    bob_radius: float = 0.022

    # Atrito do mancal / eixo do encoder
    pivot_viscous_friction: float = 0.0008
    pivot_coulomb_friction: float = 0.0018
    coulomb_smoothing_omega: float = 0.25

    # Acionamento por motor de passo NEMA 17
    drive_mode: DriveMode = DriveMode.BELT_PULLEY
    motor_rotor_inertia: float = 5.4e-5
    motor_holding_torque: float = 0.45
    motor_corner_rpm: float = 400.0
    motor_cutoff_rpm: float = 2200.0

    # Parâmetros de polia GT2 20 dentes (passo 2 mm -> circunferência 40 mm)
    pulley_radius: float = 0.040 / (2.0 * math.pi)
    pulley_efficiency: float = 0.90
    belt_max_velocity: float = 1.25
    belt_max_acceleration: float = 8.5

    # Parâmetros de barra roscada TR8x8 (passo 8 mm/rev)
    lead_screw_pitch: float = 0.008
    lead_screw_efficiency: float = 0.55
    screw_max_velocity: float = 0.32
    screw_max_acceleration: float = 4.5

    # Imperfeições mecânicas do mundo real
    backlash_m: float = 0.0004
    enforce_torque_limit: bool = True
    simulate_step_loss: bool = False

    # Aerodinâmica
    air_density: float = 1.225
    rod_cd: float = 1.15
    bob_cd: float = 0.95

    # Integração numérica e taxa de controle
    control_dt: float = 0.02
    physics_substeps: int = 10

    @property
    def pendulum_mass(self) -> float:
        return self.rod_mass + self.bob_mass

    @property
    def com_distance(self) -> float:
        """Distância do pivô ao centro de massa combinado do pêndulo."""
        bob_d = min(max(self.bob_distance, 0.0), self.rod_length)
        return (self.rod_mass * (0.5 * self.rod_length) + self.bob_mass * bob_d) / self.pendulum_mass

    @property
    def inertia_pivot(self) -> float:
        """Momento de inércia total em relação ao eixo do pivô (eixo paralelo)."""
        bob_d = min(max(self.bob_distance, 0.0), self.rod_length)
        i_rod = (1.0 / 3.0) * self.rod_mass * (self.rod_length**2)
        i_bob = self.bob_mass * (bob_d**2) + 0.5 * self.bob_mass * (self.bob_radius**2)
        return i_rod + i_bob

    @property
    def effective_radius(self) -> float:
        """Raio equivalente (m/rad do eixo do motor) da transmissão ativa."""
        if self.drive_mode == DriveMode.BELT_PULLEY:
            return self.pulley_radius
        return self.lead_screw_pitch / (2.0 * math.pi)

    @property
    def drive_efficiency(self) -> float:
        if self.drive_mode == DriveMode.BELT_PULLEY:
            return self.pulley_efficiency
        return self.lead_screw_efficiency

    @property
    def max_velocity(self) -> float:
        if self.drive_mode == DriveMode.BELT_PULLEY:
            return self.belt_max_velocity
        return self.screw_max_velocity

    @property
    def max_acceleration(self) -> float:
        if self.drive_mode == DriveMode.BELT_PULLEY:
            return self.belt_max_acceleration
        return self.screw_max_acceleration

    @property
    def top_potential_energy(self) -> float:
        """Energia necessária para erguer o pêndulo do repouso embaixo até o repouso no topo."""
        return 2.0 * self.pendulum_mass * self.g * self.com_distance

    def pullout_torque(self, linear_speed: float) -> float:
        """Curva típica de torque pull-out do NEMA 17 em função da velocidade linear."""
        motor_omega = abs(linear_speed) / max(self.effective_radius, 1e-6)
        rpm = motor_omega * 60.0 / (2.0 * math.pi)
        if rpm <= self.motor_corner_rpm:
            # Torque próximo ao de retenção (fator dinâmico ~0.82 em baixa rotação)
            return 0.82 * self.motor_holding_torque
        if rpm >= self.motor_cutoff_rpm:
            return 0.05 * self.motor_holding_torque
        frac = (rpm - self.motor_corner_rpm) / (self.motor_cutoff_rpm - self.motor_corner_rpm)
        # Decaimento suave por impedância indutiva + back-EMF
        return max(0.05, 0.82 * (1.0 - 0.85 * (frac**0.85))) * self.motor_holding_torque


@dataclass
class SensorConfig:
    """Parâmetros de odometria do stepper, encoder rotativo e latência."""

    encoder_ppr: int = 600
    quadrature_multiplier: int = 4
    quantize_encoder: bool = True
    omega_ema_alpha: float = 0.70
    angle_noise_std: float = 0.001
    odom_bias_std: float = 0.0
    action_delay_steps: int = 1
    frame_stack: int = 12
    omega_obs_scale: float = 15.0

    @property
    def encoder_counts_per_rev(self) -> int:
        return self.encoder_ppr * self.quadrature_multiplier


@dataclass
class DisturbanceConfig:
    """Inclinação da mesa, vento, rajadas, petelecos e randomização física."""

    table_tilt_deg: float = 0.0
    tilt_range_deg: tuple[float, float] = (0.0, 0.0)

    wind_mean_mps: float = 0.0
    wind_mean_range_mps: tuple[float, float] = (0.0, 0.0)
    wind_gust_std_mps: float = 0.0
    wind_gust_tau_s: float = 0.8

    flick_prob_per_sec: float = 0.0
    flick_impulse_range: tuple[float, float] = (0.04, 0.12)
    flick_duration_s: float = 0.04

    param_randomization_pct: float = 0.0


@dataclass
class RewardConfig:
    """Recompensa base minimalista + PBRS estritamente conforme Ng et al. (1999)."""

    gamma: float = 0.99
    w_height: float = 1.0
    w_upright: float = 0.5
    upright_sigma_rad: float = 0.35
    w_center: float = 0.20
    w_action: float = 0.008
    w_action_diff: float = 0.015
    wall_penalty: float = 25.0

    use_pbrs: bool = True
    w_potential: float = 3.0
    potential_height_weight: float = 0.7
    potential_energy_weight: float = 0.3


@dataclass
class SimConfig:
    """Configuração agregada completa do ambiente."""

    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    max_episode_steps: int = 1000
    eval_mode: bool = False
    randomize_disturbances: bool = True
