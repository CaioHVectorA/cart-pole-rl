"""Testes unitários da física fidedigna (conservação, período, rampa, vento, peteleco, folga)."""

from __future__ import annotations

import math

import pytest

from cartpole_sim.config import DisturbanceConfig, DriveMode, PhysicsConfig
from cartpole_sim.physics import CartPolePhysics, wrap_angle


def test_energy_conservation_free_pendulum() -> None:
    """Sem atrito, ar ou movimento do carrinho, energia mecânica deve se conservar no RK4."""
    cfg = PhysicsConfig(
        pivot_viscous_friction=0.0,
        pivot_coulomb_friction=0.0,
        air_density=0.0,
        backlash_m=0.0,
    )
    phys = CartPolePhysics(config=cfg)
    phys.reset(x=0.0, v=0.0, theta=math.pi - 0.9, omega=0.4)
    e0 = phys.mechanical_energy()

    for _ in range(250):  # 5 segundos a 50 Hz (2500 sub-passos RK4)
        phys.step(0.0, randomize_disturbances=False)

    e1 = phys.mechanical_energy()
    assert abs(e1 - e0) < 1e-6


def test_small_oscillation_natural_period() -> None:
    """Pêndulo físico com massa posicionável deve oscilar no período teórico exato."""
    cfg = PhysicsConfig(
        bob_distance=0.16,  # peso no meio da haste de 0.32m
        pivot_viscous_friction=0.0,
        pivot_coulomb_friction=0.0,
        air_density=0.0,
        control_dt=0.005,
        physics_substeps=5,
    )
    phys = CartPolePhysics(config=cfg)
    amp = 0.04  # pequena oscilação em torno de pi
    phys.reset(x=0.0, v=0.0, theta=math.pi + amp, omega=0.0)

    expected_period = 2.0 * math.pi * math.sqrt(
        cfg.inertia_pivot / (cfg.pendulum_mass * cfg.g * cfg.com_distance)
    )

    crossings: list[float] = []
    prev_delta = phys.state.theta - math.pi
    for _ in range(int(3.0 * expected_period / cfg.control_dt)):
        phys.step(0.0, randomize_disturbances=False)
        delta = phys.state.theta - math.pi
        if prev_delta < 0.0 <= delta:
            crossings.append(phys.state.time_s)
        prev_delta = delta

    assert len(crossings) >= 2
    measured_period = crossings[1] - crossings[0]
    assert measured_period == pytest.approx(expected_period, rel=0.01)


def test_tilted_table_encoder_offset_and_true_vertical() -> None:
    """Com mesa inclinada, vertical real (theta=0) segue estável no equilíbrio sem torque grav."""
    cfg = PhysicsConfig(
        pivot_viscous_friction=0.0,
        pivot_coulomb_friction=0.0,
        air_density=0.0,
        backlash_m=0.0,
    )
    dist = DisturbanceConfig(table_tilt_deg=12.0)
    phys = CartPolePhysics(config=cfg, disturbance=dist)
    phys.reset(x=0.0, v=0.0, theta=0.0, omega=0.0)

    assert math.degrees(phys.state.theta_encoder) == pytest.approx(12.0, abs=1e-6)
    phys.step(0.0, randomize_disturbances=False)
    assert abs(phys.state.theta) < 1e-6
    assert abs(phys.state.omega) < 1e-6


def test_wind_pushes_pendulum_in_correct_direction() -> None:
    """Vento positivo (+X) deve empurrar haste erguida no sentido horário (+omega)."""
    cfg = PhysicsConfig(pivot_coulomb_friction=0.0)
    dist = DisturbanceConfig(wind_mean_mps=6.0)
    phys = CartPolePhysics(config=cfg, disturbance=dist)
    phys.reset(x=0.0, v=0.0, theta=0.0, omega=0.0)

    metrics = phys.step(0.0, randomize_disturbances=False)
    assert metrics.aero_torque_nm > 0.0
    assert phys.state.omega > 0.0


def test_flick_impulse_transfers_angular_momentum() -> None:
    """Peteleco de impulso J num pêndulo livre em repouso gera delta_omega ≈ J / I_o."""
    cfg = PhysicsConfig(
        pivot_viscous_friction=0.0,
        pivot_coulomb_friction=0.0,
        air_density=0.0,
    )
    phys = CartPolePhysics(config=cfg)
    phys.reset(x=0.0, v=0.0, theta=math.pi, omega=0.0)

    impulse = 0.05  # N*m*s
    phys.apply_flick(impulse, duration_s=0.04)
    for _ in range(2):  # 2 * 0.02s = 0.04s
        phys.step(0.0, randomize_disturbances=False)

    # Pequeno deslocamento próximo a pi gera leve torque restaurador, logo tolerância ~3%
    expected_omega = impulse / cfg.inertia_pivot
    assert phys.state.omega == pytest.approx(expected_omega, rel=0.03)


def test_drive_modes_and_backlash() -> None:
    """Polia GT2 acelera/atinge velocidade maior que barra roscada; folga cria lag limitado."""
    belt_cfg = PhysicsConfig(drive_mode=DriveMode.BELT_PULLEY, backlash_m=0.002)
    screw_cfg = PhysicsConfig(drive_mode=DriveMode.LEAD_SCREW, backlash_m=0.0)

    belt = CartPolePhysics(config=belt_cfg)
    screw = CartPolePhysics(config=screw_cfg)
    belt.reset()
    screw.reset()

    for _ in range(15):
        belt.step(belt_cfg.max_velocity, randomize_disturbances=False)
        screw.step(screw_cfg.max_velocity, randomize_disturbances=False)

    assert belt.state.v_motor > screw.state.v_motor
    assert belt.state.x_motor - belt.state.x == pytest.approx(0.001, abs=1e-4)

    # Inverte sentido e verifica que folga nunca excede metade da folga total
    for _ in range(10):
        belt.step(-belt_cfg.max_velocity, randomize_disturbances=False)
    assert abs(belt.state.x_motor - belt.state.x) <= 0.001 + 1e-6
    assert abs(wrap_angle(0.0)) == 0.0


def test_dynamic_tilt_and_wind_adaptation() -> None:
    """Inclinação e vento médio devem variar intra-episódio quando habilitados."""
    dist = DisturbanceConfig(
        tilt_range_deg=(-12.0, 12.0),
        tilt_change_prob_per_sec=20.0,
        tilt_slew_deg_per_s=15.0,
        wind_mean_range_mps=(-5.0, 5.0),
        wind_change_prob_per_sec=20.0,
    )
    phys = CartPolePhysics(disturbance=dist, rng=__import__("numpy").random.default_rng(7))
    phys.reset(randomize_disturbances=True)
    alpha0 = phys.state.alpha
    wind0 = phys.state.wind_mean

    alphas = {alpha0}
    winds = {wind0}
    for _ in range(50):
        phys.step(0.0, randomize_disturbances=True)
        alphas.add(round(phys.state.alpha, 6))
        winds.add(round(phys.state.wind_mean, 6))

    assert len(alphas) > 3
    assert len(winds) > 3
    assert abs(math.degrees(phys.state.alpha)) <= 12.0 + 1e-6

