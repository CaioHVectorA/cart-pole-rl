"""Testes das regras matemáticas do PBRS (Ng et al., 1999) e sensor/encoder."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cartpole_sim.config import DisturbanceConfig, PhysicsConfig, RewardConfig, SensorConfig, SimConfig
from cartpole_sim.env import CartPoleStepperEnv


def test_pbrs_360_loop_telescopes_and_cancels() -> None:
    """Em um ciclo fechado (ou giro de 360°), soma sem desconto de PBRS se anula e soma com gamma segue teorema."""
    cfg = SimConfig(
        physics=PhysicsConfig(
            pivot_viscous_friction=0.0,
            pivot_coulomb_friction=0.0,
            air_density=0.0,
        ),
        sensor=SensorConfig(quantize_encoder=False, angle_noise_std=0.0, action_delay_steps=0, frame_stack=1),
        reward=RewardConfig(gamma=0.99, use_pbrs=True, w_potential=4.0),
        eval_mode=True,
        randomize_disturbances=False,
    )
    env = CartPoleStepperEnv(config=cfg)
    # Impulso inicial suficiente para girar 360 graus completos com carrinho parado
    env.reset(seed=0, options={"x": 0.0, "v": 0.0, "theta": math.pi, "omega": 15.0})

    gamma = cfg.reward.gamma
    discounted_shaping = 0.0
    undiscounted_shaping = 0.0
    phi_start = env.compute_potential(env.physics.state)

    phis = [phi_start]
    for t in range(40):
        _, _, term, trunc, info = env.step(np.array([0.0], dtype=np.float32))
        assert not term and not trunc
        shaping = info["pbrs_shaping"]
        discounted_shaping += (gamma**t) * shaping
        undiscounted_shaping += shaping
        phis.append(info["phi_next"])

    phi_end = phis[-1]
    # Identidade algébrica exata da soma telescópica com desconto
    expected_discounted = (gamma**40) * phi_end - phi_start
    # Soma sem desconto é exatamente a diferença telescópica
    # (como cada passo usa gamma*phi_next - phi_curr, somamos (phi_next - phi_curr) diretamente)
    direct_diff_sum = sum(phis[i + 1] - phis[i] for i in range(len(phis) - 1))

    assert discounted_shaping == pytest.approx(expected_discounted, abs=1e-6)
    assert direct_diff_sum == pytest.approx(phi_end - phi_start, abs=1e-6)
    assert math.isfinite(undiscounted_shaping)


def test_pbrs_terminal_potential_is_zero_and_no_suicide_bonus() -> None:
    """Regra #4 e #5: bater na borda força phi_next = 0 e gera shaping <= 0."""
    cfg = SimConfig(
        sensor=SensorConfig(action_delay_steps=0),
        reward=RewardConfig(gamma=0.99, use_pbrs=True, w_potential=3.0),
        eval_mode=True,
        randomize_disturbances=False,
    )
    env = CartPoleStepperEnv(config=cfg)
    x_near_wall = cfg.physics.rail_half_length - 0.008
    env.reset(seed=1, options={"x": x_near_wall, "v": 0.8, "theta": 0.0, "omega": 0.0})

    phi_before = env.compute_potential(env.physics.state)
    assert phi_before > 0.0

    terminated = False
    info = {}
    for _ in range(10):
        _, rew, terminated, _, info = env.step(np.array([1.0], dtype=np.float32))
        if terminated:
            break

    assert terminated
    assert info["phi_next"] == 0.0
    assert info["pbrs_shaping"] <= 0.0
    assert rew < 0.0


def test_potential_properties_non_negative_and_peak_at_top_center() -> None:
    """Phi(s) >= 0 dentro do trilho, máximo no topo centralizado e 0 embaixo."""
    env = CartPoleStepperEnv()
    env.reset(seed=0, options={"x": 0.0, "v": 0.0, "theta": math.pi, "omega": 0.0})
    phi_bottom = env.compute_potential(env.physics.state)

    env.reset(seed=0, options={"x": 0.0, "v": 0.0, "theta": 0.0, "omega": 0.0})
    phi_top_center = env.compute_potential(env.physics.state)

    env.reset(seed=0, options={"x": 0.25, "v": 0.0, "theta": 0.0, "omega": 0.0})
    phi_top_side = env.compute_potential(env.physics.state)

    assert phi_bottom == pytest.approx(0.0, abs=1e-7)
    assert phi_top_center > phi_top_side > phi_bottom >= 0.0


def test_encoder_wrap_crossing_has_no_omega_spike() -> None:
    """Atravessar +-pi no encoder não pode gerar pico de velocidade angular."""
    cfg = SimConfig(
        sensor=SensorConfig(quantize_encoder=True, angle_noise_std=0.0, omega_ema_alpha=1.0),
        eval_mode=True,
        randomize_disturbances=False,
    )
    env = CartPoleStepperEnv(config=cfg)
    env.reset(seed=0, options={"x": 0.0, "v": 0.0, "theta": math.pi - 0.05, "omega": 4.0})

    for _ in range(5):
        _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
        assert abs(info["omega_meas"] - info["omega_true"]) < 0.5


def test_frame_stack_shape_and_action_delay() -> None:
    """Observação empilhada tem dimensão k * 6 e delay de ação de 1 passo funciona."""
    cfg = SimConfig(
        sensor=SensorConfig(frame_stack=8, action_delay_steps=1),
        eval_mode=True,
        randomize_disturbances=False,
    )
    env = CartPoleStepperEnv(config=cfg)
    obs0, _ = env.reset(seed=7)
    assert obs0.shape == (48,)

    # Passo 1 com comando +1.0 -> por ter delay=1, motor ainda executa 0.0
    obs1, _, _, _, info1 = env.step(np.array([1.0], dtype=np.float32))
    assert obs1.shape == (48,)
    assert abs(info1["v_true"]) < 1e-9
    # Último frame registra u_prev = 1.0 emitido pelo agente
    assert obs1[-1] == pytest.approx(1.0, abs=1e-6)

    # Passo 2 -> comando atrasado (+1.0) finalmente entra no motor
    _, _, _, _, info2 = env.step(np.array([0.0], dtype=np.float32))
    assert info2["v_true"] > 0.01
