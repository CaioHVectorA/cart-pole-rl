"""Motor de física fidedigna (RK4) para pêndulo invertido com motor de passo.

Modela explicitamente:
- Pêndulo físico rígido (haste uniforme + massa posicionável)
- Trilho inclinável (ângulo alpha) distinguindo vertical real de ângulo no encoder
- Motor de passo (rampa de aceleração, inércia do rotor, curva torque pull-out, folga)
- Atrito viscoso + atrito seco de Coulomb suavizado no mancal
- Vento horizontal com arrasto quadrático relativo (quadratura de Gauss na haste + massa)
- Rajadas (processo de Ornstein-Uhlenbeck) e petelecos de duração finita
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cartpole_sim.config import DisturbanceConfig, PhysicsConfig

# Pontos e pesos de quadratura Gauss-Legendre de 3 pontos em [0, 1]
_GL_NODES = (
    0.5 * (1.0 - math.sqrt(3.0 / 5.0)),
    0.5,
    0.5 * (1.0 + math.sqrt(3.0 / 5.0)),
)
_GL_WEIGHTS = (5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0)


def wrap_angle(angle: float) -> float:
    """Normaliza ângulo para (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class PhysicsState:
    """Estado físico instantâneo completo da bancada."""

    x: float = 0.0
    v: float = 0.0
    x_motor: float = 0.0
    v_motor: float = 0.0
    theta: float = math.pi  # rad, contínuo (0 = vertical para cima, pi = para baixo)
    omega: float = 0.0

    alpha: float = 0.0  # rad, inclinação da mesa (>0 sobe para +x)
    wind_mean: float = 0.0
    wind_gust: float = 0.0

    flick_torque: float = 0.0
    flick_remaining_s: float = 0.0

    time_s: float = 0.0

    @property
    def theta_wrapped(self) -> float:
        return wrap_angle(self.theta)

    @property
    def theta_encoder(self) -> float:
        """Ângulo relativo à normal do carrinho/trilho medido pelo encoder."""
        return self.theta + self.alpha

    @property
    def total_wind(self) -> float:
        return self.wind_mean + self.wind_gust


@dataclass
class StepMetrics:
    """Telemetria detalhada do passo de controle para análise e UI."""

    motor_torque_nm: float
    pullout_torque_nm: float
    torque_utilization: float
    step_lost: bool
    aero_torque_nm: float
    friction_torque_nm: float
    wind_total_mps: float
    active_flick_torque_nm: float
    mechanical_energy_j: float


class CartPolePhysics:
    """Simulador físico desacoplado de renderização e frameworks de RL."""

    def __init__(
        self,
        config: PhysicsConfig | None = None,
        disturbance: DisturbanceConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.base_cfg = config or PhysicsConfig()
        self.cfg = PhysicsConfig(**vars(self.base_cfg))
        self.dist = disturbance or DisturbanceConfig()
        self.rng = rng or np.random.default_rng()
        self.state = PhysicsState()

    def reset(
        self,
        *,
        x: float = 0.0,
        v: float = 0.0,
        theta: float = math.pi,
        omega: float = 0.0,
        randomize_params: bool = False,
        randomize_disturbances: bool = False,
    ) -> PhysicsState:
        """Reinicia o estado físico e aplica randomização opcional de parâmetros/distúrbios."""
        self.cfg = PhysicsConfig(**vars(self.base_cfg))

        pct = self.dist.param_randomization_pct if randomize_params else 0.0
        if pct > 0.0:
            self.cfg.cart_mass *= float(self.rng.uniform(1.0 - pct, 1.0 + pct))
            self.cfg.bob_mass *= float(self.rng.uniform(1.0 - pct, 1.0 + pct))
            self.cfg.pivot_viscous_friction *= float(self.rng.uniform(1.0 - pct, 1.0 + pct))
            self.cfg.pivot_coulomb_friction *= float(self.rng.uniform(1.0 - pct, 1.0 + pct))
            self.cfg.backlash_m *= float(self.rng.uniform(0.5, 1.0 + pct))

        if randomize_disturbances and self.dist.tilt_range_deg != (0.0, 0.0):
            tilt_deg = float(self.rng.uniform(*self.dist.tilt_range_deg))
        else:
            tilt_deg = self.dist.table_tilt_deg

        if randomize_disturbances and self.dist.wind_mean_range_mps != (0.0, 0.0):
            wind_mean = float(self.rng.uniform(*self.dist.wind_mean_range_mps))
        else:
            wind_mean = self.dist.wind_mean_mps

        self.state = PhysicsState(
            x=float(x),
            v=float(v),
            x_motor=float(x),
            v_motor=float(v),
            theta=float(theta),
            omega=float(omega),
            alpha=math.radians(tilt_deg),
            wind_mean=float(wind_mean),
            wind_gust=0.0,
            flick_torque=0.0,
            flick_remaining_s=0.0,
            time_s=0.0,
        )
        return self.state

    def apply_flick(self, impulse_nms: float, duration_s: float | None = None) -> None:
        """Aplica um peteleco externo (impulso angular distribuído em janela curta)."""
        dur = duration_s if duration_s is not None else self.dist.flick_duration_s
        dur = max(dur, self.cfg.control_dt / max(self.cfg.physics_substeps, 1))
        self.state.flick_torque = impulse_nms / dur
        self.state.flick_remaining_s = dur

    def mechanical_energy(self, state: PhysicsState | None = None) -> float:
        """Energia mecânica do pêndulo relativa ao repouso pendurado embaixo (>= 0)."""
        s = state or self.state
        cfg = self.cfg
        # E = 0.5 * I_o * omega^2 + m_p * g * l * (1 + cos(theta))
        # No fundo (theta=pi, omega=0), cos(pi)=-1 -> E = 0.
        # No topo (theta=0, omega=0), cos(0)=1 -> E = 2 * m_p * g * l = top_potential_energy.
        return (
            0.5 * cfg.inertia_pivot * (s.omega**2)
            + cfg.pendulum_mass * cfg.g * cfg.com_distance * (1.0 + math.cos(s.theta))
        )

    def _aero_forces(
        self,
        theta: float,
        omega: float,
        v_cart: float,
        alpha: float,
        wind_mps: float,
    ) -> tuple[float, float]:
        """Retorna (torque_aero_pivo, forca_aero_no_trilho)."""
        cfg = self.cfg
        cos_t = math.cos(theta)
        cos_ta = math.cos(theta + alpha)
        v_base_perp = wind_mps * cos_t - v_cart * cos_ta

        # 1. Haste uniforme (quadratura Gauss-Legendre ao longo de [0, rod_length])
        q_rod = 0.5 * cfg.air_density * cfg.rod_cd * cfg.rod_diameter * cfg.rod_length
        tau_rod = 0.0
        force_rod = 0.0
        for node, w in zip(_GL_NODES, _GL_WEIGHTS):
            s_pos = node * cfg.rod_length
            v_perp = v_base_perp - s_pos * omega
            df = w * q_rod * v_perp * abs(v_perp)
            force_rod += df
            tau_rod += s_pos * df

        # 2. Peso acoplado à haste
        bob_d = min(max(cfg.bob_distance, 0.0), cfg.rod_length)
        bob_area = math.pi * (cfg.bob_radius**2)
        v_perp_bob = v_base_perp - bob_d * omega
        force_bob = 0.5 * cfg.air_density * cfg.bob_cd * bob_area * v_perp_bob * abs(v_perp_bob)
        tau_bob = bob_d * force_bob

        total_tau = tau_rod + tau_bob
        total_rail_force = (force_rod + force_bob) * cos_ta
        return total_tau, total_rail_force

    def _friction_torque(self, omega: float) -> float:
        cfg = self.cfg
        coulomb = cfg.pivot_coulomb_friction * math.tanh(
            omega / max(cfg.coulomb_smoothing_omega, 1e-4)
        )
        return -cfg.pivot_viscous_friction * omega - coulomb

    def _pendulum_alpha_terms(
        self,
        theta: float,
        omega: float,
        v_cart: float,
        alpha: float,
        wind_mps: float,
        flick_torque: float,
    ) -> tuple[float, float, float, float]:
        """Calcula termos da equação de movimento para acoplamento exato com o motor."""
        cfg = self.cfg
        m_p = cfg.pendulum_mass
        ell = cfg.com_distance
        i_o = cfg.inertia_pivot

        tau_fric = self._friction_torque(omega)
        tau_aero, force_aero_rail = self._aero_forces(theta, omega, v_cart, alpha, wind_mps)
        tau_no_accel = m_p * cfg.g * ell * math.sin(theta) + tau_fric + tau_aero + flick_torque
        return tau_no_accel / i_o, tau_aero, tau_fric, force_aero_rail

    def _motor_torque_coeffs(
        self,
        theta: float,
        omega: float,
        v_cart: float,
        alpha: float,
        wind_mps: float,
        flick_torque: float,
    ) -> tuple[float, float]:
        """Retorna (K_a, tau_bias) tais que tau_motor(a_c) = K_a * a_c + tau_bias."""
        cfg = self.cfg
        m_total = cfg.cart_mass + cfg.pendulum_mass
        m_p = cfg.pendulum_mass
        ell = cfg.com_distance
        i_o = cfg.inertia_pivot
        cos_ta = math.cos(theta + alpha)
        sin_ta = math.sin(theta + alpha)

        alpha_no_accel, _, _, f_aero_rail = self._pendulum_alpha_terms(
            theta, omega, v_cart, alpha, wind_mps, flick_torque
        )

        # Inércia linear efetiva após eliminar ddot_theta(a_c)
        m_eff = m_total - ((m_p * ell * cos_ta) ** 2) / i_o
        f_bias = (
            m_p * ell * (alpha_no_accel * cos_ta - (omega**2) * sin_ta)
            + m_total * cfg.g * math.sin(alpha)
            + cfg.cart_viscous_friction * v_cart
            - f_aero_rail
        )

        r_eff = cfg.effective_radius
        eta = cfg.drive_efficiency
        k_a = (cfg.motor_rotor_inertia / r_eff) + (r_eff / eta) * m_eff
        tau_bias = (r_eff / eta) * f_bias
        return k_a, tau_bias

    def _rk4_pendulum_substep(
        self,
        theta: float,
        omega: float,
        v0: float,
        v1: float,
        a_cart: float,
        alpha: float,
        wind_mps: float,
        flick_torque: float,
        dt: float,
    ) -> tuple[float, float]:
        """Integra (theta, omega) em um sub-passo usando Runge-Kutta de 4ª ordem."""
        cfg = self.cfg
        m_p = cfg.pendulum_mass
        ell = cfg.com_distance
        i_o = cfg.inertia_pivot
        v_mid = 0.5 * (v0 + v1)

        def deriv(th: float, om: float, v_c: float) -> float:
            alpha_free, _, _, _ = self._pendulum_alpha_terms(
                th, om, v_c, alpha, wind_mps, flick_torque
            )
            return alpha_free - (m_p * ell * a_cart * math.cos(th + alpha)) / i_o

        k1_t = omega
        k1_o = deriv(theta, omega, v0)

        t2 = theta + 0.5 * dt * k1_t
        o2 = omega + 0.5 * dt * k1_o
        k2_t = o2
        k2_o = deriv(t2, o2, v_mid)

        t3 = theta + 0.5 * dt * k2_t
        o3 = omega + 0.5 * dt * k2_o
        k3_t = o3
        k3_o = deriv(t3, o3, v_mid)

        t4 = theta + dt * k3_t
        o4 = omega + dt * k3_o
        k4_t = o4
        k4_o = deriv(t4, o4, v1)

        next_theta = theta + (dt / 6.0) * (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t)
        next_omega = omega + (dt / 6.0) * (k1_o + 2.0 * k2_o + 2.0 * k3_o + k4_o)
        return next_theta, next_omega

    def step(self, target_velocity_mps: float, randomize_disturbances: bool = True) -> StepMetrics:
        """Avança um passo de controle (`control_dt`) com sub-passos RK4."""
        cfg = self.cfg
        s = self.state
        substeps = max(1, cfg.physics_substeps)
        dt = cfg.control_dt / substeps

        v_target = max(-cfg.max_velocity, min(cfg.max_velocity, float(target_velocity_mps)))

        # Atualiza rajada de vento (Ornstein-Uhlenbeck) e petelecos aleatórios por passo de controle
        if randomize_disturbances and self.dist.wind_gust_std_mps > 0.0:
            tau_g = max(self.dist.wind_gust_tau_s, 1e-3)
            decay = math.exp(-cfg.control_dt / tau_g)
            sigma = self.dist.wind_gust_std_mps * math.sqrt(1.0 - decay * decay)
            s.wind_gust = s.wind_gust * decay + float(self.rng.normal(0.0, sigma))
        elif self.dist.wind_gust_std_mps <= 0.0 and not randomize_disturbances:
            # Permite controle manual na UI quando wind_gust_std_mps == 0
            pass

        if randomize_disturbances and self.dist.flick_prob_per_sec > 0.0 and s.flick_remaining_s <= 0.0:
            p_flick = 1.0 - math.exp(-self.dist.flick_prob_per_sec * cfg.control_dt)
            if self.rng.random() < p_flick:
                mag = float(self.rng.uniform(*self.dist.flick_impulse_range))
                sign = -1.0 if self.rng.random() < 0.5 else 1.0
                self.apply_flick(sign * mag)

        max_util = 0.0
        last_tau_req = 0.0
        last_tau_pullout = cfg.pullout_torque(s.v_motor)
        any_step_lost = False
        half_backlash = 0.5 * max(0.0, cfg.backlash_m)

        for _ in range(substeps):
            # Aceleração desejada pelo perfilador de rampa do motor de passo
            a_des = (v_target - s.v_motor) / dt
            a_des = max(-cfg.max_acceleration, min(cfg.max_acceleration, a_des))

            # Checa limite físico de torque do NEMA 17
            k_a, tau_bias = self._motor_torque_coeffs(
                s.theta, s.omega, s.v, s.alpha, s.total_wind, s.flick_torque
            )
            tau_req = k_a * a_des + tau_bias
            tau_pullout = cfg.pullout_torque(s.v_motor)
            util = abs(tau_req) / max(tau_pullout, 1e-6)
            if util > max_util:
                max_util = util
            last_tau_req = tau_req
            last_tau_pullout = tau_pullout

            a_motor = a_des
            if abs(tau_req) > tau_pullout and cfg.enforce_torque_limit:
                if cfg.simulate_step_loss:
                    any_step_lost = True
                    # Motor perde sincronismo momentaneamente
                    a_motor = 0.0
                    s.v_motor *= 0.85
                else:
                    tau_clamped = max(-tau_pullout, min(tau_pullout, tau_req))
                    a_motor = (tau_clamped - tau_bias) / max(k_a, 1e-9)
                    a_motor = max(-cfg.max_acceleration, min(cfg.max_acceleration, a_motor))

            # Atualiza cinemática do motor (odometria aberta)
            v_m_next = max(-cfg.max_velocity, min(cfg.max_velocity, s.v_motor + a_motor * dt))
            v_m_avg = 0.5 * (s.v_motor + v_m_next)
            x_m_next = s.x_motor + v_m_avg * dt

            # Acoplamento com folga mecânica (backlash)
            v0 = s.v
            if half_backlash <= 0.0:
                x_next = x_m_next
                v_next = v_m_next
                a_cart = (v_next - v0) / dt
            else:
                m_total = cfg.cart_mass + cfg.pendulum_mass
                a_free = (
                    -cfg.cart_viscous_friction * v0 - m_total * cfg.g * math.sin(s.alpha)
                ) / m_total
                v_free = v0 + a_free * dt
                x_free = s.x + 0.5 * (v0 + v_free) * dt
                diff = x_m_next - x_free
                if diff > half_backlash:
                    x_next = x_m_next - half_backlash
                    v_next = v_m_next
                elif diff < -half_backlash:
                    x_next = x_m_next + half_backlash
                    v_next = v_m_next
                else:
                    x_next = x_free
                    v_next = v_free
                a_cart = (v_next - v0) / dt

            # Torque de peteleco ativo no sub-passo
            flick_now = s.flick_torque if s.flick_remaining_s > 0.0 else 0.0

            # Integra pêndulo via RK4
            theta_next, omega_next = self._rk4_pendulum_substep(
                s.theta,
                s.omega,
                v0,
                v_next,
                a_cart,
                s.alpha,
                s.total_wind,
                flick_now,
                dt,
            )

            s.x_motor = x_m_next
            s.v_motor = v_m_next
            s.x = x_next
            s.v = v_next
            s.theta = theta_next
            s.omega = omega_next
            s.time_s += dt

            if s.flick_remaining_s > 0.0:
                s.flick_remaining_s = max(0.0, s.flick_remaining_s - dt)
                if s.flick_remaining_s <= 0.0:
                    s.flick_torque = 0.0

        _, tau_aero, tau_fric, _ = self._pendulum_alpha_terms(
            s.theta, s.omega, s.v, s.alpha, s.total_wind, s.flick_torque
        )

        return StepMetrics(
            motor_torque_nm=last_tau_req,
            pullout_torque_nm=last_tau_pullout,
            torque_utilization=max_util,
            step_lost=any_step_lost,
            aero_torque_nm=tau_aero,
            friction_torque_nm=tau_fric,
            wind_total_mps=s.total_wind,
            active_flick_torque_nm=s.flick_torque,
            mechanical_energy_j=self.mechanical_energy(s),
        )
