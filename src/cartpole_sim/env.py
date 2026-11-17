"""Ambiente Gymnasium para pêndulo invertido com motor de passo, encoder e PBRS."""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cartpole_sim.config import DisturbanceConfig, PhysicsConfig, SimConfig
from cartpole_sim.physics import CartPolePhysics, PhysicsState, StepMetrics, wrap_angle


class CartPoleStepperEnv(gym.Env[np.ndarray, np.ndarray]):
    """Ambiente Gymnasium com observação realista (odometria + encoder) e PBRS."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    FRAME_DIM = 6

    def __init__(
        self,
        config: SimConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or SimConfig()
        self.render_mode = render_mode

        self.physics = CartPolePhysics(
            config=self.cfg.physics,
            disturbance=self.cfg.disturbance,
            rng=np.random.default_rng(),
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        obs_dim = self.FRAME_DIM * max(1, self.cfg.sensor.frame_stack)
        high = np.full((obs_dim,), 2.5, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self._delay_queue: deque[float] = deque()
        self._frame_buffer: deque[np.ndarray] = deque()
        self._prev_u_cmd: float = 0.0
        self._prev_theta_enc_q: float = math.pi
        self._omega_meas: float = 0.0
        self._odom_bias: float = 0.0
        self._step_count: int = 0
        self._last_potential: float = 0.0
        self._last_metrics: StepMetrics | None = None
        self._renderer: Any = None

    def set_disturbance_config(self, disturbance: DisturbanceConfig) -> None:
        """Atualiza parâmetros de distúrbio (usado pelo curriculum ou pela UI)."""
        self.cfg.disturbance = disturbance
        self.physics.dist = disturbance

    def set_physics_config(self, physics_cfg: PhysicsConfig) -> None:
        """Atualiza parâmetros físicos base (ex: troca entre polia e barra roscada)."""
        self.cfg.physics = physics_cfg
        self.physics.base_cfg = physics_cfg
        self.physics.cfg = PhysicsConfig(**vars(physics_cfg))

    def compute_potential(self, state: PhysicsState) -> float:
        """Função potencial Phi(s) >= 0 para PBRS (Ng et al., 1999)."""
        rc = self.cfg.reward
        if not rc.use_pbrs:
            return 0.0

        x_lim = max(self.physics.cfg.rail_half_length, 1e-6)
        x_ratio = min(abs(state.x) / x_lim, 1.0)
        rail_safety = max(0.0, 1.0 - x_ratio * x_ratio)

        height_norm = 0.5 * (1.0 + math.cos(state.theta))
        e_top = max(self.physics.cfg.top_potential_energy, 1e-6)
        e_norm = min(max(self.physics.mechanical_energy(state) / e_top, 0.0), 1.0)

        core = (
            rc.potential_height_weight * height_norm
            + rc.potential_energy_weight * e_norm
        )
        return float(rc.w_potential * core * rail_safety)

    def _quantize_encoder(self, angle_rad: float) -> float:
        sc = self.cfg.sensor
        if not sc.quantize_encoder:
            return angle_rad
        step_rad = (2.0 * math.pi) / max(1, sc.encoder_counts_per_rev)
        return round(angle_rad / step_rad) * step_rad

    def _read_encoder_frame(self, u_prev: float, *, init: bool = False) -> np.ndarray:
        """Constrói o vetor de observação de 1 frame usando apenas dados de sensores reais."""
        s = self.physics.state
        sc = self.cfg.sensor
        pc = self.physics.cfg

        noise = 0.0 if (init or sc.angle_noise_std <= 0.0) else float(
            self.np_random.normal(0.0, sc.angle_noise_std)
        )
        theta_enc_q = self._quantize_encoder(s.theta_encoder + noise)

        if init:
            self._prev_theta_enc_q = theta_enc_q
            self._omega_meas = s.omega
        else:
            omega_raw = (theta_enc_q - self._prev_theta_enc_q) / max(pc.control_dt, 1e-6)
            self._prev_theta_enc_q = theta_enc_q
            alpha_ema = min(max(sc.omega_ema_alpha, 0.0), 1.0)
            self._omega_meas = (
                alpha_ema * omega_raw + (1.0 - alpha_ema) * self._omega_meas
            )

        x_odom = s.x_motor + self._odom_bias
        x_norm = float(np.clip(x_odom / max(pc.rail_half_length, 1e-6), -1.5, 1.5))
        v_norm = float(np.clip(s.v_motor / max(pc.max_velocity, 1e-6), -1.5, 1.5))
        omega_norm = float(np.clip(self._omega_meas / max(sc.omega_obs_scale, 1e-6), -2.0, 2.0))

        return np.array(
            [
                x_norm,
                v_norm,
                math.sin(theta_enc_q),
                math.cos(theta_enc_q),
                omega_norm,
                float(np.clip(u_prev, -1.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def _get_stacked_obs(self) -> np.ndarray:
        return np.concatenate(list(self._frame_buffer), axis=0).astype(np.float32)

    def _sample_initial_state(self, options: dict[str, Any] | None) -> tuple[float, float, float, float]:
        opts = options or {}
        if all(k in opts for k in ("x", "v", "theta", "omega")):
            return (
                float(opts["x"]),
                float(opts["v"]),
                float(opts["theta"]),
                float(opts["omega"]),
            )

        mode = opts.get("init_mode")
        x_lim = self.cfg.physics.rail_half_length
        v_max = self.cfg.physics.max_velocity

        if mode == "upright":
            if self.cfg.eval_mode:
                return (
                    float(opts.get("x", 0.0)),
                    float(opts.get("v", 0.0)),
                    float(opts.get("theta", 0.0)),
                    float(opts.get("omega", 0.0)),
                )
            return (
                float(self.np_random.uniform(-0.25, 0.25) * x_lim),
                float(self.np_random.uniform(-0.15, 0.15) * v_max),
                float(self.np_random.normal(0.0, 0.18)),
                float(self.np_random.normal(0.0, 0.5)),
            )

        if self.cfg.eval_mode or mode == "hanging":
            return (
                float(opts.get("x", 0.0)),
                float(opts.get("v", 0.0)),
                float(opts.get("theta", math.pi)),
                float(opts.get("omega", 0.0)),
            )

        # Mistura de estados iniciais durante treino para acelerar propagação de valor no swing-up
        p = float(self.np_random.random())
        if p < 0.65:
            return (
                float(self.np_random.uniform(-0.15, 0.15) * x_lim),
                0.0,
                float(math.pi + self.np_random.normal(0.0, 0.12)),
                float(self.np_random.normal(0.0, 0.3)),
            )
        if p < 0.85:
            return (
                float(self.np_random.uniform(-0.35, 0.35) * x_lim),
                float(self.np_random.uniform(-0.2, 0.2) * v_max),
                float(self.np_random.uniform(-math.pi, math.pi)),
                float(self.np_random.uniform(-3.0, 3.0)),
            )
        return (
            float(self.np_random.uniform(-0.30, 0.30) * x_lim),
            float(self.np_random.uniform(-0.15, 0.15) * v_max),
            float(self.np_random.normal(0.0, 0.20)),
            float(self.np_random.normal(0.0, 0.6)),
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.physics.rng = self.np_random

        x0, v0, th0, om0 = self._sample_initial_state(options)
        randomize = self.cfg.randomize_disturbances
        if options and "randomize_disturbances" in options:
            randomize = bool(options["randomize_disturbances"])

        self.physics.reset(
            x=x0,
            v=v0,
            theta=th0,
            omega=om0,
            randomize_params=randomize,
            randomize_disturbances=randomize,
        )

        sc = self.cfg.sensor
        self._odom_bias = (
            float(self.np_random.normal(0.0, sc.odom_bias_std))
            if (not self.cfg.eval_mode and sc.odom_bias_std > 0.0)
            else 0.0
        )

        self._delay_queue = deque([0.0] * max(0, sc.action_delay_steps))
        self._prev_u_cmd = 0.0
        self._step_count = 0
        self._last_potential = self.compute_potential(self.physics.state)
        self._last_metrics = None

        init_frame = self._read_encoder_frame(u_prev=0.0, init=True)
        k_stack = max(1, sc.frame_stack)
        self._frame_buffer = deque([init_frame.copy() for _ in range(k_stack)], maxlen=k_stack)

        obs = self._get_stacked_obs()
        info = self._build_info(
            base_reward=0.0,
            pbrs_shaping=0.0,
            phi_curr=self._last_potential,
            phi_next=self._last_potential,
            metrics=None,
        )
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        u_cmd = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))

        # Aplica latência realista de atuação (ex: 1 ciclo = 20 ms)
        if len(self._delay_queue) > 0:
            u_exec = self._delay_queue.popleft()
            self._delay_queue.append(u_cmd)
        else:
            u_exec = u_cmd

        phi_curr = self._last_potential
        target_vel = u_exec * self.physics.cfg.max_velocity
        metrics = self.physics.step(
            target_vel,
            randomize_disturbances=self.cfg.randomize_disturbances,
        )
        self._last_metrics = metrics
        self._step_count += 1

        s = self.physics.state
        x_lim = max(self.physics.cfg.rail_half_length, 1e-6)
        terminated = abs(s.x) >= x_lim
        truncated = (not terminated) and (self._step_count >= self.cfg.max_episode_steps)

        # Regra #4 do PBRS: estado terminal tem potencial estritamente igual a 0
        phi_next = 0.0 if terminated else self.compute_potential(s)

        rc = self.cfg.reward
        shaping = (rc.gamma * phi_next - phi_curr) if rc.use_pbrs else 0.0

        h_next = 0.5 * (1.0 + math.cos(s.theta))
        theta_wrapped = wrap_angle(s.theta)
        upright_bonus = math.exp(-0.5 * ((theta_wrapped / max(rc.upright_sigma_rad, 1e-4)) ** 2))
        x_norm_true = s.x / x_lim
        r_height = rc.w_height * h_next + rc.w_upright * upright_bonus
        r_center = -rc.w_center * min(x_norm_true * x_norm_true, 1.0)
        r_action = -rc.w_action * (u_cmd * u_cmd)
        r_action_diff = -rc.w_action_diff * ((u_cmd - self._prev_u_cmd) ** 2)
        r_wall = -rc.wall_penalty if terminated else 0.0

        base_reward = r_height + r_center + r_action + r_action_diff + r_wall
        total_reward = float(base_reward + shaping)

        frame = self._read_encoder_frame(u_prev=u_cmd, init=False)
        self._frame_buffer.append(frame)

        self._prev_u_cmd = u_cmd
        self._last_potential = phi_next

        obs = self._get_stacked_obs()
        info = self._build_info(
            base_reward=base_reward,
            pbrs_shaping=shaping,
            phi_curr=phi_curr,
            phi_next=phi_next,
            metrics=metrics,
        )

        if self.render_mode == "human":
            self.render()

        return obs, total_reward, terminated, truncated, info

    def _build_info(
        self,
        *,
        base_reward: float,
        pbrs_shaping: float,
        phi_curr: float,
        phi_next: float,
        metrics: StepMetrics | None,
    ) -> dict[str, Any]:
        s = self.physics.state
        upright = math.cos(s.theta) > 0.96 and abs(s.omega) < 1.5
        return {
            "x_true": s.x,
            "x_odom": s.x_motor + self._odom_bias,
            "v_true": s.v,
            "theta_true": wrap_angle(s.theta),
            "theta_enc": wrap_angle(s.theta_encoder),
            "omega_true": s.omega,
            "omega_meas": self._omega_meas,
            "alpha_deg": math.degrees(s.alpha),
            "wind_mps": s.total_wind,
            "upright": upright,
            "base_reward": float(base_reward),
            "pbrs_shaping": float(pbrs_shaping),
            "phi_curr": float(phi_curr),
            "phi_next": float(phi_next),
            "torque_utilization": metrics.torque_utilization if metrics else 0.0,
            "step_lost": metrics.step_lost if metrics else False,
            "time_s": s.time_s,
        }

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from cartpole_sim.renderer import PygameStudioRenderer

            self._renderer = PygameStudioRenderer(
                mode="rgb_array" if self.render_mode == "rgb_array" else "human"
            )
        return self._renderer.render_env(self)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
