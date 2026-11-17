"""Estúdio visual interativo em Pygame totalmente desacoplado da física e do treino."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pygame

from cartpole_sim.config import DriveMode
from cartpole_sim.physics import wrap_angle

if TYPE_CHECKING:
    from cartpole_sim.env import CartPoleStepperEnv

# Paleta escura estilo laboratório/telemetria
BG_COLOR = (17, 20, 27)
PANEL_BG = (24, 28, 38)
PANEL_BORDER = (46, 54, 72)
GRID_COLOR = (30, 35, 48)
TEXT_COLOR = (230, 236, 245)
MUTED_TEXT = (148, 160, 182)
ACCENT_BLUE = (76, 163, 255)
ACCENT_CYAN = (56, 217, 217)
ACCENT_GREEN = (74, 222, 128)
ACCENT_AMBER = (251, 191, 36)
ACCENT_RED = (248, 113, 113)
ACCENT_PURPLE = (167, 139, 250)
RAIL_COLOR = (120, 132, 154)
CART_COLOR = (59, 130, 246)
ROD_COLOR = (226, 232, 240)
BOB_COLOR = (245, 158, 11)


@dataclass
class SliderWidget:
    """Slider horizontal interativo para controles ao vivo."""

    label: str
    rect: pygame.Rect
    min_val: float
    max_val: float
    value: float
    unit: str = ""
    fmt: str = "{:.1f}"
    dragging: bool = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            hit_rect = self.rect.inflate(10, 14)
            if hit_rect.collidepoint(event.pos):
                self.dragging = True
                self._update_from_x(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                return True
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_from_x(event.pos[0])
            return True
        return False

    def _update_from_x(self, mx: int) -> None:
        t = (mx - self.rect.x) / max(1, self.rect.width)
        t = max(0.0, min(1.0, t))
        self.value = self.min_val + t * (self.max_val - self.min_val)

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        val_str = self.fmt.format(self.value) + self.unit
        lbl_surf = font.render(f"{self.label}: {val_str}", True, TEXT_COLOR)
        surface.blit(lbl_surf, (self.rect.x, self.rect.y - 20))

        track_y = self.rect.centery
        pygame.draw.line(
            surface, PANEL_BORDER, (self.rect.x, track_y), (self.rect.right, track_y), 4
        )
        t = (self.value - self.min_val) / max(self.max_val - self.min_val, 1e-9)
        t = max(0.0, min(1.0, t))
        knob_x = int(self.rect.x + t * self.rect.width)

        # Marca central zero se o range cruza 0
        if self.min_val < 0.0 < self.max_val:
            z_t = (0.0 - self.min_val) / (self.max_val - self.min_val)
            zx = int(self.rect.x + z_t * self.rect.width)
            pygame.draw.line(surface, MUTED_TEXT, (zx, track_y - 6), (zx, track_y + 6), 1)
            pygame.draw.line(surface, ACCENT_BLUE, (zx, track_y), (knob_x, track_y), 4)
        else:
            pygame.draw.line(surface, ACCENT_BLUE, (self.rect.x, track_y), (knob_x, track_y), 4)

        color = ACCENT_CYAN if self.dragging else TEXT_COLOR
        pygame.draw.circle(surface, color, (knob_x, track_y), 7)
        pygame.draw.circle(surface, BG_COLOR, (knob_x, track_y), 3)


@dataclass
class ButtonWidget:
    """Botão simples clicável."""

    label: str
    rect: pygame.Rect
    active: bool = False
    accent: tuple[int, int, int] = ACCENT_BLUE

    def clicked(self, event: pygame.event.Event) -> bool:
        return (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        )

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        fill = self.accent if self.active else (35, 42, 56)
        border = self.accent if self.active else PANEL_BORDER
        text_col = (15, 20, 28) if self.active else TEXT_COLOR
        pygame.draw.rect(surface, fill, self.rect, border_radius=6)
        pygame.draw.rect(surface, border, self.rect, width=1, border_radius=6)
        txt = font.render(self.label, True, text_col)
        surface.blit(txt, txt.get_rect(center=self.rect.center))


@dataclass
class StudioActions:
    """Sinais disparados pela interação do usuário na UI."""

    toggle_mode: bool = False
    reset_hanging: bool = False
    reset_upright: bool = False
    toggle_drive: bool = False
    toggle_pause: bool = False
    flick_impulse: float = 0.0
    manual_action: float = 0.0
    quit_requested: bool = False


class PygameStudioRenderer:
    """Renderizador interativo com bancada inclinável, partículas de vento e telemetria."""

    WIDTH = 1280
    HEIGHT = 760
    SIDEBAR_W = 410

    def __init__(self, mode: str = "human", history_len: int = 240) -> None:
        self.mode = mode
        self._initialized = False
        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.font_sm: pygame.font.Font | None = None
        self.font_md: pygame.font.Font | None = None
        self.font_lg: pygame.font.Font | None = None

        self.history_len = history_len
        self.hist_theta: deque[float] = deque(maxlen=history_len)
        self.hist_x: deque[float] = deque(maxlen=history_len)
        self.hist_phi: deque[float] = deque(maxlen=history_len)
        self.hist_reward: deque[float] = deque(maxlen=history_len)
        self.hist_torque: deque[float] = deque(maxlen=history_len)

        self._wind_particles: list[list[float]] = []
        self._last_bob_screen: tuple[int, int] = (0, 0)
        self._last_flick_vis_timer: float = 0.0
        self._last_flick_sign: float = 1.0

        # Widgets construídos no init
        sx = self.WIDTH - self.SIDEBAR_W + 24
        sw = self.SIDEBAR_W - 48
        self.slider_tilt = SliderWidget("Inclinação Mesa (α)", pygame.Rect(sx, 86, sw, 14), -15.0, 15.0, 0.0, "°")
        self.slider_wind = SliderWidget("Vento Médio", pygame.Rect(sx, 134, sw, 14), -8.0, 8.0, 0.0, " m/s")
        self.slider_gust = SliderWidget("Rajadas (OU)", pygame.Rect(sx, 182, sw, 14), 0.0, 4.0, 0.0, " m/s")
        self.slider_flick = SliderWidget(
            "Força do Peteleco", pygame.Rect(sx, 230, sw, 14), 0.02, 0.20, 0.08, " N·m·s", "{:.2f}"
        )

        bw = (sw - 10) // 2
        self.btn_ai = ButtonWidget("Modo: IA (PPO)", pygame.Rect(sx, 260, bw, 32), active=True, accent=ACCENT_GREEN)
        self.btn_drive = ButtonWidget("Polia GT2", pygame.Rect(sx + bw + 10, 260, bw, 32), active=False, accent=ACCENT_CYAN)
        self.btn_reset_hang = ButtonWidget("Reset Embaixo (R)", pygame.Rect(sx, 300, bw, 30))
        self.btn_reset_up = ButtonWidget("Reset Topo (U)", pygame.Rect(sx + bw + 10, 300, bw, 30))
        self.btn_flick_l = ButtonWidget("← Peteleco (Q)", pygame.Rect(sx, 338, bw, 30), accent=ACCENT_AMBER)
        self.btn_flick_r = ButtonWidget("Peteleco → (E)", pygame.Rect(sx + bw + 10, 338, bw, 30), accent=ACCENT_AMBER)

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        pygame.init()
        pygame.font.init()
        if self.mode == "human":
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Cart-Pole Stepper RL Studio (PBRS + Sim-to-Real)")
        else:
            self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont("dejavusansmono", 13)
        self.font_md = pygame.font.SysFont("dejavusans", 15, bold=True)
        self.font_lg = pygame.font.SysFont("dejavusans", 19, bold=True)

        rng = np.random.default_rng(42)
        view_w = self.WIDTH - self.SIDEBAR_W
        self._wind_particles = [
            [float(rng.uniform(0, view_w)), float(rng.uniform(40, self.HEIGHT - 40)), float(rng.uniform(0.6, 1.4))]
            for _ in range(55)
        ]
        self._initialized = True

    def sync_sliders_from_env(self, env: CartPoleStepperEnv) -> None:
        """Sincroniza sliders com o estado atual do ambiente."""
        self.slider_tilt.value = math.degrees(env.physics.state.alpha)
        self.slider_wind.value = env.physics.state.wind_mean
        self.slider_gust.value = env.cfg.disturbance.wind_gust_std_mps

    def poll_events(self, env: CartPoleStepperEnv, *, ai_active: bool, paused: bool) -> StudioActions:
        """Processa teclado, mouse, sliders e peteleco no pêndulo."""
        self._ensure_init()
        actions = StudioActions()
        self.btn_ai.active = ai_active
        self.btn_ai.label = "Modo: IA (PPO)" if ai_active else "Modo: MANUAL"
        self.btn_drive.label = (
            "Polia GT2 (M)"
            if env.physics.cfg.drive_mode == DriveMode.BELT_PULLEY
            else "Fuso TR8x8 (M)"
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                actions.quit_requested = True
                return actions

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    actions.quit_requested = True
                    return actions
                if event.key == pygame.K_SPACE:
                    actions.toggle_mode = True
                elif event.key == pygame.K_r:
                    actions.reset_hanging = True
                elif event.key == pygame.K_u:
                    actions.reset_upright = True
                elif event.key == pygame.K_m:
                    actions.toggle_drive = True
                elif event.key == pygame.K_p:
                    actions.toggle_pause = True
                elif event.key in (pygame.K_q, pygame.K_f):
                    actions.flick_impulse = -self.slider_flick.value
                elif event.key in (pygame.K_e, pygame.K_g):
                    actions.flick_impulse = +self.slider_flick.value

            # Sliders
            if self.slider_tilt.handle_event(event):
                env.physics.state.alpha = math.radians(self.slider_tilt.value)
                env.cfg.disturbance.table_tilt_deg = self.slider_tilt.value
            if self.slider_wind.handle_event(event):
                env.physics.state.wind_mean = self.slider_wind.value
                env.cfg.disturbance.wind_mean_mps = self.slider_wind.value
            if self.slider_gust.handle_event(event):
                env.cfg.disturbance.wind_gust_std_mps = self.slider_gust.value
                if self.slider_gust.value <= 0.0:
                    env.physics.state.wind_gust = 0.0
            self.slider_flick.handle_event(event)

            # Botões
            if self.btn_ai.clicked(event):
                actions.toggle_mode = True
            elif self.btn_drive.clicked(event):
                actions.toggle_drive = True
            elif self.btn_reset_hang.clicked(event):
                actions.reset_hanging = True
            elif self.btn_reset_up.clicked(event):
                actions.reset_upright = True
            elif self.btn_flick_l.clicked(event):
                actions.flick_impulse = -self.slider_flick.value
            elif self.btn_flick_r.clicked(event):
                actions.flick_impulse = +self.slider_flick.value
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Clique no canvas principal aplica peteleco no pêndulo na direção do clique
                if event.pos[0] < self.WIDTH - self.SIDEBAR_W:
                    bx, _ = self._last_bob_screen
                    sign = 1.0 if event.pos[0] < bx else -1.0
                    actions.flick_impulse = sign * self.slider_flick.value

        if abs(actions.flick_impulse) > 0.0:
            self._last_flick_vis_timer = 0.35
            self._last_flick_sign = 1.0 if actions.flick_impulse > 0 else -1.0

        keys = pygame.key.get_pressed()
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        if left and not right:
            actions.manual_action = -1.0
        elif right and not left:
            actions.manual_action = 1.0

        return actions

    def record_step(self, info: dict) -> None:
        self.hist_theta.append(float(info.get("theta_true", 0.0)))
        self.hist_x.append(float(info.get("x_true", 0.0)))
        self.hist_phi.append(float(info.get("phi_next", 0.0)))
        self.hist_reward.append(
            float(info.get("base_reward", 0.0)) + float(info.get("pbrs_shaping", 0.0))
        )
        self.hist_torque.append(float(info.get("torque_utilization", 0.0)))

    def clear_history(self) -> None:
        self.hist_theta.clear()
        self.hist_x.clear()
        self.hist_phi.clear()
        self.hist_reward.clear()
        self.hist_torque.clear()

    def _draw_wind_particles(self, surface: pygame.Surface, wind_mps: float, view_w: int, dt: float) -> None:
        if abs(wind_mps) < 0.15:
            return
        speed_px = wind_mps * 42.0
        streak_len = max(6, min(36, int(abs(wind_mps) * 4.5)))
        alpha_c = (95, 165, 240)
        for p in self._wind_particles:
            p[0] = (p[0] + speed_px * p[2] * dt) % view_w
            x0, y0 = int(p[0]), int(p[1])
            x1 = x0 + (streak_len if wind_mps > 0 else -streak_len)
            pygame.draw.line(surface, alpha_c, (x0, y0), (x1, y0), 1)

    def _draw_strip_chart(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        title: str,
        series: deque[float],
        y_min: float,
        y_max: float,
        color: tuple[int, int, int],
        zero_line: bool = True,
    ) -> None:
        assert self.font_sm is not None
        pygame.draw.rect(surface, (18, 22, 30), rect, border_radius=6)
        pygame.draw.rect(surface, PANEL_BORDER, rect, width=1, border_radius=6)

        if zero_line and y_min < 0.0 < y_max:
            zt = (0.0 - y_min) / (y_max - y_min)
            zy = int(rect.bottom - zt * rect.height)
            pygame.draw.line(surface, (55, 65, 85), (rect.x + 2, zy), (rect.right - 2, zy), 1)

        val_txt = f"{series[-1]:+.2f}" if series else "0.00"
        lbl = self.font_sm.render(f"{title}: {val_txt}", True, MUTED_TEXT)
        surface.blit(lbl, (rect.x + 6, rect.y + 4))

        if len(series) < 2:
            return

        pts: list[tuple[int, int]] = []
        n = len(series)
        span = max(y_max - y_min, 1e-6)
        pad_top = 20
        usable_h = max(10, rect.height - pad_top - 4)
        for idx, val in enumerate(series):
            px = int(rect.x + 4 + (idx / max(1, self.history_len - 1)) * (rect.width - 8))
            t = max(0.0, min(1.0, (val - y_min) / span))
            py = int(rect.bottom - 4 - t * usable_h)
            pts.append((px, py))
        if len(pts) >= 2:
            pygame.draw.lines(surface, color, False, pts, 2)

    def render_env(
        self,
        env: CartPoleStepperEnv,
        *,
        extra_status: str = "",
        paused: bool = False,
    ) -> np.ndarray | None:
        """Desenha bancada, pêndulo, partículas de vento, sliders e gráficos."""
        self._ensure_init()
        assert self.screen is not None and self.clock is not None
        assert self.font_sm is not None and self.font_md is not None and self.font_lg is not None

        s = env.physics.state
        cfg = env.physics.cfg
        view_w = self.WIDTH - self.SIDEBAR_W
        dt = cfg.control_dt

        self.screen.fill(BG_COLOR)

        # Grade de fundo
        for gx in range(0, view_w, 40):
            pygame.draw.line(self.screen, GRID_COLOR, (gx, 0), (gx, self.HEIGHT), 1)
        for gy in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, GRID_COLOR, (0, gy), (view_w, gy), 1)

        self._draw_wind_particles(self.screen, s.total_wind, view_w, dt)

        # Escala espacial: rail_half_length ocupa ~340 px de cada lado
        px_per_m = 320.0 / max(cfg.rail_half_length, 0.1)
        cx, cy = view_w // 2, int(self.HEIGHT * 0.56)

        # Vetores do trilho inclinado por alpha (eixo +x sobe à direita quando alpha > 0)
        ca, sa = math.cos(s.alpha), math.sin(s.alpha)
        tx, ty = ca, -sa  # +x na tela: direita (+), cima (-)
        nx, ny = -sa, -ca  # normal para cima da mesa

        half_px = cfg.rail_half_length * px_per_m
        p_left = (cx - half_px * tx, cy - half_px * ty)
        p_right = (cx + half_px * tx, cy + half_px * ty)

        # Base / mesa de apoio inclinada
        base_off = 32.0
        b_left = (p_left[0] - base_off * nx, p_left[1] - base_off * ny)
        b_right = (p_right[0] - base_off * nx, p_right[1] - base_off * ny)
        pygame.draw.line(self.screen, (60, 70, 90), b_left, b_right, 8)
        pygame.draw.line(self.screen, RAIL_COLOR, p_left, p_right, 5)

        # Marca central (x = 0) e fins de curso nas extremidades
        c_mark_a = (cx - 10 * nx, cy - 10 * ny)
        c_mark_b = (cx + 10 * nx, cy + 10 * ny)
        pygame.draw.line(self.screen, ACCENT_GREEN, c_mark_a, c_mark_b, 2)
        for end_pt in (p_left, p_right):
            pygame.draw.circle(self.screen, ACCENT_RED, (int(end_pt[0]), int(end_pt[1])), 7)

        # Transmissão visual: polia + correia OU barra roscada
        if cfg.drive_mode == DriveMode.BELT_PULLEY:
            for end_pt in (p_left, p_right):
                pygame.draw.circle(self.screen, (90, 102, 124), (int(end_pt[0]), int(end_pt[1])), 13, 3)
            belt_off = -12.0
            pygame.draw.line(
                self.screen,
                (170, 180, 196),
                (p_left[0] + belt_off * nx, p_left[1] + belt_off * ny),
                (p_right[0] + belt_off * nx, p_right[1] + belt_off * ny),
                2,
            )
        else:
            # Fuso com espiras animadas conforme x_motor
            screw_off = -10.0
            sl = (p_left[0] + screw_off * nx, p_left[1] + screw_off * ny)
            sr = (p_right[0] + screw_off * nx, p_right[1] + screw_off * ny)
            pygame.draw.line(self.screen, (150, 162, 180), sl, sr, 4)
            phase = (s.x_motor / max(cfg.lead_screw_pitch, 1e-4)) % 1.0
            n_threads = 34
            for i in range(n_threads):
                u = -1.0 + 2.0 * ((i + phase) / n_threads)
                if -1.0 <= u <= 1.0:
                    px = cx + u * half_px * tx + screw_off * nx
                    py = cy + u * half_px * ty + screw_off * ny
                    pygame.draw.line(
                        self.screen,
                        (210, 220, 235),
                        (px - 4 * nx, py - 4 * ny),
                        (px + 4 * nx, py + 4 * ny),
                        2,
                    )

        # Carrinho
        cart_x = cx + s.x * px_per_m * tx
        cart_y = cy + s.x * px_per_m * ty
        cart_w, cart_h = 56, 30
        corners_local = [
            (-cart_w / 2, -cart_h / 2),
            (cart_w / 2, -cart_h / 2),
            (cart_w / 2, cart_h / 2),
            (-cart_w / 2, cart_h / 2),
        ]
        cart_poly = [
            (cart_x + lx * tx + ly * nx, cart_y + lx * ty + ly * ny)
            for lx, ly in corners_local
        ]
        pygame.draw.polygon(self.screen, CART_COLOR, cart_poly)
        pygame.draw.polygon(self.screen, TEXT_COLOR, cart_poly, 2)

        # Pivô (eixo do encoder)
        pivot_x = cart_x + 12.0 * nx
        pivot_y = cart_y + 12.0 * ny

        # Pêndulo no referencial vertical verdadeiro: theta=0 para cima, +theta para direita (+X)
        sin_t, cos_t = math.sin(s.theta), math.cos(s.theta)
        rod_px = cfg.rod_length * px_per_m
        bob_px = min(cfg.bob_distance, cfg.rod_length) * px_per_m

        tip_x = pivot_x + rod_px * sin_t
        tip_y = pivot_y - rod_px * cos_t
        bob_x = pivot_x + bob_px * sin_t
        bob_y = pivot_y - bob_px * cos_t
        self._last_bob_screen = (int(bob_x), int(bob_y))

        # Guia tracejada da vertical gravitacional
        pygame.draw.line(
            self.screen, (60, 75, 95), (pivot_x, pivot_y), (pivot_x, pivot_y - rod_px - 22), 1
        )
        # Haste rígida e peso
        pygame.draw.line(self.screen, ROD_COLOR, (pivot_x, pivot_y), (tip_x, tip_y), 6)
        bob_r_px = max(10, int(cfg.bob_radius * px_per_m))
        pygame.draw.circle(self.screen, BOB_COLOR, (int(bob_x), int(bob_y)), bob_r_px)
        pygame.draw.circle(self.screen, TEXT_COLOR, (int(bob_x), int(bob_y)), bob_r_px, 2)

        # Disco do encoder no pivô girando junto com theta_encoder
        pygame.draw.circle(self.screen, (25, 32, 44), (int(pivot_x), int(pivot_y)), 10)
        enc_ang = s.theta_encoder
        pygame.draw.line(
            self.screen,
            ACCENT_CYAN,
            (pivot_x, pivot_y),
            (pivot_x + 9 * math.sin(enc_ang), pivot_y - 9 * math.cos(enc_ang)),
            2,
        )

        # Seta visual de peteleco
        if self._last_flick_vis_timer > 0.0:
            self._last_flick_vis_timer = max(0.0, self._last_flick_vis_timer - dt)
            ax0 = int(bob_x - self._last_flick_sign * 55)
            ax1 = int(bob_x - self._last_flick_sign * 16)
            pygame.draw.line(self.screen, ACCENT_RED, (ax0, int(bob_y)), (ax1, int(bob_y)), 4)
            pygame.draw.circle(self.screen, ACCENT_RED, (ax1, int(bob_y)), 5)

        # Cabeçalho HUD na área da bancada
        theta_deg = math.degrees(wrap_angle(s.theta))
        enc_deg = math.degrees(wrap_angle(s.theta_encoder))
        phi_now = env.compute_potential(s)
        hud_lines = [
            f"x: {s.x:+.3f} m  (odom {s.x_motor:+.3f} m) | v: {s.v:+.2f} m/s",
            f"θ_real: {theta_deg:+6.1f}° | θ_encoder: {enc_deg:+6.1f}° | ω: {s.omega:+5.2f} rad/s",
            f"Mesa α: {math.degrees(s.alpha):+5.1f}° | Vento: {s.total_wind:+5.2f} m/s | Φ(s): {phi_now:4.2f}",
        ]
        if extra_status:
            hud_lines.append(extra_status)
        if paused:
            hud_lines.append("[PAUSADO - pressione P para retomar]")

        for idx, line in enumerate(hud_lines):
            txt = self.font_sm.render(line, True, TEXT_COLOR)
            self.screen.blit(txt, (16, 14 + idx * 20))

        hint = self.font_sm.render(
            "Clique no canvas para dar peteleco | Setas/A-D: Manual | Espaço: IA/Manual | M: Polia/Fuso | R: Reset",
            True,
            MUTED_TEXT,
        )
        self.screen.blit(hint, (16, self.HEIGHT - 26))

        # Barra lateral (Controles + Gráficos de Telemetria)
        sidebar_rect = pygame.Rect(view_w, 0, self.SIDEBAR_W, self.HEIGHT)
        pygame.draw.rect(self.screen, PANEL_BG, sidebar_rect)
        pygame.draw.line(self.screen, PANEL_BORDER, (view_w, 0), (view_w, self.HEIGHT), 2)

        title = self.font_lg.render("Controles & Telemetria", True, TEXT_COLOR)
        self.screen.blit(title, (view_w + 24, 18))

        self.slider_tilt.draw(self.screen, self.font_sm)
        self.slider_wind.draw(self.screen, self.font_sm)
        self.slider_gust.draw(self.screen, self.font_sm)
        self.slider_flick.draw(self.screen, self.font_sm)

        self.btn_ai.draw(self.screen, self.font_sm)
        self.btn_drive.draw(self.screen, self.font_sm)
        self.btn_reset_hang.draw(self.screen, self.font_sm)
        self.btn_reset_up.draw(self.screen, self.font_sm)
        self.btn_flick_l.draw(self.screen, self.font_sm)
        self.btn_flick_r.draw(self.screen, self.font_sm)

        # Gráficos em tempo real
        gx = view_w + 24
        gw = self.SIDEBAR_W - 48
        self._draw_strip_chart(
            self.screen, pygame.Rect(gx, 384, gw, 80), "θ real (rad)", self.hist_theta, -math.pi, math.pi, ACCENT_CYAN
        )
        self._draw_strip_chart(
            self.screen,
            pygame.Rect(gx, 474, gw, 80),
            "Posição x (m)",
            self.hist_x,
            -cfg.rail_half_length,
            cfg.rail_half_length,
            ACCENT_BLUE,
        )
        self._draw_strip_chart(
            self.screen,
            pygame.Rect(gx, 564, gw, 80),
            "Potencial PBRS Φ(s)",
            self.hist_phi,
            0.0,
            max(env.cfg.reward.w_potential, 1.0),
            ACCENT_PURPLE,
            zero_line=False,
        )
        t_color = ACCENT_RED if (self.hist_torque and self.hist_torque[-1] > 1.0) else ACCENT_GREEN
        self._draw_strip_chart(
            self.screen,
            pygame.Rect(gx, 654, gw, 80),
            "Uso Torque NEMA 17 (1.0 = limite)",
            self.hist_torque,
            0.0,
            1.4,
            t_color,
            zero_line=False,
        )

        if self.mode == "human":
            pygame.display.flip()
            self.clock.tick(int(1.0 / max(dt, 1e-3)))
            return None

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def close(self) -> None:
        if self._initialized:
            pygame.quit()
            self._initialized = False
