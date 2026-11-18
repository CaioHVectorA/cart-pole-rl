# PROTOCOL.md — Especificação Técnica, Física e Contratos de RL

Este documento define formalmente os modelos matemáticos, convenções de coordenadas, contratos de observação/ação, formulação PBRS, protocolo de *curriculum* e parâmetros de treinamento e avaliação do **cartpole-sim**.

---

## 1. Convenções de Coordenadas e Estado Físico

Todas as grandezas físicas utilizam o **Sistema Internacional (SI)**.

### 1.1. Referenciais
- **Posição no trilho ($x$):** medida ao longo do eixo do trilho em $[-L_{\text{rail}}, +L_{\text{rail}}]$, com $L_{\text{rail}} = 0.35\text{ m}$ (curso útil total de $0.70\text{ m}$). Positivo para a direita.
- **Inclinação da mesa ($\alpha$):** ângulo do trilho em radianos. $\alpha > 0$ eleva o lado $+x$ (subida para a direita).
- **Ângulo do pêndulo ($\theta$):** ângulo contínuo (sem *wrap*) relativo à **vertical gravitacional verdadeira**:
  - $\theta = 0\text{ rad}$: vertical para cima (equilíbrio invertido);
  - $\theta = \pi\text{ rad}$: vertical para baixo (repouso pendurado);
  - sentido positivo ($+\theta$, $+\omega$): rotação horária (inclinando para $+x$).
- **Ângulo medido pelo encoder ($\theta_{\text{enc}}$):**
  $$\theta_{\text{enc}} = \theta + \alpha$$
  Medido em relação à normal do carrinho/trilho inclinado.

### 1.2. Vetor de Estado Interno (`PhysicsState`)
| Variável | Unidade | Significado |
| :--- | :---: | :--- |
| `x`, `v` | $\text{m}$, $\text{m/s}$ | Posição e velocidade reais do carrinho no trilho |
| `x_motor`, `v_motor` | $\text{m}$, $\text{m/s}$ | Odometria em malha aberta do motor de passo |
| `theta`, `omega` | $\text{rad}$, $\text{rad/s}$ | Ângulo contínuo e velocidade angular do pêndulo (vertical real) |
| `alpha` | $\text{rad}$ | Inclinação atual da mesa |
| `wind_mean`, `wind_gust` | $\text{m/s}$ | Vento médio e componente estocástica de rajada |
| `flick_torque`, `flick_remaining_s` | $\text{N}\cdot\text{m}$, $\text{s}$ | Torque externo ativo de peteleco e tempo restante |

---

## 2. Modelo Físico e Integração (`CartPolePhysics`)

### 2.1. Parâmetros Inerciais do Pêndulo Composto
O pêndulo é composto por uma haste uniforme ($m_r = 0.035\text{ kg}$, $L_r = 0.32\text{ m}$, diâmetro $d_r = 0.008\text{ m}$) e uma massa cilíndrica posicionável ($m_b = 0.085\text{ kg}$, $r_b = 0.022\text{ m}$, distância $d_b = 0.22\text{ m}$):

- Massa total do pêndulo: $m_p = m_r + m_b = 0.120\text{ kg}$
- Centro de massa a partir do pivô:
  $$\ell = \frac{m_r (L_r / 2) + m_b d_b}{m_p} \approx 0.197\text{ m}$$
- Momento de inércia no pivô (eixo paralelo):
  $$I_o = \frac{1}{3} m_r L_r^2 + m_b d_b^2 + \frac{1}{2} m_b r_b^2$$
- Energia mecânica relativa ao repouso inferior ($\theta = \pi, \omega = 0$):
  $$E(s) = \frac{1}{2} I_o \omega^2 + m_p g \ell (1 + \cos\theta), \quad E_{\text{top}} = 2 m_p g \ell$$

### 2.2. Forças e Torques sobre o Pêndulo
1. **Atrito do mancal/encoder:**
   $$\tau_{\text{fric}}(\omega) = -b_p \omega - \tau_c \tanh\!\left(\frac{\omega}{\omega_0}\right)$$
   com $b_p = 0.0008\text{ N}\cdot\text{m}\cdot\text{s/rad}$, $\tau_c = 0.0018\text{ N}\cdot\text{m}$ e $\omega_0 = 0.25\text{ rad/s}$.
2. **Arrasto aerodinâmico e vento horizontal ($w = w_{\text{mean}} + w_{\text{gust}}$):**
   - Velocidade perpendicular base ao longo do pêndulo: $v_{\perp,0} = w\cos\theta - v\cos(\theta + \alpha)$.
   - Haste integrada por **quadratura Gauss-Legendre de 3 pontos** ao longo de $[0, L_r]$ com $C_{d,\text{rod}} = 1.15$, $v_\perp(s) = v_{\perp,0} - s\omega$.
   - Massa pontual em $d_b$ com área frontal $\pi r_b^2$ e $C_{d,\text{bob}} = 0.95$.
   - Soma resulta em $\tau_{\text{aero}}$ no pivô e reação $F_{\text{aero,rail}}$ projetada ao longo do trilho.
3. **Rajadas de vento (Ornstein-Uhlenbeck discreto por passo $\Delta t$):**
   $$w_{\text{gust}, t+1} = w_{\text{gust}, t} e^{-\Delta t / \tau_g} + \mathcal{N}\!\left(0,\; \sigma_g^2 (1 - e^{-2\Delta t / \tau_g})\right), \quad \tau_g \in \{0.6, 0.8\}\text{ s}$$
4. **Adaptação intra-episódio de rampa e vento (`DisturbanceConfig`):**
   - **Inclinação móvel:** a cada passo $\Delta t$, sorteia novo alvo $\alpha_{\text{target}} \in [\alpha_{\min}, \alpha_{\max}]$ com probabilidade $p_{\text{tilt}} = 1 - e^{-\lambda_{\text{tilt}} \Delta t}$ (`tilt_change_prob_per_sec`) e aproxima $\alpha$ com velocidade angular limitada a `tilt_slew_deg_per_s` ($\text{rad/s}$).
   - **Vento médio variável:** sorteia novo alvo $w_{\text{target}} \in [w_{\min}, w_{\max}]$ com probabilidade $p_{\text{wind}} = 1 - e^{-\lambda_{\text{wind}} \Delta t}$ (`wind_change_prob_per_sec`) e aplica suavização exponencial $w_{\text{mean}} \leftarrow w_{\text{mean}} + (1 - e^{-\Delta t / 1.0})(w_{\text{target}} - w_{\text{mean}})$.
5. **Petelecos (impulsos de duração finita):**
   Um impulso $J$ ($\text{N}\cdot\text{m}\cdot\text{s}$) é distribuído em uma janela $\Delta t_{\text{flick}} = 0.04\text{ s}$ como torque constante $\tau_{\text{flick}} = J / \Delta t_{\text{flick}}$.
6. **Randomização paramétrica por episódio (`param_randomization_pct = p`):**
   No `reset`, perturba uniformemente em $[1-p, 1+p]$: `cart_mass`, `bob_mass`, `cart_viscous_friction`, `pivot_viscous_friction`, `pivot_coulomb_friction`; em $[1 - 0.5p, 1 + 0.5p]$: `motor_holding_torque`; e em $[0.5, 1+p]$: `backlash_m`.

### 2.3. Dinâmica Acoplada e Motor de Passo NEMA 17
Cada passo de controle $\Delta t = 0.02\text{ s}$ ($50\text{ Hz}$) executa $N = 10$ sub-passos de $dt = 0.002\text{ s}$:

1. **Rampa de velocidade desejada:** $a_{\text{des}} = \text{clip}((v_{\text{target}} - v_{\text{motor}})/dt, -a_{\max}, +a_{\max})$.
2. **Acoplamento exato motor-pêndulo:** eliminando $\ddot{\theta}(a_c)$ na equação de Lagrange, obtém-se a relação linear instantânea $\tau_{\text{req}}(a_c) = K_a a_c + \tau_{\text{bias}}$ onde:
   $$m_{\text{eff}} = m_{\text{total}} - \frac{(m_p \ell \cos(\theta+\alpha))^2}{I_o}, \qquad K_a = \frac{J_{\text{rotor}}}{r_{\text{eff}}} + \frac{r_{\text{eff}}}{\eta} m_{\text{eff}}$$
   e $\tau_{\text{bias}}$ agrega termos centrífugos, acoplamento gravitacional do pêndulo, rampa ($m_{\text{total}} g \sin\alpha$), atrito linear do carrinho e reação aerodinâmica.
3. **Curva *pull-out* do NEMA 17** (`holding_torque = 0.45 N·m`, `corner_rpm = 400`, `cutoff_rpm = 2200`):
   - $\text{RPM} \le 400$: $\tau_{\text{pull}} = 0.82\,\tau_{\text{hold}}$
   - $400 < \text{RPM} < 2200$: decaimento exponencial suave $(1 - 0.85 \cdot \text{frac}^{0.85}) \cdot 0.82\,\tau_{\text{hold}}$
   - $\text{RPM} \ge 2200$: $0.05\,\tau_{\text{hold}}$
   - Se $|\tau_{\text{req}}| > \tau_{\text{pull}}$ e `enforce_torque_limit=True`: com `simulate_step_loss=False` (padrão), a aceleração é limitada ao torque máximo; com `simulate_step_loss=True`, o motor perde sincronismo (`a_motor = 0`, `v_motor *= 0.85`).
4. **Transmissões:**
   - `belt_pulley` (GT2 20T): $r_{\text{eff}} = \frac{0.040}{2\pi}\text{ m}$, $\eta = 0.90$, $v_{\max} = 1.25\text{ m/s}$, $a_{\max} = 8.5\text{ m/s}^2$.
   - `lead_screw` (TR8x8): $r_{\text{eff}} = \frac{0.008}{2\pi}\text{ m}$, $\eta = 0.55$, $v_{\max} = 0.32\text{ m/s}$, $a_{\max} = 4.5\text{ m/s}^2$.
5. **Folga (*backlash*) mecânica ($b = 0.4\text{ mm}$):** dentro da zona morta $\pm b/2$, o carrinho evolui livre sob atrito viscoso e gravidade da rampa; fora dela, acopla-se rigidamente ao deslocamento do motor.
6. **Integração do pêndulo:** Runge-Kutta de 4ª ordem (RK4) sobre $(\theta, \omega)$ usando velocidade interpolada do carrinho entre $v_0$ e $v_1$ e aceleração $a_c$.

---

## 3. Contrato do Ambiente Gymnasium (`CartPoleStepperEnv`)

### 3.1. Espaço de Ação e Latência
- **Ação:** `Box(low=-1.0, high=1.0, shape=(1,), dtype=float32)` representando a fração de velocidade comandada $u_t \in [-1, 1]$.
- **Latência:** fila FIFO de atraso de `action_delay_steps = 1` passo (20 ms). No passo $t$, o motor executa $u_{t-1}$ (inicializado em `0.0` no `reset`) como $v_{\text{target}} = u_{\text{exec}} \cdot v_{\max}$.

### 3.2. Espaço de Observação
- **Dimensão:** `Box(low=-2.5, high=2.5, shape=(72,), dtype=float32)` — empilhamento de `k = 12` frames (`FRAME_DIM = 6`), cobrindo `240 ms`.
- **Estrutura de cada frame (6 floats):**
  1. `x_norm` = $\text{clip}((x_{\text{motor}} + \text{bias}_{\text{odom}}) / L_{\text{rail}}, -1.5, 1.5)$
  2. `v_norm` = $\text{clip}(v_{\text{motor}} / v_{\max}, -1.5, 1.5)$
  3. $\sin(\theta_{\text{enc}, q})$
  4. $\cos(\theta_{\text{enc}, q})$
  5. `omega_norm` = $\text{clip}(\hat{\omega}_{\text{EMA}} / 15.0, -2.0, 2.0)$, onde $\hat{\omega}$ é derivado por diferença finita sobre $\theta_{\text{enc}, q}$ contínuo (quantizado em $2400$ ticks/volta + ruído $\sigma = 0.001\text{ rad}$) filtrado com EMA ($\alpha_{\text{EMA}} = 0.70$).
  6. `u_prev` = $\text{clip}(u_{\text{cmd}}, -1.0, 1.0)$.

### 3.3. Condições de Término e Inicialização
- **Terminação (`terminated`):** $|x| \ge L_{\text{rail}}$ (`0.35` m).
- **Truncamento (`truncated`):** `step_count >= max_episode_steps` (`1000` passos = `20.0` s).
- **Modos de `reset(options=...)`:**
  - Explicitando todas as chaves `x`, `v`, `theta`, `omega`: inicialização determinística.
  - `init_mode="hanging"` (ou `eval_mode=True` sem `init_mode`): pêndulo em repouso pendurado ($\theta = \pi$).
  - `init_mode="upright"`: em `eval_mode=True`, inicia exatamente em $\theta = 0$; em treino, com pequena perturbação.
  - Treino padrão (`eval_mode=False`, sem `init_mode`): mistura estocástica (65% próximo ao fundo, 20% estados aleatórios em $[- \pi, \pi]$, 15% próximo ao topo) para acelerar propagação de valor no *swing-up*.

---

## 4. Função de Recompensa e PBRS

A recompensa total por passo é $r_{\text{total}} = r_{\text{base}} + F(s, a, s')$:

### 4.1. Recompensa Base ($r_{\text{base}}$)
$$r_{\text{base}} = w_h h_{\text{next}} + w_u \exp\!\left(-\frac{\text{wrap}(\theta)^2}{2\sigma_u^2}\right) - w_c \min\!\left(\left(\frac{x}{L_{\text{rail}}}\right)^2, 1\right) - w_a u_{\text{cmd}}^2 - w_{\Delta a}(u_{\text{cmd}} - u_{\text{prev}})^2 - r_{\text{wall}}$$
onde:
- $h_{\text{next}} = \frac{1}{2}(1 + \cos\theta) \in [0, 1]$
- $w_h = 1.0$, $w_u = 0.5$, $\sigma_u = 0.35\text{ rad}$, $w_c = 0.20$, $w_a = 0.008$, $w_{\Delta a} = 0.015$
- $r_{\text{wall}} = 25.0$ se `terminated` else `0.0`.

### 4.2. Potential-Based Reward Shaping (Ng et al., 1999)
$$F(s, a, s') = \gamma \Phi(s') - \Phi(s), \qquad \gamma = 0.99$$
- **Potencial dentro da pista ($\Phi(s) \ge 0$):**
  $$\Phi(s) = w_{\Phi} \cdot \left(0.7 \cdot \frac{1 + \cos\theta}{2} + 0.3 \cdot \min\!\left(\frac{E(s)}{E_{\text{top}}}, 1\right)\right) \cdot \max\!\left(0,\; 1 - \left(\frac{|x|}{L_{\text{rail}}}\right)^2\right)$$
  com $w_{\Phi} = 3.0$.
- **Invariante terminal:** se `terminated == True`, $\Phi(s') \equiv 0.0$.

---

## 5. Protocolo de Curriculum Learning

Gerenciado por [`CurriculumManager`](src/cartpole_sim/curriculum.py) com avaliação periódica determinística (`CurriculumEvalCallback`), oferecendo duas sequências de estágios e modo fixo (`--fixed-stage`):

| Estágio | Descrição | Inclinação (`deg`) | Vento Médio (`m/s`) | Rajada $\sigma$ (`m/s`) | Peteleco (`1/s`) | Rand. Física | Dinâmica Intra-Episódio | Critério de Promoção |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | Mesa plana limpa | `0.0` | `0.0` | `0.0` | `0.0` | `0%` | Estático | `steps >= 80k` e `reward >= 1050` |
| **1** | Mesa inclinada + rand. leve | `[-7, +7]` | `0.0` | `0.0` | `0.0` | `±5%` | Estático | `steps >= 100k` e `reward >= 980` |
| **2** | Inclinada + vento/rajada leve | `[-10, +10]` | `[-4.0, +4.0]` | `0.8` | `0.0` | `±8%` | Estático | `steps >= 120k` e `reward >= 900` |
| **3** | Caos completo (`DEFAULT_STAGES` final) | `[-12, +12]` | `[-5.5, +5.5]` | `1.8` ($\tau=0.8\text{s}$) | `0.25` (`0.04–0.12 N·m·s`) | `±10%` | Estático | Final em `DEFAULT_STAGES`; em `ADAPTIVE_CHAOS_STAGES`: `steps >= 120k` e `reward >= 880` |
| **4** | Caos Adaptativo Dinâmico (`--adaptive-chaos`) | `[-14, +14]` | `[-6.5, +6.5]` | `2.2` ($\tau=0.6\text{s}$) | `0.35` (`0.04–0.14 N·m·s`) | `±15%` | Rampa `0.15/s` (`8.0°/s`), Vento `0.20/s` | Estágio final |

Ao avançar de estágio, o algoritmo salva `stage_<id>_mastered.zip`, zera `best_mean_reward` no callback de avaliação e atualiza `training_env` e `eval_env`. A cada avaliação, persiste `curriculum_state.json`, `last_model.zip` e `vec_normalize.pkl`; ao usar `--resume`, restaura `current_idx`, `steps_in_stage`, `best_mean_reward` e estatísticas de `VecNormalize`. Com `--fixed-stage`, apenas o estágio indicado por `--initial-stage` permanece ativo sem promoções.

---

## 6. Protocolo de Treinamento e Avaliação

### 6.1. Hiperparâmetros do PPO (`train.py`)
- **Política:** `MlpPolicy`, `net_arch={"pi": [128, 128], "vf": [128, 128]}`, `activation_fn=nn.Tanh`, `log_std_init=-0.4`.
- **Otimização padrão:** `lr=2.5e-4`, `n_steps=1024` (ou `256` se `total_timesteps < 50_000`), `batch_size=min(256, n_steps * n_envs)`, `n_epochs=10`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.001`, `vf_coef=0.5`, `max_grad_norm=0.5`.
- **Normalização:** `VecNormalize(norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99)`.
- **Critério `upright` (telemetria/avaliação):** $\cos(\theta) > 0.96$ (aprox. $|\theta| < 16.26^\circ$) **e** $|\omega| < 1.5\text{ rad/s}$.

### 6.2. Cenários Oficiais de Benchmark (`scripts/evaluate.py`)
Todos os cenários avaliam `eval_mode=True`, `init_mode="hanging"`, 1000 passos (`20` s) com ação determinística:
1. **Plano limpo** (`alpha=0°`, sem vento).
2. **Mesa inclinada $+10^\circ$**.
3. **Mesa inclinada $-10^\circ$**.
4. **Vento constante $+4.5\text{ m/s}$**.
5. **Rampa $-8^\circ$ + Vento $+4.0\text{ m/s}$ + Rajadas $\sigma=1.5\text{ m/s}$**.
6. **Caos + Petelecos determinísticos:** rampa $+6^\circ$, vento $-3.0\text{ m/s}$, rajadas $\sigma=1.2\text{ m/s}$, peteleco $+0.09\text{ N}\cdot\text{m}\cdot\text{s}$ em $t=6\text{ s}$ (passo 300) e $-0.09\text{ N}\cdot\text{m}\cdot\text{s}$ em $t=12\text{ s}$ (passo 600).
