# CONTEXT.md — Contexto de Engenharia e Estado do Projeto

Este documento explica a motivação *Sim-to-Real*, as principais decisões arquiteturais e o estado atual da codebase **cart-pole-rl**.

---

## 1. Propósito do Projeto

O objetivo do projeto é simular uma **bancada real de pêndulo invertido linear** acionada por motor de passo **NEMA 17** com sensorização realista (odometria em malha aberta do motor + encoder rotativo incremental no pivô), e treinar um controlador neural em **Reinforcement Learning (PPO)** capaz de executar *swing-up* completo (partindo pendurado embaixo) e estabilização robusta no topo sob perturbações severas não observadas diretamente:

- **Inclinação da mesa ($\alpha$)** sem IMU/inclinômetro no carrinho;
- **Vento constante e rajadas turbulentas** agindo sobre haste e massa;
- **Impulsos externos súbitos ("petelecos")**;
- **Incertezas paramétricas e imperfeições mecânicas** (folga/*backlash*, atrito seco de Coulomb, curva de torque *pull-out* dependente da RPM, atraso de atuação de 1 ciclo).

---

## 2. Decisões de Projeto e Racional (*Why*)

### 2.1. Física Fidedigna vs. CartPole Clássico
Ao contrário do `CartPole-v1` padrão (barra uniforme sem massa concentrada, força ideal ilimitada e estado perfeito), o simulador em [`physics.py`](src/cartpole_sim/physics.py) modela:
- **Pêndulo físico composto:** haste de alumínio/aço (`0.32` m, `35` g) + massa posicionável (`85` g a `0.22` m do pivô), com momento de inércia exato pelo teorema dos eixos paralelos.
- **Duas transmissões mecânicas:**
  - `belt_pulley` (polia GT2 20 dentes, passo `40` mm/rev, eficiência `90%`, até `1.25` m/s e `8.5` m/s²);
  - `lead_screw` (fuso trapezoidal TR8x8, passo `8` mm/rev, eficiência `55%`, até `0.32` m/s e `4.5` m/s²).
- **Acoplamento dinâmico exato motor-carga:** em vez de impor aceleração cinemática infinita, a equação calcula o torque necessário no eixo do motor considerando inércia do rotor refletida, força inercial do pêndulo, componente gravitacional da rampa ($m_{\text{total}} g \sin\alpha$) e força aerodinâmica, limitando a aceleração real pela curva *pull-out* do NEMA 17.
- **Integração RK4 com 10 sub-passos:** cada passo de controle (`20` ms = 50 Hz) integra a dinâmica do pêndulo em passos de `2` ms com Runge-Kutta de 4ª ordem, garantindo estabilidade numérica mesmo em atrito seco suavizado e impactos rápidos.

### 2.2. Observabilidade Parcial e *Frame Stacking*
Na bancada física real não há sensor de inclinação da mesa nem anemômetro:
- O encoder mede $\theta_{\text{enc}} = \theta + \alpha$ (relativo à normal do carrinho, quantizado em 2400 ticks/volta com ruído gaussiano).
- A posição e velocidade lineares vêm da odometria comandada do motor de passo (`x_motor`, `v_motor`), sujeitas a *backlash* (`0.4` mm) em relação à posição real `x` do carrinho.
- A rede MLP não é recorrente (LSTM), portanto usa **empilhamento temporal de 12 frames** (`12 × 20 ms = 240 ms` de janela) incluindo o comando anterior `u_prev`. Pela discrepância entre comandos passados e a resposta angular observada, a política infere implicitamente a inclinação da mesa $\alpha$, a força do vento e impulsos recentes.

### 2.3. Shaping com PBRS Estrito
*Swing-up* com recompensa esparsa ou heurísticas ad-hoc frequentemente introduz ótimos locais (ex.: oscilar indefinidamente sem subir, ou bater na parede para encerrar um episódio com retorno negativo). Para evitar isso, [`env.py`](src/cartpole_sim/env.py) aplica **Potential-Based Reward Shaping** (Ng, Harada & Russell, 1999), garantindo teoricamente que a política ótima da recompensa base seja preservada.

### 2.4. Curriculum Learning e Adaptabilidade Dinâmica (`DEFAULT_STAGES` vs `ADAPTIVE_CHAOS_STAGES`)
Treinar diretamente no cenário de caos completo torna o aprendizado do *swing-up* inicial muito ruidoso. O [`CurriculumManager`](src/cartpole_sim/curriculum.py) oferece:
- **`DEFAULT_STAGES` (4 estágios):** progressão de mesa plana (Estágio 0) $\to$ mesa inclinada ±7° (Estágio 1) $\to$ vento/rajadas ±4 m/s (Estágio 2) $\to$ caos completo ±12° + vento ±5.5 m/s + petelecos + *domain randomization* de ±10% (Estágio 3).
- **`ADAPTIVE_CHAOS_STAGES` (`--adaptive-chaos`, 5 estágios):** promove do Estágio 3 (`reward >= 880`) para o **Estágio 4 (Caos Adaptativo Dinâmico)**, onde a inclinação da mesa (±14° a 8°/s) e o vento médio (±6.5 m/s) mudam continuamente **no meio do episódio**, com rajadas mais intensas ($\sigma=2.2$ m/s, $\tau=0.6$ s), petelecos frequentes (`0.35/s`) e ±15% de randomização física.
- **Modo especialista (`--fixed-stage`):** fixa o treino em um único estágio (ex.: Estágio 0 plano limpo em `runs/ppo_simple`) para avaliar o teto de precisão sem perturbações.

### 2.5. Normalização Física Embutida (`norm_obs=False`)
Optou-se por normalizar manualmente todas as features no ambiente (`x_norm`, `v_norm`, `sin`, `cos`, `omega_norm`, `u_prev`) e desabilitar `norm_obs` no `VecNormalize`. Isso torna os arquivos `.zip` de modelo **autossuficientes** para inferência no `play.py`, `evaluate.py` ou eventual exportação para hardware embarcado.

---

## 3. Estado Atual do Repositório

- **Maturidade:** Funcional de ponta a ponta (física, ambiente Gymnasium, treino PPO com curriculum/adaptabilidade, retomada via `curriculum_state.json`, avaliação multi-cenário, estúdio Pygame e testes automatizados).
- **Suíte de Testes:** **18/18 testes passando** (`tests/test_physics.py`, `tests/test_pbrs.py`, `tests/test_env_and_ppo.py`).
- **Modelos Pré-Treinados Versionados (`drive_mode="belt_pulley"`, `use_pbrs=True`, `frame_stack=12`, `gamma=0.99`):**
  - [`runs/ppo_cartpole/best_model.zip`](runs/ppo_cartpole/best_model.zip) — baseline campeão generalista em `DEFAULT_STAGES` (estágios 0–3 estáticos por episódio).
  - [`runs/ppo_simple/best_model.zip`](runs/ppo_simple/best_model.zip) — treinado por **4h (~4,52M passos, `n_envs=1`)** em **Estágio 0 fixo** (`--fixed-stage`, mesa plana sem vento/randomização; recorde de avaliação `1392.22`).
  - [`runs/ppo_chaotic/best_model.zip`](runs/ppo_chaotic/best_model.zip) — treinado por **4h (~11,04M passos, `n_envs=1`)** com `--adaptive-chaos` (Curriculum 0 $\to$ 4 dinâmico intra-episódio; recorde de avaliação no Estágio 4: `944.73`).

### Comparativo Final Pós-Treino de 4h (`scripts/evaluate.py --episodes 5`)
Todos os cenários partem pendurados embaixo (`init_mode="hanging"`, 1000 passos / 20 s, ação determinística):

| Cenário | `ppo_simple` (Sobrev / Rew / Topo% / `\|x\|`) | `ppo_chaotic` (Sobrev / Rew / Topo% / `\|x\|`) | `ppo_cartpole` (Sobrev / Rew / Topo% / `\|x\|`) |
| :--- | :---: | :---: | :---: |
| 1. Plano limpo | **100% / 1392.1 / 92.8% / 0.85 cm** | 100% / 1249.0 / 81.0% / 19.94 cm | 100% / 1377.1 / 91.4% / 3.64 cm |
| 2. Mesa inclinada +10° | 100% / 788.5 / 5.5% / 8.88 cm | 80% / 949.9 / 28.7% / 10.15 cm | **100% / 1340.3 / 91.1% / 17.96 cm** |
| 3. Mesa inclinada -10° | 100% / 646.7 / 3.3% / 11.65 cm | 100% / 881.7 / 24.7% / 11.23 cm | **100% / 1354.2 / 91.5% / 15.22 cm** |
| 4. Vento constante +4.5 m/s | 100% / 1002.9 / 44.1% / 15.32 cm | 100% / 1038.6 / 43.7% / 12.85 cm | **100% / 1373.5 / 92.3% / 5.73 cm** |
| 5. Rampa -8° + Vento +4 m/s + Rajadas | 100% / -72.4 / 0.0% / 20.82 cm | 100% / 893.7 / 20.4% / 10.62 cm | **100% / 1348.2 / 92.1% / 15.30 cm** |
| 6. Caos + Petelecos (6s e 12s) | 80% / 630.8 / 8.4% / 10.20 cm | 100% / 1143.4 / 59.1% / 7.23 cm | **100% / 1271.1 / 76.4% / 10.52 cm** |

### Teste Dinâmico de Adaptabilidade — Estágio 4 (8 episódios determinísticos)
Avaliação com variação contínua intra-episódio de rampa (±14° a 8°/s), vento (±6.5 m/s), rajadas ($\sigma=2.2$ m/s), petelecos (`0.35/s`) e ±15% de randomização paramétrica:

| Modelo | Sobrevivência | Reward Médio | Tempo no Topo | `\|x\|` Médio |
| :--- | :---: | :---: | :---: | :---: |
| `ppo_simple` (4h Estágio 0 fixo) | 37.5% | `306.3 ± 328.9` | 3.7% | 11.26 cm |
| `ppo_chaotic` (4h Estágio 0→4 dinâmico) | **100.0%** | `852.4 ± 112.4` | 24.7% | **11.41 cm** |
| `ppo_cartpole` (Baseline Estágios 0→3) | **100.0%** | **`1029.2 ± 156.2`** | **48.6%** | 12.83 cm |

**Achado de Engenharia (Trade-off Especialização vs. Estresse Extremo com `n_envs=1`):**
- `ppo_simple` atingiu o **teto histórico de precisão em bancada plana** (`1392.1`, erro médio de posição de apenas `0.85 cm` e *swing-up* em `1.42 s`), mas não generaliza fora de 0° (caindo para `3.7%` no topo e `37.5%` de sobrevivência no Estágio 4).
- `ppo_chaotic` manteve **100% de sobrevivência** tanto no Estágio 4 dinâmico quanto nos cenários 1, 3, 4, 5 e 6, porém o longo treinamento sob estresse contínuo extremo (±14° móveis + `0.35` petelecos/s + `±15%` randomização) em `n_envs=1` favoreceu uma política conservadora focada em evitar colisão de fim de curso em vez de centralização agressiva — ficando atrás de `ppo_cartpole` (treinado com mais envs no Estágio 3 antes de saturar `std`). Em máquinas com mais núcleos, treinar `--adaptive-chaos` com `n_envs >= 8` e permanência maior no Estágio 3 (`promote_mean_reward >= 1050`) tende a unir a precisão do `ppo_cartpole` com a imunidade a variações dinâmicas do `ppo_chaotic`.

---

## 4. Limitações Conhecidas e Pontos de Evolução

1. **Transmissão `lead_screw` (TR8x8):** A física e os scripts já suportam `--drive lead_screw`, porém não há um checkpoint pré-treinado versionado para esse modo (e sua velocidade máxima de `0.32` m/s exige uma estratégia de *swing-up* por bombeamento diferente da polia GT2).
2. **Perda de passo (`simulate_step_loss`):** Por padrão está `False` (`enforce_torque_limit=True` apenas satura a aceleração ao torque *pull-out*). Nota-se nos benchmarks que os picos ficam em `102%–129%` — ativar `simulate_step_loss=True` torna o cenário ainda mais rígido para testes *Sim-to-Real* extremos.
3. **Viés de odometria (`odom_bias_std`):** Configurado por padrão em `0.0` m; pode ser habilitado para simular escorregamento de correia ou erro de *homing*.
4. **Série `hist_reward` no `PygameStudioRenderer`:** [`renderer.py`](src/cartpole_sim/renderer.py) acumula o histórico de recompensa por passo em `hist_reward`, mas atualmente renderiza apenas os 4 gráficos de `theta`, `x`, `Φ(s)` e uso de torque.
5. **Ausência de pipeline CI/CD:** Não há workflows automatizados (`.github/workflows/`) configurados no repositório ainda; a verificação depende da execução manual de `uv run pytest`.
