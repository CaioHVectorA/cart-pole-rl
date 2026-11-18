# AGENTS.md — Guia Operacional para Agentes de Código

Este documento orienta agentes de desenvolvimento que trabalham no repositório **cart-pole-rl** (`cartpole-sim`). Leia também [`CONTEXT.md`](CONTEXT.md) para entender as decisões de engenharia e [`PROTOCOL.md`](PROTOCOL.md) para os contratos matemáticos da física, PBRS e RL.

---

## 1. Ambiente e Comandos Essenciais

O projeto usa **Python >= 3.11** gerenciado via **`uv`** (com `torch` CPU definido em `pyproject.toml`) e layout `src/`.

```bash
# Sincronizar ambiente virtual (.venv) e dependências
uv sync

# Executar toda a suíte de testes (18 testes: física, invariância PBRS, Gymnasium, adaptabilidade, smoke test PPO)
uv run pytest -v

# Avaliar modelos salvos nos 6 cenários de robustez
uv run python scripts/evaluate.py --model runs/ppo_cartpole/best_model.zip
uv run python scripts/evaluate.py --model runs/ppo_simple/best_model.zip
uv run python scripts/evaluate.py --model runs/ppo_chaotic/best_model.zip

# Treinar PPO com curriculum padrão (belt_pulley, 600k passos, 8 envs)
uv run python scripts/train_ppo.py --total-timesteps 600000 --n-envs 8

# Treinar PPO especialista em condição simples (estágio 0 fixo: mesa plana, sem vento)
uv run python scripts/train_ppo.py --output-dir runs/ppo_simple --initial-stage 0 --fixed-stage --max-minutes 240

# Treinar PPO com caos completo e adaptabilidade dinâmica intra-episódio (Estágio 4)
uv run python scripts/train_ppo.py --output-dir runs/ppo_chaotic --adaptive-chaos --max-minutes 240

# Treinar com fuso TR8x8
uv run python scripts/train_ppo.py --drive lead_screw --output-dir runs/ppo_lead_screw

# Abrir estúdio interativo Pygame (requer display gráfico)
uv run python scripts/play.py --model runs/ppo_cartpole/best_model.zip
```

> **Ambiente headless / CI:** Testes que instanciam `PygameStudioRenderer` em modo `rgb_array` definem `SDL_VIDEODRIVER=dummy` (veja `tests/test_env_and_ppo.py`). Ao testar renderização sem servidor X, preserve esse padrão.

---

## 2. Mapa de Arquitetura e Responsabilidades

```text
src/cartpole_sim/
├── config.py       # Dataclasses de configuração (SI), NEMA 17, polia/fuso, sensor, distúrbios (estáticos e dinâmicos), reward
├── physics.py      # Motor físico RK4 desacoplado de RL/UI (CartPolePhysics, PhysicsState, StepMetrics)
├── env.py          # Wrapper Gymnasium (CartPoleStepperEnv), sensorização realista, PBRS e reward base
├── curriculum.py   # Estágios de dificuldade progressiva (DEFAULT_STAGES, ADAPTIVE_CHAOS_STAGES, CurriculumManager)
├── train.py        # Pipeline SB3 PPO, VecNormalize(norm_obs=False), callbacks de curriculum/telemetria/tempo e retomada
└── renderer.py     # Estúdio Pygame interativo (PygameStudioRenderer) desacoplado da física e do treino

scripts/
├── train_ppo.py    # CLI de treinamento e retomada (--resume, --max-minutes, --fixed-stage, --adaptive-chaos)
├── evaluate.py     # Benchmark determinístico em 6 cenários adversos
└── play.py         # Loop interativo em tempo real (IA PPO ou Manual)

tests/
├── test_physics.py     # Conservação de energia, período teórico, rampa, vento, peteleco, transmissão/folga, adaptação dinâmica
├── test_pbrs.py        # Soma telescópica PBRS, terminal=0, propriedades de Phi, wrapping do encoder, delay
└── test_env_and_ppo.py # gymnasium.utils.env_checker, eval_mode, curriculum, --fixed-stage/--adaptive-chaos, rgb_array headless, smoke PPO

runs/{ppo_cartpole,ppo_simple,ppo_chaotic}/
├── best_model.zip  # Checkpoints campeões versionados (ver .gitignore)
└── run_config.json # Metadados de reprodutibilidade (drive_mode, frame_stack, adaptive_chaos, fixed_stage, etc.)
```

### Fronteiras de Desacoplamento (Não Violar)
1. **`physics.py` não importa `gymnasium`, `torch`, `stable_baselines3` nem `pygame`.** A física é pura (`math` + `numpy`) e pode rodar isolada.
2. **`env.py` importa `renderer.py` apenas de forma tardia (*lazy import*)** dentro de `CartPoleStepperEnv.render()`, evitando inicializar Pygame durante treinos headless.
3. **`config.py` é a fonte única de verdade** para constantes físicas, dimensões de observação (`SensorConfig.frame_stack`), descontos (`RewardConfig.gamma`) e limites mecânicos.

---

## 3. Invariantes Críticas do Projeto

Antes de alterar qualquer arquivo em `src/cartpole_sim/`, verifique se você não quebrou nenhuma das invariantes abaixo:

1. **Separação entre Referencial Real e Encoder (`physics.py`, `env.py`):**
   - `PhysicsState.theta` é contínuo (sem *wrap*) no referencial **vertical gravitacional** ($\theta = 0$ no topo, $\theta = \pi$ pendurado embaixo).
   - `PhysicsState.theta_encoder = theta + alpha` é o ângulo no referencial do chassi inclinado.
   - **Nunca** exponha `theta` verdadeiro, `alpha`, `wind_mean`, `wind_gust` ou `flick_torque` no vetor de observação de `CartPoleStepperEnv`. O agente deve inferir perturbações (incluindo variações dinâmicas intra-episódio de rampa e vento) apenas via histórico de sensores (`frame_stack`).
   - **Nunca** faça *wrap* antes da derivada finita no encoder (`_read_encoder_frame`), para não gerar picos artificiais de velocidade angular ao cruzar $\pm\pi$.

2. **Regras de PBRS — Ng, Harada & Russell (1999) (`env.py`):**
   - O fator de desconto `RewardConfig.gamma` usado no cálculo de `F(s, a, s') = gamma * phi_next - phi_curr` **deve ser exatamente igual** ao `gamma` passado ao `PPO` e ao `VecNormalize` em `train.py`.
   - Em estados terminais (`terminated == True`, i.e., `abs(s.x) >= rail_half_length`), `phi_next` **deve ser estritamente `0.0`** (nunca `compute_potential(s)`), evitando bônus de suicídio.
   - `compute_potential(s)` deve ser $\ge 0$ dentro da pista útil.

3. **Contrato de Observação e Deploy Direto (`env.py`, `train.py`):**
   - As observações já saem normalizadas fisicamente de `CartPoleStepperEnv`.
   - `VecNormalize` deve manter `norm_obs=False`. Alterar `norm_obs=True` quebraria o carregamento direto de `best_model.zip` em `scripts/play.py` e `scripts/evaluate.py` (que não carregam `vec_normalize.pkl`).
   - Mudar `FRAME_DIM`, `SensorConfig.frame_stack` ou a ordem das features invalida modelos `.zip` previamente treinados (como `runs/*/best_model.zip`).

4. **Sincronia entre `train_env` e `eval_env` no Curriculum e Retomada (`train.py`):**
   - `CurriculumEvalCallback` propaga o `DisturbanceConfig` do estágio atual via `env_method("set_disturbance_config", dist)` para ambos `training_env` e `eval_env` e reinicia `self.best_mean_reward = -np.inf` a cada promoção de estágio.
   - A cada avaliação (`_save_checkpoint_state()`), o callback persiste `curriculum_state.json`, `last_model.zip` e `vec_normalize.pkl`, permitindo retomada fiel via `--resume` sem perder o estágio, passos no estágio ou referência de melhor reward.

---

## 4. Convenções de Código, Testes e Git

- **Estilo e Tipagem:** Use `from __future__ import annotations`, type hints completos, dataclasses para configurações e métricas, e docstrings/comentários em português (PT-BR) preservando os comentários existentes.
- **Reprodutibilidade em `CartPolePhysics`:** Ao resetar `CartPoleStepperEnv`, `self.physics.rng` é vinculado a `self.np_random` do Gymnasium para que `env.reset(seed=...)` torne parâmetros e distúrbios determinísticos.
- **Critério de Aceite para Alterações:**
  - Qualquer mudança na física exige rodar `uv run pytest` e confirmar todos os 18 testes passando.
  - Mudanças em `RewardConfig`, `SensorConfig` ou `CurriculumManager` devem manter `test_pbrs.py` e `test_env_and_ppo.py` verdes.
  - Caso altere `SensorConfig.frame_stack` ou o espaço de observação/ação e re-treine algum modelo campeão, atualize o respectivo `run_config.json` e documente a incompatibilidade com checkpoints anteriores.
- **Política de Versionamento (`.gitignore`):** `runs/*` é ignorado pelo Git, com exceção explícita de `best_model.zip` e `run_config.json` dentro de `runs/ppo_cartpole/`, `runs/ppo_simple/` e `runs/ppo_chaotic/`. Não adicione logs do TensorBoard (`tb/`), `eval_logs/`, `curriculum_state.json`, `vec_normalize.pkl` ou checkpoints intermediários (`last_model.zip`, `stage_*_mastered.zip`) ao controle de versão.
