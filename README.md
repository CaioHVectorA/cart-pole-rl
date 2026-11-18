# Inverted Pendulum RL Suite (Stepper + Encoder + PBRS)

Simulador físico de alta fidelidade (RK4) e suíte de treinamento em **Reinforcement Learning (PPO + Potential-Based Reward Shaping + Frame Stacking)** para controle e *swing-up* de pêndulo invertido rígido acionado por motor de passo (correia/polia GT2 ou barra roscada TR8x8) com leitura odometria + encoder rotativo.

## Destaques Técnicos

- **Física Fidedigna (Sim-to-Real):**
  - Pêndulo físico real (haste + massa posicionável) com inércia exata pelo teorema dos eixos paralelos.
  - Trilho inclinável ($\alpha$) com separação explícita entre a **vertical gravitacional** (física/recompensa) e o **ângulo medido pelo encoder no chassi** ($\theta_{\text{enc}} = \theta + \alpha$).
  - Motor de passo NEMA 17 com rampa de aceleração, inércia do rotor refletida, curva de torque *pull-out* vs RPM e folga (*backlash*) configurável.
  - Atrito viscoso + atrito seco de Coulomb suavizado no mancal do encoder.
  - Vento horizontal com arrasto quadrático relativo, rajadas contínuas (processo Ornstein-Uhlenbeck) e "petelecos" (impulsos de duração finita).
  - Quantização do encoder (ex: 600 PPR $\times 4 = 2400$ ticks/volta), derivada finita com filtro EMA e latência de controle.
- **PBRS Rigoroso (Ng, Harada & Russell, 1999):**
  - Shaping $F(s, a, s') = \gamma\Phi(s') - \Phi(s)$ com casamento exato de $\gamma$, $\Phi(s_{\text{terminal}}) = 0$ e $\Phi(s) \ge 0$ dentro da pista.
- **Observabilidade & Adaptação:**
  - *Frame stacking* ($k = 12$ frames a 50 Hz, vetor de 72 dimensões) incluindo comando anterior $u_{t-1}$ para inferência implícita de inclinação da mesa, vento constante/rajadas e impulsos.
  - *Curriculum learning* automático (`DEFAULT_STAGES` em 4 estágios e `ADAPTIVE_CHAOS_STAGES` com 5º estágio de variação contínua intra-episódio de rampa ±14° e vento ±6.5 m/s + *domain randomization* de até ±15%).
- **Estúdio Interativo em Pygame:**
  - Controles ao vivo de inclinação da mesa, vento, rajadas, modo de transmissão (polia vs fuso), petelecos com mouse/teclado, gráficos em tempo real e modo manual ou IA.

## Documentação do Repositório

- [`AGENTS.md`](AGENTS.md) — Guia operacional para agentes de código: comandos, arquitetura, invariantes de física/PBRS e regras de teste/versionamento.
- [`CONTEXT.md`](CONTEXT.md) — Contexto de engenharia, motivação *Sim-to-Real*, decisões de projeto, resultados comparativos do treino de 4h e pontos de atenção.
- [`PROTOCOL.md`](PROTOCOL.md) — Especificação matemática e contratual da física (RK4, NEMA 17, arrasto), espaço de observação/ação, PBRS, *curriculum* e protocolo de treino/avaliação.

## Instalação

```bash
uv sync
```

## Uso Rápido

### Rodar testes unitários e de invariância PBRS (18 testes)
```bash
uv run pytest
```

### Treinar o agente PPO com Curriculum
```bash
# Curriculum padrão (estágios 0 a 3)
uv run python scripts/train_ppo.py --total-timesteps 600000 --n-envs 8

# Estágio 0 fixo (mesa reta, sem vento) ou Caos Adaptativo Dinâmico (estágio 4)
uv run python scripts/train_ppo.py --output-dir runs/ppo_simple --initial-stage 0 --fixed-stage --max-minutes 240
uv run python scripts/train_ppo.py --output-dir runs/ppo_chaotic --adaptive-chaos --max-minutes 240
```

### Abrir o Estúdio Interativo Pygame (IA ou Manual)
```bash
# Modo manual (setas / A-D) ou com um dos modelos pré-treinados (ppo_cartpole, ppo_simple, ppo_chaotic)
uv run python scripts/play.py --model runs/ppo_cartpole/best_model.zip
```

### Avaliar robustez (inclinação, vento, petelecos, polia vs fuso)
```bash
uv run python scripts/evaluate.py --model runs/ppo_cartpole/best_model.zip
uv run python scripts/evaluate.py --model runs/ppo_simple/best_model.zip
uv run python scripts/evaluate.py --model runs/ppo_chaotic/best_model.zip
```

