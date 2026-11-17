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
  - *Frame stacking* ($k = 8$ frames a 50 Hz) incluindo comando anterior $u_{t-1}$ para inferência implícita de inclinação da mesa, vento constante/rajadas e impulsos.
  - *Curriculum learning* automático em 4 estágios (plano $\to$ inclinado $\to$ vento $\to$ rajadas + petelecos + *domain randomization*).
- **Estúdio Interativo em Pygame:**
  - Controles ao vivo de inclinação da mesa, vento, rajadas, modo de transmissão (polia vs fuso), petelecos com mouse/teclado, gráficos em tempo real e modo manual ou IA.

## Instalação

```bash
uv sync
```

## Uso Rápido

### Rodar testes unitários e de invariância PBRS
```bash
uv run pytest
```

### Treinar o agente PPO com Curriculum
```bash
uv run python scripts/train_ppo.py --total-timesteps 800000 --n-envs 8
```

### Abrir o Estúdio Interativo Pygame (IA ou Manual)
```bash
# Modo manual (setas / A-D) ou com modelo treinado se existir
uv run python scripts/play.py
```

### Avaliar robustez (inclinação, vento, petelecos, polia vs fuso)
```bash
uv run python scripts/evaluate.py --model runs/ppo_cartpole/best_model.zip
```
