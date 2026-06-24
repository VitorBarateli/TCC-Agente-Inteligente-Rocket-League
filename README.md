# Desenvolvimento de um Agente Inteligente para Rocket League (PPO)
Este repositório contém o código-fonte de um agente inteligente autônomo capaz de jogar Rocket League de forma competitiva. Desenvolvido como um Trabalho de Conclusão de Curso (TCC), o projeto utiliza Aprendizado por Reforço Profundo (*Deep Reinforcement Learning*) fundamentado no algoritmo **Proximal Policy Optimization (PPO)**.

---

## 📌 Sumário
- [Visão Geral do Treinamento](#-visão-geral-do-treinamento)
- [Hiperparâmetros da Rede Neural](#%EF%B8%8F-hiperparâmetros-da-rede-neural)
- [Modelagem de Recompensas (Reward Engineering)](#-modelagem-de-recompensas-reward-engineering)
- [Estrutura de Episódios](#-estrutura-de-episódios)
- [Resultados e Validação](#-resultados-e-validação)
- [Instalação e Execução](#-instalação-e-execução)

---

## 🧠 Visão Geral do Treinamento

O agente foi treinado sob o paradigma de **Self-Play**, onde o modelo interage contra cópias de si mesmo em tempo real dentro do simulador, garantindo um ambiente de aprendizado dinâmico e auto-adaptativo. 

* **Framework de Conexão:** RLGym v2 (interpolação Python-Jogo via RocketSim Engine).
* **Volume de Treinamento:** 500 milhões de timesteps distribuidos e otimizados em 9 macro-etapas de refinamento evolutivo.

---

## 🛠️ Hiperparâmetros da Rede Neural

A arquitetura do modelo separa as redes de Política (Ator) e Crítico, utilizando estruturas densas e profundas para capturar a física complexa do jogo:

| Parâmetro | Configuração Base |
| :--- | :--- |
| **Camadas da Política (Policy)** | `[1024, 1024, 512, 512]` |
| **Camadas do Crítico (Critic)** | `[1024, 1024, 512, 512]` |
| **PPO Batch Size** | `100.000` |
| **PPO Minibatch Size** | `50.000` |
| **Learning Rate (Policy/Critic)** | `1e-4` |
| **PPO Epochs** | `2` |
| **Entropy Coefficient (coef_ent)** | `0.01` |
| **Buffer de Experiência** | `300.000` |

---

## 📊 Modelagem de Recompensas (Reward Engineering)

A função de recompensa final do agente é uma composição ponderada de componentes **contínuos** (atribuídos a cada tick) e **condicionais** (atribuídos por eventos macro). O vetor de recompensas final refinado consolida comportamentos como:

* **Mecânicas de Vetor de Movimento:** `SpeedTowardBallReward` (recompensa por mover-se linearmente em direção à bola) e `VelocityBallToGoalReward` (força do chute direcionada ao gol adversário).
* **Controle de Recursos:** `AdvancedCollectBoostReward` (coleta eficiente de boost ponderada pela distância do oponente) e `WasteBoostAtMaxSpeedPunishment` (penalização por gastar boost quando já está em velocidade supersônica).
* **Mecânicas Aéreas:** `AerialTouchReward` (recompensa proporcional ao tempo de voo e altura do toque) associado ao `InAirReward`.
* **Kickoff Avançado (`AdvancedKickoffReward`):** Penaliza desvios laterais na linha reta em direção ao centro do campo e bonifica o primeiro toque na bola.

```python
# Composição interna do combinador de recompensas (CombinedReward)
reward_fn = CombinedReward(
    (AerialTouchReward(), 15),
    (AdvancedTouchForceReward(), 30),
    (AdvancedCollectBoostReward(), 25),
    (VelocityBallToGoalReward(), 20),
    (BallCloserToGoalReward(), 15),
    (AdvancedKickoffReward(), 40),
    # ... (demais pesos configurados no ambiente)
)
```
## ⏱️ Estrutura de Episódios

Para acelerar a convergência do gradiente, a simulação descarta a estrutura cronológica de uma partida padrão (5 minutos fixos) e adota condições dinâmicas de **Término (Termination)** e **Truncamento (Truncation)**:

1. **Gol Marcado (`GoalCondition`):** Finaliza o episódio imediatamente.
2. **Inatividade (`NoTouchTimeoutCondition`):** Reseta o cenário se a bola não sofrer nenhuma alteração ou toque por 30 segundos consecutivos.
3. **Timeout Global (`TimeoutCondition`):** Limite físico máximo de 5 minutos (300 segundos) por simulação.

---

## 🏆 Resultados e Validação

Para validação em ambiente real de jogo, o modelo gerado foi injetado através do framework **RLBot v5**, que se comunica via sockets locais com o cliente do Rocket League.

O agente foi submetido a uma bateria de testes contra os quatro níveis de bots nativos do jogo (fácil, médio, difícil e All-Star), totalizando 20 partidas competitivas (5 jogos por nível):

```text
📊 TAXA DE VITÓRIA DO AGENTE INTELIGENTE:
├── Contra Nível 1 (Fácil):   100% [████████████████████]
├── Contra Nível 2 (Médio):   100% [████████████████████]
├── Contra Nível 3 (Difícil): 100% [████████████████████]
└── Contra Nível 4 (All-Star): 80% [████████████████░░░░]
```
## 💻 Instalação e Execução

### Pré-requisitos
* Python 3.9 ou 3.10
* Rocket League (instalado via Epic Games ou Steam)
* RLBot v5

### Inicialização do Treinamento

1. **Instale as dependências básicas:**
   Certifique-se de ter o ambiente virtual configurado e instale os pacotes principais do ecossistema de aprendizado por reforço:
   ```bash
   pip install rlgym[all]
   ```
2. **Inicie o script de treinamento:**
   Execute o arquivo principal para instanciar os subprocessos parallelizados e iniciar a inferência do algoritmo PPO:
   ```bash
   python main.py
   ```
