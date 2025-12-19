
# TCC: Desenvolvimento de um Agente Inteligente para Rocket League
Agente inteligente desenvolvido com o objetivo de ser capaz de jogar de forma autonôma o jogo Rocket League utilizando Proximal Policy Optmization (PPO).

## Treinamento:
Para a realização deste trabalho, foi utilizada a API RLGym, versão 2. Essa API permite criar ambientes de aprendizado por reforço em Python, mais especificamente algoritmos de Proximal Policy Optmization (PPO)

O treinamento foi feito usando o modelo self-play, onde o agente joga contra versões de si mesmo para aprimorar suas habilidades. Em relação à definição dos episódios, ao invés de utilizar uma partida completa como um único episódio, adotou-se como critério de término o gol marcado, além de duas condições adicionais: se a bola ficar sem ser tocada por 30 segundos ou se a partida atingir 5 minutos de duração.

As funções de recompensa foram categorizadas em dois tipos: condicionais, que são atribuídas quando uma condição específica é atendida, e contínuas, que são atribuídas a cada passo do agente. Cada recompensa é multiplicada por um peso determinado, resultando na recompensa final que o agente recebe a cada interação.

O treinamento foi estruturado em 9 etapas, nas quais a tabela de recompensas foi constantemente aprimorada e refinada. No final do processo, o agente foi treinado por 500 milhões de passos e utilizou 18 funções de recompensa.

## Resultados:
Para testar o desempenho do agente dentro do jogo, foi utilizado o framework RLBot, que estabelece a conexão com o jogo através de uma API e se comunica com o agente via sockets. A versão utilizada do framework foi a 5, por ser a única compatível com a versão 2 da RLGym utilizada neste trabalho.

Para avaliar o desempenho do agente, ele foi submetido a cinco partidas contra cada nível do bot padrão. Dado que o bot tem 4 níveis, logo o agente jogou 20 partidas no total. Contra os três primeiros níveis, o agente obteve uma taxa de vitória de 100%, enquanto que no último nível, obteve uma taxa de vitória de 80%.