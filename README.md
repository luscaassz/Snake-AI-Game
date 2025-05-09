# Snake-AI-Game

Este projeto implementa o clássico **Jogo da Cobrinha (Snake Game)**, onde a cobrinha aprende a jogar **sozinha** utilizando **Reinforcement Learning (Aprendizado por Reforço)** com **Deep Q-Learning**.

O objetivo é treinar um agente para maximizar a pontuação com o tempo, aprendendo a se mover estrategicamente e evitar colisões.

---

## 📁 Estrutura do Projeto

```bash
.
├── agent.py     # Implementação do agente inteligente (Q-Learning)
├── model.py     # Arquitetura da rede neural utilizada pelo agente
├── game.py      # Lógica do jogo (Pygame)
├── helper.py    # Funções auxiliares (memória, plotagens, etc.)
└── README.md    # Este arquivo
