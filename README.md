# Snake-AI-Game

Este projeto implementa o clássico **Jogo da Cobrinha (Snake Game)**, onde a cobrinha aprende a jogar **sozinha** utilizando **Reinforcement Learning (Aprendizado por Reforço)** com **PyTorch** e **Deep Q-Learning**.

O objetivo é treinar um agente para maximizar a pontuação com o tempo, aprendendo a se mover estrategicamente e evitar colisões.

![Tela do Jogo](https://github.com/user-attachments/assets/e0269c25-2ff2-463b-b35f-5ee4eaf3c03c)

---

## 📁 Estrutura do Projeto

```
.
├── agent.py     # Implementação do agente inteligente (Q-Learning)
├── model.py     # Arquitetura da rede neural utilizada pelo agente
├── game.py      # Lógica do jogo (Pygame)
├── helper.py    # Funções auxiliares (plotagens)
└── README.md    # Este arquivo
```

---

## 🧠 Aprendizado por Reforço

Este projeto usa **Deep Q-Learning**, onde o agente:

- Observa o estado atual do ambiente (posição da cobra, da comida, obstáculos).
- Escolhe uma ação com base em uma política (ε-greedy).
- Recebe uma **recompensa** pelo resultado da ação.
- Atualiza sua rede neural para melhorar as decisões futuras.

---

## 🖥️ Pré-requisitos

Para rodar o projeto, você precisará do Python 3.7+ e instalar os seguintes pacotes:

```
pip install pygame numpy matplotlib torch
```

---

## ▶️ Como Executar

1. Clone o repositório:

```
git clone https://github.com/luscaassz/Snake-AI-Game.git
cd Snake-AI-Game
```

2. Execute o treinamento:

```
python agent.py
```

O agente começará a jogar várias partidas e aprender com os erros. Um gráfico será exibido com a evolução da pontuação ao longo dos episódios.

---

## 📊 Resultados

Durante o treinamento, o agente melhora sua pontuação média gradualmente, aprendendo estratégias mais eficientes para encontrar comida e evitar colisões.

Gráfico de pontuação ao longo do tempo:

![Gráfico de Desempenho](https://github.com/user-attachments/assets/ebc28b38-fe48-4916-8600-39afe45414fb)


---

## 📌 Observações

- Este projeto foi feito com fins educacionais, demonstrando o uso prático de **Redes Neurais** em jogos simples.
- Pode ser expandido com melhorias como **Double DQN**, **Prioritized Replay**, **Reward Shaping**, etc.

---

## 📚 Referências
- **Principal referência:**
    - [GitHub da principal referência](https://github.com/patrickloeber/snake-ai-pytorch)
    - [Playlist do Youtube](https://www.youtube.com/playlist?list=PLqnslRFeH2UrDh7vUmJ60YrmWd64mTTKV)
- [Deep Q-Learning](https://en.wikipedia.org/wiki/Q-learning)
- [PyTorch](https://pytorch.org/)
- [Pygame](https://www.pygame.org/)

---

## 🧑‍💻 Autor

Feito por [Lucas Vieira dos Santos Souza](https://github.com/luscaassz) — Estudante de Engenharia de Controle e Automação.
