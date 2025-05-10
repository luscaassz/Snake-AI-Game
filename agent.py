import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameIA, Direction, Point
from model import Linear_QNET, QTrainer
from helper import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self): # Inicializa o agente
       self.n_games = 0 # registra o numero de runs jogadas (inicia em 0)
       self.epsilon = 0 # um índice de "aleatoriedade" (inicia em 0)
       self.gamma = 0.9 # taxa de desconto (presente na equação de bellman)
       self.memory = deque(maxlen=MAX_MEMORY) # lista que, caso exceda o limite máximo, remove os elementos a partir da esquerda
       
       self.model = Linear_QNET(15, 256, 3) # camadas de entrada, ocultas e de saída
       self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)

    def get_state(self,game): # obtém o estado atual 

        # obtém os pontos ao redor da posição atual da cabeça (20 pq é o tamanho de cada bloco do grid do jogo)
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)  # ponto à esquerda da cabeça
        point_r = Point(head.x + 20, head.y) # ponto à direita da cabeça
        point_u = Point(head.x, head.y - 20) # ponto acima da cabeça
        point_d = Point(head.x, head.y + 20) # ponto abaixo da cabeça

        # booleanas para determinar em qual direção a cobra está indo atualmente
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN


        '''
        Verificar o estado atual usando as 11 possíveis entradas:

        [
        Perigo -> 3 possibilidades (perigo reto, perigo à direita e perigo à esquerda)

        Direção do movimento -> 4 possibilidades (esquerda, direita, cima, baixo)

        Direção para chegar à fruta -> 4 possibilidades (fruta à esquerda, fruta à direita, fruta acima, fruta abaixo)
        ]

        No final o resultado será uma lista binária, por exemplo:

        A cobrinha está indo para a direita, abaixo dela tem a parede e a fruta está diretamente acima dela, a lista será:

        [0, 1, 0,   # perigo à direita
        0, 1, 0, 0,  # direção da cobra: direita
        0, 0, 1, 0] # fruta acima

        '''
        state = [
            # perigo se continuar seguindo reto
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # perigo se virar para direita
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # perigo se virar para esquerda
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # direção do movimento
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # localização da fruta
            game.food.x < game.head.x, # fruta à esquerda
            game.food.x > game.head.x, # fruta à direita
            game.food.y < game.head.y, # fruta acima
            game.food.y > game.head.y, # fruta abaixo

            any(enemy.x < game.head.x for enemy in game.enemies),
            any(enemy.x > game.head.x for enemy in game.enemies),
            any(enemy.y < game.head.y for enemy in game.enemies),
            any(enemy.y > game.head.y for enemy in game.enemies)
        ]

        return np.array(state, dtype=int) # converte a lista acima em um array numpy para transformar os valores booleanos em 0 e 1

    def remember(self, state, action, reward, next_state, done): # se lembra dos resultados de uma ação tomada
        self.memory.append((state, action, reward, next_state, done)) # como dito no __init__, remove o valor à esquerda se chegar ao valor máximo

    def train_long_memory(self): # treina a memória a longo prazo

        # vamos guardar uma cerca quantidade de jogos anteriores e resultados na variável mini_sample
        if len(self.memory) > BATCH_SIZE:

            # se o número de jogos exceder nosso batch size, pegaremos jogos aleatorios dentro da memoria
            mini_sample = random.sample(self.memory, BATCH_SIZE)  
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample) # zip extrai os valores da tupla

        #utiliza o trainer do modelo
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done): # treina a memória a curto prazo
        
        #utiliza o trainer do modelo
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state): # define a ação que tomará a partir do estado atual
        'movimentos aleatorios: pesquisar "exploration x exploitation in deep learning'

        # traduzindo este bloco: quanto mais jogos jogados, menos aleatório vai ser
        self.epsilon = 80 - self.n_games  
        final_move = [0, 0, 0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            #transformando o estado em um tensor e utilizando para prever a próxima ação
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)

            ''' 
            mas 'prediction' pode ser qualquer valor float, porém queremos transformar em um vetor binário, portanto para 
            evitar que a ação prevista seja algo como [5.0, 2.7, 0.1] vamos pegar o valor máximo e transformá-lo em 1 para
            se tornar [1, 0, 0]
            '''

            move = torch.argmax(prediction).item()  # encontra o índice do maior valor do tensor
            final_move[move] = 1
        
        return final_move

def train():
    plot_scores = []  # lista que registra as pontuações obtidas (será usada para plot)
    plot_mean_scores = [] # lista que registra a média das pontuações
    total_score = 0 # registra a pontuação total feita
    record = 0 # registra a melhor pontuação feita
    agent = Agent() # instância a classe do agente
    game = SnakeGameIA() # instância a classe do jogo

    while True:  
        # obter o estado anterior
        state_old = agent.get_state(game)

        # obter a ação a ser tomada
        final_move = agent.get_action(state_old)

        # realizar a ação e obter novo estado
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # treinar a memória a curto prazo (ou seja, memória da ação anterior)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # ser lembrar dos resultados
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:   #se o jogo acabar
            # treinar a memória a longo prazo (ou seja, a memória de todo o jogo anterior)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # atualizar o record e salvar no modelo
            if score > record:
                record = score
                agent.model.save()

            # printar as informações após o jogo
            print('Game:', agent.n_games, 'Score:', score, 'Record:', record)

            # plotar
            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()