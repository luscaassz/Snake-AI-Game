import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Estruturando a rede neural
class Linear_QNET(nn.Module):

    # Inicializando com as camadas de entrada, ocultas e de saída
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        '''
        Declaramos as ligações entre as camadas, como implementaremos um modelo feed forward,
        as ligações são simples. A primeira é entre as camadas de entrada e as camadas ocultas,
        A segunda é entre as camadas ocultas e as de saída.
        '''

        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # implementa a camada linear 
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    # cria e salva o arquivo do modelo
    def save(self, file_name='model.pth'):  
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


# Classe do Trainer
class QTrainer:

    # inicializando com o modelo, o learning rate e a taxa de desconto
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model

        # escolher um optmizer (escolhi o Adam pq o tutorial que eu vi escolheu esse :D)
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)

        # definir a função de perda (a partir de deduções vemos que é a diferença do quadrado do Q value, 
        # mas o pytorch tem uma função pronta que já calcula)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):

        '''
        transformando os valores em tensores para ser possível dar reshape nas dimensões
        pois no agent.py temos a função train_short_memory que chama esta função train_step com parâmetros de 1 dimensão e a função
        train_long_memory, que chama com parâmetros de várias dimensões
        ''' 
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # redimensionar para dimensão 0 (somente um valor)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)

            done = (done, ) # convertendo em uma tupla com só um valor
        
        # Obter os Q values previstos a partir do estado atual
        pred = self.model(state)

        target = pred.clone() # clonar o valor previsto para armazenar ele

        # Utilizando equação de Bellman: Calcular o new_Q = R + y*max(next_predicted_Q)  -> somente se "not done"
        for index in range(len(done)):
            Q_new = reward[index]

            if not done[index]:
                Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))

            target[index][torch.argmax(action).item()] = Q_new

        # Função para "limpar" o gradiente (melhorar performance)
        self.optimizer.zero_grad()

        # Calculando a perda usando o quadrado da diferença inicializado no __init__
        # target = Q novo; pred = Q atual
        loss = self.criterion(target, pred) 
        

        # Aplicando backpropagation
        loss.backward() 
        self.optimizer.step()
        