# Módulo de treinamento do modelo

# Imports
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from modelo.pixelcnn import PixelCNN
from tqdm import tqdm

print("\nMódulo de Treinamento do Modelo!\n")

# Hiperparâmetros
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
NUM_EPOCHS = 50

# Pasta para salvar o modelo treinado
SAVE_DIR = "modelo_salvo/"
os.makedirs(SAVE_DIR, exist_ok = True)

# Carrega dados de treino e validação
train_dataset = datasets.MNIST(root = "dados/", train = True, download = True, transform = transforms.ToTensor())
valid_dataset = datasets.MNIST(root = "dados/", train = False, download = True, transform = transforms.ToTensor())

print("Número de Imagens no Dataset de Treino:", len(train_dataset))
print("Número de Imagens no Dataset de Validação:", len(valid_dataset))

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Cria os dataloaders
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
valid_loader = DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle = False)

# Cria o modelo e envia para o device
model = PixelCNN()
model.to(device)

# Otimizador
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

# Função de erro
# BCE é a abreviação de Binary Cross Entropy. Esta função de perda é comumente usada em problemas de classificação binária.
# BCELoss = - [y * log(p) + (1 - y) * log(1 - p)]
bce_loss = nn.BCELoss()

print("\nIniciando o Treinamento do Modelo!\n")

# Loop pelo número de epochs
for epoch in range(NUM_EPOCHS):
    
    # Inicializa o erro médio
    mean_loss = 0
    
    # Loop pelo dataloader
    for x, _ in tqdm(train_loader):
        
        # Envia os dados para a memória do device
        x = x.to(device)
        
        # Aqui estamos criando um novo tensor que tem o mesmo formato que o tensor x. Para cada posição no tensor x, 
        # se o valor em x for 0.0, então o valor correspondente no novo tensor será também 0.0, caso contrário será 1.0.
        # Esta linha de código está substituindo todos os valores não-zero em x por 1.0 e deixando os zeros como estão. 
        # Isto é, está convertendo x para uma máscara binária, onde os valores são 0.0 onde x era 0.0 e 1.0 em todos os outros lugares.
        x = torch.where(x == 0., 0., 1.)

        # Zera os gradientes
        optimizer.zero_grad()
        
        # Previsão do modelo
        out = model(x)
        
        # Cálculo do erro do modelo
        loss = bce_loss(out, x)
        
        # Esta linha calcula o gradiente da função de perda em relação a cada parâmetro do modelo.
        loss.backward()
        
        # Esta linha faz o otimizador ajustar cada parâmetro do modelo usando o gradiente calculado pela função backward().
        optimizer.step()

        # Calcula o erro médio
        mean_loss += loss.item()
    
    print(f"Epoch: {epoch:>3d} Erro do Modelo: {mean_loss / len(train_loader):>6f}")

    # Salva o modelo a cada 10 épocas
    if epoch % 10 == 9:
        save_path = os.path.join(SAVE_DIR, "modelo_pixelcnn{}.pt".format(epoch))
        torch.save(model.state_dict(), save_path)

print("\nTreinamento Concluído!\n")

