# Módulo de Geração de Imagens com o Modelo Treinado

# Imports
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from modelo.pixelcnn import PixelCNN
from tqdm import tqdm

# Pasta para salvar as imagens geradas
SAVE_DIR = "resultado/"
os.makedirs(SAVE_DIR, exist_ok = True)

# Caminho para o modelo treinado
MODEL_PATH = "modelo_salvo/modelo_pixelcnn49.pt"

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instância do modelo
model = PixelCNN()

# Carrega o modelo treinado e manda para o device
model.load_state_dict(torch.load(MODEL_PATH, map_location = device))
model.to(device)

# Imagem gerada preenchida com zeros no formato numpy (apenas para criar a estrutura de dados).
# São 16 matrizes de 1x28x28, ou seja, 16 imagens de tamanho 28x28 com um único canal de cor (escala de cinza).
imagem_gerada = np.zeros((16, 1, 28, 28), dtype = np.float32)

# Converte para o formato pytorch e manda para o device
imagem_gerada = torch.from_numpy(imagem_gerada)
imagem_gerada = imagem_gerada.to(device)

# Usa o modelo para gerar a imagem. O modelo faz a previsão de cada pixel.
with torch.no_grad():

    # Loops aninhados que percorrem cada pixel da imagem a ser gerada. Neste caso, a imagem gerada é de 28 por 28 pixels.
    for h in tqdm(range(28)):
        for w in range(28):

            # O modelo treinado é usado para prever o próximo pixel. 
            previsao = model(imagem_gerada)

            # A função torch.bernoulli recebe uma probabilidade e retorna 0 ou 1 com base nessa probabilidade.
            pixel_previsto = torch.bernoulli(previsao[:, :, h, w])

            # O pixel gerado é adicionado à imagem gerada.
            imagem_gerada[:, :, h, w] = pixel_previsto

# Plot da imagem
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(imagem_gerada[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.axis("off")

plt.suptitle("Imagens Geradas Pelo Modelo PixelCNN")
plt.savefig(os.path.join(SAVE_DIR, "imagem_gerada_pixelcnn.png"))


