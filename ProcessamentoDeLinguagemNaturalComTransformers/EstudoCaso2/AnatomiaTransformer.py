# Anatomia de Um Modelo Transformer com PyTorch

# Esse código define uma classe Transformer que estende a classe nn.Module do PyTorch e implementa um modelo transformador 
# para processamento de linguagem natural. 

# O método __init__() inicializa o modelo do transformador e define as várias camadas do modelo. 

# O método forward() define como a sequência de entrada é transformada pelo modelo para gerar a sequência de saída.

# Imports
import torch
from torch import nn

# Classe
class Transformer(nn.Module):

  # Método construtor
  def __init__(self, vocab_size, embedding_dim, n_heads, n_layers, dropout):
    
    # Inicializa o construtor da classe mãe (nn..Module)
    super().__init__()

    # Inicializa atributos
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.dropout = dropout

    # Define a camada de embedding que transformará a sequência de entrada em uma sequência de vetores de dimensão fixa
    self.embedding = nn.Embedding(vocab_size, embedding_dim)

    # Define o mecanismo de auto-atenção multi-headed
    self.attention = nn.MultiheadAttention(embedding_dim, n_heads, dropout = dropout)

    # Define a rede neural feed-forward que será usada para gerar a sequência de saída a partir da sequência de entrada
    self.feed_forward = nn.Sequential(
        nn.Linear(embedding_dim, embedding_dim),
        nn.ReLU(),
        nn.Linear(embedding_dim, embedding_dim)
    )

    # Define a camada de saída final que transformará a sequência de saída na forma de saída desejada
    self.out = nn.Linear(embedding_dim, vocab_size)

  # Método forward
  def forward(self, x):
    
    # Aplica a camada de embedding à sequência de entrada
    x = self.embedding(x)

    # Aplica o mecanismo multi-headed self-attention 
    x = self.attention(x)

    # Aplica o método feed-forward
    x = self.feed_forward(x)

    # Aplica a camada final
    x = self.out(x)

    return x


# Para criar um modelo transformador, você pode instanciar a classe Transformer e especificar os hiperparâmetros do modelo, 
# como o tamanho do vocabulário, a dimensão da embedding, o número de "cabeças" de atenção, o número de camadas e a taxa de dropout. 


# Por exemplo:
model = Transformer(vocab_size = 1000, embedding_dim = 32, n_heads = 4, n_layers = 2, dropout = 0.5)


# Isso cria um modelo transformador com um tamanho de vocabulário de 1000, uma dimensão de embedding de 32, 
# 4 cabeças de atenção, 2 camadas ocultas  e uma taxa de desistência (dropout) de 0,5. 

# Você pode usar esse modelo para processar sequências de entrada e gerar sequências de saída usando o método forward().



