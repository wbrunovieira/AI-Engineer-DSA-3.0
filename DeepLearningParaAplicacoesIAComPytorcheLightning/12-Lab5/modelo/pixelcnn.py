# Modelo

# pip install torchsummary

# Imports
import torch
import torch.nn as nn
import torchsummary

# Essa é uma classe personalizada para implementar uma operação convolucional que é "mascarada". Isso é específico para o PixelCNN 
# e é usado para garantir que a predição para um pixel em uma imagem não seja influenciada por qualquer informação dos pixels "futuros". 
# A máscara é criada de tal forma que as convoluções podem cobrir apenas pixels que já foram vistos. Dois tipos de máscaras são usados aqui, 'A' e 'B'. 
# A máscara 'A' é usada para a primeira camada, onde o pixel central é definido como 0 para garantir que cada pixel não possa se "ver". 
# A máscara 'B' é usada para as camadas restantes, onde o pixel central pode se ver.
class MaskedConvolution(nn.Conv2d):
    
    def __init__(self, mask_type, *args, **kwargs):
        
        super(MaskedConvolution, self).__init__(*args, **kwargs)
        
        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        yc, xc = self.weight.data.size()[-2] // 2, self.weight.data.size()[-1] // 2
        
        self.mask[..., :yc, :] = 1.
        self.mask[..., yc, :xc+1] = 1.

        if mask_type == 'A':
            self.mask[..., yc, xc] = 0.

    def forward(self, x):
        
        self.weight.data *= self.mask
        
        out = super(MaskedConvolution, self).forward(x)
        
        return out


# Essa é uma classe que implementa um bloco residual, que é um componente comum em muitas redes neurais convolucionais. 
# O bloco residual ajuda a resolver o problema do desaparecimento do gradiente ao permitir que ativações "saltem" camadas.
class ResidualBlock(nn.Module):
    
    def __init__(self, in_channel):
        
        super().__init__()

        self.layers = nn.Sequential(nn.Conv2d(in_channels = in_channel, 
                                              out_channels = in_channel // 2, 
                                              kernel_size = 1, 
                                              padding = "same", 
                                              bias = False),
                                    
                                    nn.ReLU(inplace = True),
                                    
                                    MaskedConvolution(mask_type = 'B', 
                                                      in_channels = in_channel // 2, 
                                                      out_channels = in_channel // 2, 
                                                      kernel_size = 3, 
                                                      padding = "same", 
                                                      bias = False),
                                    
                                    nn.ReLU(inplace = True),
                                    
                                    nn.Conv2d(in_channels = in_channel // 2, 
                                              out_channels = in_channel, 
                                              kernel_size = 1, 
                                              padding = "same", 
                                              bias = False),
                                    
                                    nn.ReLU(inplace = True)
        )

    def forward(self, inputs):
        
        x = self.layers(inputs)
        
        out = x + inputs
        
        return out


# Esta é a classe que implementa a própria PixelCNN. A arquitetura do modelo consiste em uma camada de entrada 
# que usa a convolução mascarada do tipo 'A', seguida por várias camadas (num_layers) de blocos residuais que usam convoluções mascaradas do tipo 'B', 
# e finalmente uma camada de saída. O modelo recebe como entrada uma imagem (ou um lote de imagens) e produz uma nova imagem do mesmo tamanho.
class PixelCNN(nn.Module):
    
    def __init__(self, num_filters = 128, num_layers = 5):
        
        super(PixelCNN, self).__init__()

        self.input_conv = nn.Sequential(MaskedConvolution(mask_type = 'A', 
                                                          in_channels = 1, 
                                                          out_channels = num_filters, 
                                                          kernel_size = 7, 
                                                          padding = "same", 
                                                          bias = False),
                                        
                                        nn.ReLU(True)
        )

        layers = [ResidualBlock(in_channel = num_filters) for _ in range(num_layers)]
        
        self.layers = nn.Sequential(*layers)

        self.last_conv = nn.Sequential(MaskedConvolution(mask_type = 'B', 
                                                         in_channels = num_filters, 
                                                         out_channels = num_filters, 
                                                         kernel_size = 1),
                                       
                                       nn.ReLU(True),
                                       
                                       MaskedConvolution(mask_type = 'B', 
                                                         in_channels = num_filters, 
                                                         out_channels = num_filters, 
                                                         kernel_size = 1),
                                       
                                       nn.ReLU(True)
        )

        self.out = nn.Sequential(nn.Conv2d(in_channels = num_filters, out_channels = 1, kernel_size = 1),
                                 nn.Sigmoid()
        )

    def forward(self, inputs):
        
        x = self.input_conv(inputs)
        
        x = self.layers(x)
        
        x = self.last_conv(x)
        
        out = self.out(x)
        
        return out


# Nesta parte do código, duas instâncias da classe MaskedConvolution são criadas, uma com a máscara do tipo 'A' e outra com a máscara do tipo 'B'. 
# Ele imprime as máscaras dessas convoluções. Em seguida, uma instância do modelo PixelCNN é criada e uma visão geral do modelo é impressa usando 
# a função summary() da biblioteca torchsummary.
if __name__ == "__main__":
    
    A_mask_conv = MaskedConvolution('A', in_channels = 1, out_channels = 3, kernel_size = 7, padding = "same", bias = False)
    
    B_mask_conv = MaskedConvolution('B', in_channels = 1, out_channels = 3, kernel_size = 7, padding = "same", bias = False)

    print("Mask A", A_mask_conv.mask)
    print("Mask B", B_mask_conv.mask)

    model = PixelCNN()

    print(torchsummary.summary(model, (1, 28, 28)))


