"""
Técnica B: Transfer Learning com ResNet18 pré treinada no ImageNet.

Usamos a ResNet como EXTRATORA DE CARACTERÍSTICAS congelada: removemos a
última camada (a que classifica as 1000 classes do ImageNet) e ficamos com o
vetor de 512 números que vem antes dela. Esse vetor (embedding) descreve a
imagem em termos de bordas, texturas e formas que a rede aprendeu com
1,2 milhão de fotos.

Por que congelar em vez de fazer fine tuning?
  1. Temos poucas imagens reais; ajustar 11 milhões de pesos com 300
     exemplos normais levaria a overfitting.
  2. Custo: as 30 simulações x 3 cenários exigiriam 90 fine tunings.
     Congelada, extraímos os embeddings das imagens reais UMA vez só.
  3. Controle experimental: com a ResNet fixa, a única coisa que muda entre
     os cenários é o gerador. Se ela fosse ajustada, ela própria viraria
     uma fonte extra de variação.

Referência: He et al., "Deep Residual Learning for Image Recognition", 2016.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

import config

# Média e desvio padrão dos canais RGB do ImageNet. A ResNet foi treinada com
# imagens normalizadas por esses valores, então precisamos fazer o mesmo:
# senão as entradas chegam numa escala que a rede nunca viu.
MEDIA_IMAGENET = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DESVIO_IMAGENET = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ExtratorResNet:
    def __init__(self, dispositivo: torch.device):
        self.dispositivo = dispositivo
        modelo = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Identity "desliga" a camada final: a saída passa a ser o embedding
        modelo.fc = nn.Identity()
        for parametro in modelo.parameters():
            parametro.requires_grad = False   # congelada: nenhum peso muda
        # eval() fixa o BatchNorm nas estatísticas do ImageNet
        self.modelo = modelo.eval().to(dispositivo)
        self.media = MEDIA_IMAGENET.to(dispositivo)
        self.desvio = DESVIO_IMAGENET.to(dispositivo)

    def _preparar(self, lote: torch.Tensor) -> torch.Tensor:
        """(n, 28, 28) em [0, 1] -> (n, 3, 112, 112) normalizado."""
        x = lote.unsqueeze(1)   # adiciona o canal: (n, 1, 28, 28)
        x = F.interpolate(x, size=config.TAMANHO_ENTRADA_RESNET, mode="bilinear", align_corners=False)
        # A ResNet espera 3 canais (RGB). A radiografia é cinza, então
        # repetimos o mesmo canal três vezes em vez de inventar cor.
        x = x.repeat(1, 3, 1, 1)
        return (x - self.media) / self.desvio

    @torch.no_grad()
    def extrair(self, imagens: np.ndarray) -> np.ndarray:
        """Retorna os embeddings (N, 512) de imagens (N, 28, 28) em [0, 1]."""
        saidas = []
        for inicio in range(0, len(imagens), config.LOTE_EXTRACAO):
            lote = torch.from_numpy(imagens[inicio:inicio + config.LOTE_EXTRACAO]).float()
            lote = self._preparar(lote.to(self.dispositivo))
            saidas.append(self.modelo(lote).cpu().numpy())
        return np.concatenate(saidas, axis=0)
