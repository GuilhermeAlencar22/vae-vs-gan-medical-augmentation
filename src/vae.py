"""
Técnica A no cenário H0: Autoencoder Variacional (VAE) convolucional.

Ideia central: um autoencoder comum comprime a imagem num ponto do espaço
latente. O VAE comprime numa DISTRIBUIÇÃO (uma gaussiana com média mu e
variância sigma²). Isso obriga o espaço latente a ser contínuo e organizado,
e por isso podemos sortear z de uma N(0, I) e decodificar uma imagem nova.

Referência: Kingma e Welling, "Auto Encoding Variational Bayes", 2014.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import config


class VAE(nn.Module):
    def __init__(self, dim_latente: int = config.DIM_LATENTE_VAE):
        super().__init__()
        self.dim_latente = dim_latente

        # Codificador: 1x28x28 -> 32x14x14 -> 64x7x7
        # Convolução com stride 2 reduz a resolução pela metade e aprende o
        # que manter, em vez de descartar pixels com pooling fixo.
        # kernel 4, stride 2, padding 1 dá exatamente a metade: (28+2-4)/2+1 = 14
        self.codificador = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Duas cabeças lineares: uma para a média e outra para o log da
        # variância. Prevemos log(sigma²) em vez de sigma² porque a variância
        # precisa ser positiva, e o log pode assumir qualquer valor real.
        # Assim a rede nunca produz uma variância negativa inválida.
        self.fc_mu = nn.Linear(64 * 7 * 7, dim_latente)
        self.fc_logvar = nn.Linear(64 * 7 * 7, dim_latente)

        # Decodificador: espelho do codificador.
        self.fc_decodificador = nn.Linear(dim_latente, 64 * 7 * 7)
        self.decodificador = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            # ConvTranspose2d faz o caminho inverso: 7 -> 14 -> 28
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            # Sigmoid porque os pixels reais estão em [0, 1]. Ela também
            # combina com a perda BCE, que espera valores nesse intervalo.
            nn.Sigmoid(),
        )

    def codificar(self, x):
        h = self.codificador(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparametrizar(self, mu, logvar):
        """
        Truque da reparametrização.

        Sortear z ~ N(mu, sigma²) diretamente não é diferenciável: não dá para
        calcular gradiente através de um sorteio. A saída é reescrever
        z = mu + sigma * epsilon, com epsilon ~ N(0, 1). O sorteio fica
        isolado em epsilon, que não depende dos pesos, e o gradiente flui
        normalmente por mu e sigma.
        """
        sigma = torch.exp(0.5 * logvar)   # exp(0.5 * log sigma²) = sigma
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon

    def decodificar(self, z):
        return self.decodificador(self.fc_decodificador(z))

    def forward(self, x):
        mu, logvar = self.codificar(x)
        z = self.reparametrizar(mu, logvar)
        return self.decodificar(z), mu, logvar


def perda_vae(reconstrucao, x, mu, logvar, beta: float = config.BETA_KL):
    """
    Perda do VAE = reconstrução + beta * KL.

    Reconstrução (BCE): mede o quanto a imagem reconstruída se parece com a
    original, pixel a pixel.

    KL: mede o quanto a distribuição N(mu, sigma²) de cada imagem se afasta
    da N(0, I). Esse termo "puxa" o espaço latente para perto da origem e é
    o que permite gerar imagens novas sorteando z ~ N(0, I) depois.
    Fórmula fechada para duas gaussianas:
        KL = -0.5 * soma(1 + log sigma² - mu² - sigma²)

    Usamos soma (e não média) nos pixels para que a escala dos dois termos
    fique equilibrada, como no artigo original. Depois dividimos pelo
    tamanho do lote para a perda não depender do tamanho do lote.
    """
    bce = F.binary_cross_entropy(reconstrucao, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    n = x.size(0)
    return (bce + beta * kl) / n, bce / n, kl / n


def treinar_vae(imagens: np.ndarray, dispositivo: torch.device, epocas: int = config.EPOCAS_GERADOR):
    """
    Treina um VAE do zero nas imagens reais da classe minoritária.

    imagens: (N, 28, 28) em [0, 1]
    Retorna o modelo treinado e o histórico de perda por época.
    """
    x = torch.from_numpy(imagens).unsqueeze(1)   # (N, 1, 28, 28): canal único
    carregador = DataLoader(TensorDataset(x), batch_size=config.TAMANHO_LOTE, shuffle=True)

    modelo = VAE().to(dispositivo)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=config.LR_VAE)
    historico = {"perda_total": [], "reconstrucao": [], "kl": []}

    modelo.train()
    for _ in range(epocas):
        soma_total = soma_rec = soma_kl = 0.0
        for (lote,) in carregador:
            lote = lote.to(dispositivo)
            reconstrucao, mu, logvar = modelo(lote)
            perda, rec, kl = perda_vae(reconstrucao, lote, mu, logvar)

            otimizador.zero_grad()   # zera gradientes do passo anterior
            perda.backward()         # calcula gradientes
            otimizador.step()        # atualiza pesos

            soma_total += perda.item()
            soma_rec += rec.item()
            soma_kl += kl.item()
        n_lotes = len(carregador)
        historico["perda_total"].append(soma_total / n_lotes)
        historico["reconstrucao"].append(soma_rec / n_lotes)
        historico["kl"].append(soma_kl / n_lotes)

    return modelo, historico


@torch.no_grad()   # geração não treina nada, então dispensamos o grafo de gradientes
def gerar_vae(modelo: VAE, n: int, dispositivo: torch.device) -> np.ndarray:
    """Sorteia z ~ N(0, I) e decodifica. Retorna (n, 28, 28) em [0, 1]."""
    modelo.eval()
    z = torch.randn(n, modelo.dim_latente, device=dispositivo)
    return modelo.decodificar(z).squeeze(1).cpu().numpy()
