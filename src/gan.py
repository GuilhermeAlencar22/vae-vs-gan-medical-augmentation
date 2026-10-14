"""
Técnica A no cenário H1 (contrafactual): DCGAN.

Duas redes competem. O Gerador transforma ruído em imagens; o Discriminador
tenta distinguir imagens reais das falsas. O Gerador melhora justamente por
tentar enganar o Discriminador. Não há perda de reconstrução: a qualidade é
julgada só pela outra rede, por isso as GANs costumam gerar imagens mais
nítidas que o VAE, mas são mais instáveis de treinar.

Referência: Radford, Metz e Chintala, "Unsupervised Representation Learning
with Deep Convolutional GANs" (DCGAN), 2016.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config


class Gerador(nn.Module):
    def __init__(self, dim_latente: int = config.DIM_LATENTE_GAN):
        super().__init__()
        self.dim_latente = dim_latente
        self.rede = nn.Sequential(
            # Projeta o ruído para um "mapa" 128x7x7, ponto de partida da imagem
            nn.Linear(dim_latente, 128 * 7 * 7),
            nn.Unflatten(1, (128, 7, 7)),
            # BatchNorm estabiliza o treino da GAN: mantém as ativações com
            # média e variância controladas, evitando que o gradiente exploda
            # ou suma. Recomendação explícita do artigo do DCGAN.
            nn.BatchNorm2d(128),
            # ReLU no gerador (recomendação do DCGAN)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),    # 14 -> 28
            # Tanh na saída: valores em [-1, 1], simétricos em torno de zero.
            # Por isso as imagens reais são convertidas para [-1, 1] no treino.
            nn.Tanh(),
        )

    def forward(self, z):
        return self.rede(z)


class Discriminador(nn.Module):
    def __init__(self):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),    # 28 -> 14
            # LeakyReLU em vez de ReLU: para entradas negativas ela ainda
            # deixa passar um gradiente pequeno (inclinação 0.2). Com ReLU,
            # neurônios "mortos" cortariam o gradiente que o Gerador precisa
            # receber para aprender.
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 14 -> 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            # Saída: UM número (logit). Sem Sigmoid aqui porque usamos
            # BCEWithLogitsLoss, que aplica a sigmoid internamente de forma
            # numericamente mais estável.
            nn.Linear(128 * 7 * 7, 1),
        )
        # Não há BatchNorm na primeira camada do Discriminador, outra
        # recomendação do DCGAN: normalizar a entrada crua causa oscilação.

    def forward(self, x):
        return self.rede(x)


def inicializar_pesos(modulo: nn.Module) -> None:
    """
    Inicialização do DCGAN: pesos ~ N(0, 0.02).

    O artigo mostra que começar com pesos pequenos evita que uma das redes
    domine logo nas primeiras iterações.
    """
    if isinstance(modulo, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(modulo.weight, 0.0, 0.02)
        if modulo.bias is not None:
            nn.init.zeros_(modulo.bias)
    elif isinstance(modulo, nn.BatchNorm2d):
        nn.init.normal_(modulo.weight, 1.0, 0.02)
        nn.init.zeros_(modulo.bias)


def treinar_gan(imagens: np.ndarray, dispositivo: torch.device, epocas: int = config.EPOCAS_GERADOR):
    """
    Treina uma DCGAN do zero nas imagens reais da classe minoritária.

    imagens: (N, 28, 28) em [0, 1]
    Retorna o gerador treinado e o histórico de perdas por época.
    """
    # [0, 1] -> [-1, 1] para casar com a saída Tanh do Gerador
    x = torch.from_numpy(imagens).unsqueeze(1) * 2.0 - 1.0
    carregador = DataLoader(TensorDataset(x), batch_size=config.TAMANHO_LOTE, shuffle=True)

    gerador = Gerador().to(dispositivo)
    discriminador = Discriminador().to(dispositivo)
    gerador.apply(inicializar_pesos)
    discriminador.apply(inicializar_pesos)

    # Um otimizador para cada rede: elas têm objetivos opostos
    opt_g = torch.optim.Adam(gerador.parameters(), lr=config.LR_GAN, betas=config.BETAS_ADAM_GAN)
    opt_d = torch.optim.Adam(discriminador.parameters(), lr=config.LR_GAN, betas=config.BETAS_ADAM_GAN)
    criterio = nn.BCEWithLogitsLoss()
    historico = {"perda_gerador": [], "perda_discriminador": []}

    gerador.train()
    discriminador.train()
    for _ in range(epocas):
        soma_g = soma_d = 0.0
        for (reais,) in carregador:
            reais = reais.to(dispositivo)
            n = reais.size(0)
            # One sided label smoothing: o alvo para imagens reais é 0.9 e
            # não 1.0. Isso impede o Discriminador de ficar confiante demais,
            # o que zeraria o gradiente enviado ao Gerador.
            alvo_real = torch.full((n, 1), config.SUAVIZACAO_ROTULO, device=dispositivo)
            alvo_falso = torch.zeros((n, 1), device=dispositivo)

            # Passo 1: treinar o Discriminador
            z = torch.randn(n, gerador.dim_latente, device=dispositivo)
            falsas = gerador(z)
            # .detach() corta o grafo: neste passo só o Discriminador aprende,
            # o gradiente não deve voltar para os pesos do Gerador.
            perda_d = criterio(discriminador(reais), alvo_real) + \
                      criterio(discriminador(falsas.detach()), alvo_falso)
            opt_d.zero_grad()
            perda_d.backward()
            opt_d.step()

            # Passo 2: treinar o Gerador
            # O Gerador quer que o Discriminador diga "real" (alvo 1) para as
            # falsas. Esta é a formulação "non saturating" de Goodfellow:
            # dá gradientes fortes no começo, quando o Gerador ainda é ruim.
            perda_g = criterio(discriminador(falsas), torch.ones((n, 1), device=dispositivo))
            opt_g.zero_grad()
            perda_g.backward()
            opt_g.step()

            soma_d += perda_d.item()
            soma_g += perda_g.item()
        n_lotes = len(carregador)
        historico["perda_discriminador"].append(soma_d / n_lotes)
        historico["perda_gerador"].append(soma_g / n_lotes)

    return gerador, historico


@torch.no_grad()
def gerar_gan(gerador: Gerador, n: int, dispositivo: torch.device) -> np.ndarray:
    """Sorteia ruído e gera n imagens. Retorna (n, 28, 28) em [0, 1]."""
    # eval() faz o BatchNorm usar as estatísticas acumuladas no treino,
    # e não as do lote atual.
    gerador.eval()
    z = torch.randn(n, gerador.dim_latente, device=dispositivo)
    imagens = gerador(z).squeeze(1)
    return ((imagens + 1.0) / 2.0).clamp(0, 1).cpu().numpy()   # [-1, 1] -> [0, 1]
