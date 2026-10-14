"""
Carregamento do PneumoniaMNIST e criação do cenário de escassez.

PneumoniaMNIST (MedMNIST v2, Yang et al., 2023) contém radiografias de tórax
pediátricas em 28x28, tons de cinza, com dois rótulos: 0 = normal e
1 = pneumonia. Usamos a divisão oficial: "train" para sortear os dados de
treino de cada simulação e "test" (624 imagens, sempre a mesma) para avaliar.
"""

import numpy as np
from medmnist import PneumoniaMNIST

import config


def carregar_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna (imagens, rótulos) de um split do PneumoniaMNIST.

    imagens: float32 no intervalo [0, 1], formato (N, 28, 28)
    rótulos: int64, formato (N,)

    Dividimos por 255 porque os pixels vêm como inteiros de 0 a 255. O
    intervalo [0, 1] é o que a saída Sigmoid do VAE produz, então os dados
    reais e os sintéticos ficam na mesma escala.
    """
    config.PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    conjunto = PneumoniaMNIST(split=split, download=True, root=str(config.PASTA_DADOS))
    imagens = conjunto.imgs.astype(np.float32) / 255.0
    rotulos = conjunto.labels.reshape(-1).astype(np.int64)  # vem como (N, 1)
    return imagens, rotulos


def sortear_indices_escassez(rotulos: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sorteia, sem reposição, os índices das imagens reais desta simulação.

    Retorna dois vetores de índices: um da classe minoritária (normal) e
    outro da majoritária (pneumonia). Usamos um gerador próprio
    (np.random.default_rng) para que o sorteio dependa só da seed e não de
    quantos números aleatórios outras partes do código já consumiram.

    Como VAE e GAN recebem a MESMA seed, os dois cenários treinam com
    exatamente as mesmas imagens reais. É isso que torna o desenho pareado.
    """
    rng = np.random.default_rng(seed)
    idx_minoria = np.flatnonzero(rotulos == config.CLASSE_MINORIA)
    idx_maioria = np.flatnonzero(rotulos == config.CLASSE_MAIORIA)
    escolhidos_minoria = rng.choice(idx_minoria, size=config.N_MINORIA_TREINO, replace=False)
    escolhidos_maioria = rng.choice(idx_maioria, size=config.N_MAIORIA_TREINO, replace=False)
    return escolhidos_minoria, escolhidos_maioria
