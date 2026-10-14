"""Funções de apoio: reprodutibilidade e escolha do dispositivo."""

import os
import random

import numpy as np
import torch


def fixar_sementes(seed: int) -> None:
    """
    Fixa todas as fontes de aleatoriedade usadas no projeto.

    Precisamos fixar as quatro porque cada biblioteca tem o seu próprio
    gerador de números aleatórios: o random do Python, o NumPy (sorteio das
    imagens), o PyTorch na CPU e o PyTorch na GPU (inicialização dos pesos e
    embaralhamento dos lotes). Se uma ficar de fora, a simulação deixa de ser
    reproduzível.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def escolher_dispositivo() -> torch.device:
    """
    Usa a melhor aceleração disponível.

    mps  = GPU dos Macs com chip Apple (M1, M2, M3)
    cuda = GPU NVIDIA
    cpu  = funciona em qualquer máquina, só que mais devagar

    Observação para o paper: no MPS algumas operações não são 100%
    determinísticas, então rodar duas vezes a mesma seed pode variar na
    terceira ou quarta casa decimal. Isso não afeta o teste de hipótese.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
