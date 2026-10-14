"""
Teste de hipótese pareado entre os cenários VAE (H0) e GAN (H1).

Hipóteses sobre a métrica principal (F1 da classe minoritária):
    H0: média(F1_VAE - F1_GAN) = 0   -> trocar o VAE pela GAN não muda nada
    H1: média(F1_VAE - F1_GAN) != 0  -> há diferença de desempenho

Por que PAREADO?
Na simulação i, os dois cenários usam a mesma seed: as mesmas imagens reais,
o mesmo ensemble, o mesmo conjunto de teste. Cada par de resultados é como o
mesmo paciente medido antes e depois. Comparar as diferenças dentro de cada
par remove a variação que vem do sorteio dos dados e deixa o teste muito
mais sensível que um teste para amostras independentes.

Por que bilateral?
Não sabemos de antemão qual gerador vence. Fixar a direção depois de ver os
dados seria desonesto (é escolher o teste que dá o resultado desejado).

Escolha do teste:
  1. Shapiro Wilk nas 30 diferenças. Se não rejeitar normalidade,
  2. teste t pareado (paramétrico, mais poderoso quando a suposição vale);
  3. senão, Wilcoxon de postos sinalizados (não paramétrico, compara
     medianas e não exige normalidade).
"""

import numpy as np
from scipy import stats

import config


def comparar_pareado(a: np.ndarray, b: np.ndarray, alfa: float = config.ALFA) -> dict:
    """
    Compara duas amostras pareadas (a = VAE, b = GAN).

    Retorna um dicionário com estatísticas descritivas, o teste aplicado,
    p valor, tamanho de efeito, intervalo de confiança e a decisão.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diferencas = a - b
    n = len(diferencas)

    media_dif = diferencas.mean()
    dp_dif = diferencas.std(ddof=1)   # ddof=1: desvio padrão amostral (divide por n-1)

    # Normalidade das diferenças (o t pareado assume isso, não das amostras)
    if np.allclose(diferencas, diferencas[0]):
        p_shapiro = 0.0   # diferenças constantes: não há o que testar
    else:
        p_shapiro = stats.shapiro(diferencas).pvalue
    diferencas_normais = p_shapiro >= alfa

    if diferencas_normais:
        nome_teste = "t pareado"
        resultado = stats.ttest_rel(a, b)
    else:
        nome_teste = "Wilcoxon"
        resultado = stats.wilcoxon(a, b)
    p_valor = float(resultado.pvalue)

    # Tamanho de efeito d de Cohen para amostras pareadas (dz).
    # O p valor diz SE há diferença; o dz diz QUÃO GRANDE ela é, em unidades
    # de desvio padrão. Referência de Cohen: 0.2 pequeno, 0.5 médio, 0.8 grande.
    cohen_dz = media_dif / dp_dif if dp_dif > 0 else 0.0

    # Intervalo de confiança de 95% para a diferença média (distribuição t)
    erro_padrao = dp_dif / np.sqrt(n)
    t_critico = stats.t.ppf(1 - alfa / 2, df=n - 1)
    ic = (media_dif - t_critico * erro_padrao, media_dif + t_critico * erro_padrao)

    return {
        "n": n,
        "media_a": a.mean(), "dp_a": a.std(ddof=1),
        "media_b": b.mean(), "dp_b": b.std(ddof=1),
        "media_diferenca": media_dif, "dp_diferenca": dp_dif,
        "ic95_inferior": ic[0], "ic95_superior": ic[1],
        "p_shapiro": p_shapiro,
        "teste": nome_teste,
        "estatistica": float(resultado.statistic),
        "p_valor": p_valor,
        "cohen_dz": cohen_dz,
        "magnitude_efeito": classificar_efeito(cohen_dz),
        "rejeita_h0": p_valor < alfa,
    }


def classificar_efeito(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "desprezível"
    if d < 0.5:
        return "pequeno"
    if d < 0.8:
        return "médio"
    return "grande"
