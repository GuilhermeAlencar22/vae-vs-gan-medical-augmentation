"""
Métricas de avaliação.

Por que F1 da classe minoritária como métrica principal (e não acurácia)?
O teste tem mais pneumonia que normal. Um modelo que chuta "pneumonia" para
tudo já acerta 62,5%. A acurácia esconderia exatamente o problema que a
geração sintética tenta resolver: reconhecer bem a classe rara. O F1 combina
precisão e revocação da classe normal e só fica alto se as duas forem boas.
"""

import numpy as np
from scipy import linalg
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import config


def avaliar(modelo, x_teste: np.ndarray, y_teste: np.ndarray) -> dict:
    probabilidades = modelo.predict_proba(x_teste)[:, 1]   # P(pneumonia)
    previsoes = modelo.predict(x_teste)
    alvo = config.CLASSE_MINORIA
    return {
        "f1_minoria": f1_score(y_teste, previsoes, pos_label=alvo),
        "precisao_minoria": precision_score(y_teste, previsoes, pos_label=alvo, zero_division=0),
        "revocacao_minoria": recall_score(y_teste, previsoes, pos_label=alvo),
        "acuracia_balanceada": balanced_accuracy_score(y_teste, previsoes),
        # AUC não depende de limiar: mede se o modelo ORDENA bem os casos
        "auc": roc_auc_score(y_teste, probabilidades),
    }


def distancia_frechet(emb_reais: np.ndarray, emb_sinteticas: np.ndarray) -> float:
    """
    Distância de Fréchet entre duas nuvens de embeddings.

    É a mesma fórmula do FID (Heusel et al., 2017), mas calculada sobre os
    embeddings da ResNet18 em vez da InceptionV3, por isso chamamos de
    FD ResNet. Modela cada nuvem como uma gaussiana e mede a distância entre
    elas considerando média e covariância:
        FD = ||mu_r - mu_s||² + Tr(S_r + S_s - 2 * sqrt(S_r * S_s))
    Quanto MENOR, mais parecidas com as reais são as sintéticas.
    Serve para discutir QUALIDADE da geração, separada do desempenho final.

    Sobre o termo `epsilon`: o embedding tem 512 dimensões, mas estimamos a
    covariância com apenas algumas centenas de imagens. A matriz resultante é
    singular (rank menor que 512), e nesse caso `sqrtm` avisa "Matrix is
    singular" e devolve uma raiz imprecisa. Somar epsilon na diagonal é a
    regularização usada na implementação de referência do FID: desloca os
    autovalores nulos para um valor positivo minúsculo, o que torna a raiz
    quadrada bem definida sem alterar a distância de forma perceptível.
    """
    mu_r, mu_s = emb_reais.mean(axis=0), emb_sinteticas.mean(axis=0)
    cov_r = np.cov(emb_reais, rowvar=False)
    cov_s = np.cov(emb_sinteticas, rowvar=False)

    epsilon = 1e-6
    deslocamento = np.eye(cov_r.shape[0]) * epsilon
    raiz_produto = linalg.sqrtm((cov_r + deslocamento) @ (cov_s + deslocamento))
    # Erros numéricos podem gerar uma parte imaginária minúscula; descartamos
    raiz_produto = np.real(raiz_produto)
    diferenca = mu_r - mu_s
    return float(diferenca @ diferenca + np.trace(cov_r + cov_s - 2.0 * raiz_produto))
