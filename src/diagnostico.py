"""
Diagnóstico da qualidade do aumento sintético.

Este módulo não entra no teste de hipótese. Ele responde a uma pergunta que
apareceu ao analisar os resultados: por que balancear o treino de 300 para
1500 imagens da classe normal quase não melhora a revocação?

A resposta é que o balanceamento é NOMINAL e não EFETIVO. As imagens
sintéticas, depois de passarem pela ResNet, caem numa região do espaço de
embeddings que é separável das reais com acurácia praticamente perfeita. Do
ponto de vista do classificador elas não são "mais exemplos de normal": são
um terceiro agrupamento, que as árvores isolam com um corte e que portanto
não ensina nada sobre as radiografias normais de verdade.

Medimos duas evidências:

  1. SEPARABILIDADE. Treinamos uma regressão logística para distinguir
     embedding real de embedding sintético, com validação cruzada. Se a
     acurácia for próxima de 0,5, as nuvens se misturam (o aumento é útil).
     Próxima de 1,0, são conjuntos distintos (o aumento é decorativo).

  2. NORMA MÉDIA do embedding. A ResNet responde com ativações mais fracas a
     imagens borradas ou de baixo contraste. Uma norma sintética bem menor
     que a real indica que o gerador produz texturas pobres demais, mesmo
     quando a imagem "parece" uma radiografia a olho nu.

Uso:
    python -m src.diagnostico                  # VAE e GAN, seed padrão
    python -m src.diagnostico --seed 50 --epocas 50
"""

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.dados import carregar_split, sortear_indices_escassez
from src.extrator import ExtratorResNet
from src.gan import gerar_gan, treinar_gan
from src.metricas import distancia_frechet
from src.utilidades import escolher_dispositivo, fixar_sementes
from src.vae import gerar_vae, treinar_vae

GERADORES = {"VAE": (treinar_vae, gerar_vae), "GAN": (treinar_gan, gerar_gan)}


def separabilidade(emb_reais: np.ndarray, emb_sinteticas: np.ndarray, seed: int) -> float:
    """
    Acurácia de um classificador real vs sintético (validação cruzada, 3 folds).

    0,50 = indistinguíveis (ideal para aumento de dados)
    1,00 = trivialmente distinguíveis (o aumento não agrega informação)

    Usamos um modelo LINEAR de propósito: se até uma fronteira linear separa
    as duas nuvens, a diferença é grosseira, não um detalhe sutil de textura.
    """
    x = np.concatenate([emb_reais, emb_sinteticas])
    y = np.concatenate([np.zeros(len(emb_reais)), np.ones(len(emb_sinteticas))])
    modelo = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=config.MAX_ITER_LR, random_state=seed),
    )
    return float(cross_val_score(modelo, x, y, cv=3, scoring="accuracy").mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico do aumento sintético")
    parser.add_argument("--seed", type=int, default=config.SEED_BASE)
    parser.add_argument("--epocas", type=int, default=config.EPOCAS_GERADOR)
    args = parser.parse_args()

    dispositivo = escolher_dispositivo()
    print(f"Dispositivo: {dispositivo}    seed: {args.seed}    épocas: {args.epocas}\n")

    extrator = ExtratorResNet(dispositivo)
    imagens_treino, rotulos_treino = carregar_split("train")
    idx_min, _ = sortear_indices_escassez(rotulos_treino, args.seed)
    imagens_minoria = imagens_treino[idx_min]
    emb_reais = extrator.extrair(imagens_minoria)

    norma_real = np.linalg.norm(emb_reais, axis=1).mean()
    print(f"Imagens reais da classe normal: {len(imagens_minoria)}")
    print(f"  norma média do embedding: {norma_real:.2f}\n")

    print(f"{'Gerador':<9}{'Separabilidade':>16}{'Norma sint.':>14}{'% da real':>12}{'FD ResNet':>12}")
    print("-" * 63)
    for nome, (treinar, gerar) in GERADORES.items():
        fixar_sementes(args.seed)
        modelo, _ = treinar(imagens_minoria, dispositivo, epocas=args.epocas)
        sinteticas = gerar(modelo, config.N_SINTETICAS, dispositivo)
        emb_sinteticas = extrator.extrair(sinteticas)

        acuracia = separabilidade(emb_reais, emb_sinteticas, args.seed)
        norma_sintetica = np.linalg.norm(emb_sinteticas, axis=1).mean()
        fd = distancia_frechet(emb_reais, emb_sinteticas)
        print(f"{nome:<9}{acuracia:>16.4f}{norma_sintetica:>14.2f}"
              f"{100 * norma_sintetica / norma_real:>11.1f}%{fd:>12.1f}")

    print("\nLeitura: separabilidade perto de 1,00 significa que o classificador")
    print("distingue sintético de real sem esforço, então as imagens geradas não")
    print("funcionam como exemplos adicionais da classe normal. O balanceamento")
    print("1:1 é nominal; o ganho de desempenho esperado não se materializa.")


if __name__ == "__main__":
    main()
