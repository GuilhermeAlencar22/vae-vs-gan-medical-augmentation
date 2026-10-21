"""
Análise qualitativa: a prova visual de que o aumento sintético não é efetivo.

O estudo mede a diferença entre VAE e GAN com números (F1, AUC, distância de
Fréchet). Este módulo responde a uma pergunta que número nenhum responde
sozinho: *como* as imagens sintéticas se distribuem em relação às reais?

Cumpre o papel que o Grad-CAM cumpre num classificador: mostrar onde o modelo
está olhando. Aqui a ResNet é extratora, não classificadora, então não existe
gradiente de uma classe prevista para projetar sobre a imagem. O equivalente
honesto é visualizar o ESPAÇO DE EMBEDDINGS, que é exatamente onde o ensemble
toma a decisão.

Duas figuras:

  1. PROJEÇÃO t-SNE. Reduz os 512 números do embedding a 2 dimensões, de forma
     a preservar vizinhança: pontos próximos no espaço original continuam
     próximos no plano. Se o aumento funcionasse, os pontos sintéticos cairiam
     MISTURADOS aos reais da mesma classe. Se formarem uma ilha separada, o
     classificador consegue isolá-los com um corte e eles não ensinam nada
     sobre a classe minoritária.

  2. DISTRIBUIÇÃO DA NORMA do embedding. A ResNet responde com ativações mais
     fracas a texturas pobres. Comparar os histogramas mostra se o gerador
     reproduz a riqueza de textura das radiografias reais.

Por que t-SNE e não PCA: o PCA é linear e preserva variância global, então
tende a esconder a separação entre agrupamentos. O t-SNE preserva estrutura
local de vizinhança, que é justamente o que importa aqui.

Uso:
    python -m src.qualitativo                   # usa EPOCAS_GERADOR do config
    python -m src.qualitativo --epocas 50 --seed 42
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

import config
from src.dados import carregar_split, sortear_indices_escassez
from src.extrator import ExtratorResNet
from src.gan import gerar_gan, treinar_gan
from src.graficos import CORES, LARGURA_COLUNA, TINTA_PRIMARIA, TINTA_SECUNDARIA, _salvar
from src.utilidades import escolher_dispositivo, fixar_sementes
from src.vae import gerar_vae, treinar_vae

GERADORES = {"VAE": (treinar_vae, gerar_vae), "GAN": (treinar_gan, gerar_gan)}

# Quantas imagens sintéticas projetar. Usamos o mesmo número de reais
# disponíveis para que nenhuma nuvem domine a figura pelo tamanho.
N_PROJECAO = 300


def projetar_tsne(emb_real: np.ndarray, emb_vae: np.ndarray, emb_gan: np.ndarray,
                  seed: int) -> None:
    """
    Projeta as três nuvens de embeddings num plano 2D com t-SNE.

    Projetamos tudo JUNTO numa única chamada, e não cada nuvem separadamente,
    porque o t-SNE não produz um espaço comparável entre execuções: duas
    projeções independentes não poderiam ser sobrepostas no mesmo eixo.
    """
    todos = np.concatenate([emb_real, emb_vae, emb_gan])
    # perplexity controla quantos vizinhos cada ponto considera. O padrão (30)
    # é adequado para centenas de pontos; abaixo disso o t-SNE fragmenta a
    # estrutura em agrupamentos espúrios.
    reduzido = TSNE(n_components=2, perplexity=30, init="pca",
                    random_state=seed).fit_transform(todos)

    limites = np.cumsum([0, len(emb_real), len(emb_vae), len(emb_gan)])
    figura, eixo = plt.subplots(figsize=(LARGURA_COLUNA, 2.8))
    for i, (nome, cor) in enumerate([("Reais (normal)", "#2f6b3c"),
                                     ("VAE", CORES["VAE"]),
                                     ("GAN", CORES["GAN"])]):
        fatia = reduzido[limites[i]:limites[i + 1]]
        eixo.scatter(fatia[:, 0], fatia[:, 1], s=7, alpha=0.6, color=cor,
                     linewidth=0, label=nome)
    eixo.set_xticks([])
    eixo.set_yticks([])
    eixo.grid(visible=False)
    eixo.legend(frameon=False, loc="best", markerscale=1.8)
    eixo.set_title("Embeddings da ResNet18 projetados (t-SNE)", color=TINTA_SECUNDARIA)
    _salvar(figura, "qualitativo_tsne")


def comparar_normas(emb_real: np.ndarray, emb_vae: np.ndarray,
                    emb_gan: np.ndarray) -> None:
    """Histograma da norma L2 do embedding: proxy da riqueza de textura."""
    figura, eixo = plt.subplots(figsize=(LARGURA_COLUNA, 2.2))
    for nome, emb, cor in [("Reais (normal)", emb_real, "#2f6b3c"),
                           ("VAE", emb_vae, CORES["VAE"]),
                           ("GAN", emb_gan, CORES["GAN"])]:
        normas = np.linalg.norm(emb, axis=1)
        eixo.hist(normas, bins=30, alpha=0.55, color=cor, label=nome, linewidth=0)
    eixo.set_xlabel("Norma L2 do embedding")
    eixo.set_ylabel("Imagens")
    eixo.legend(frameon=False)
    _salvar(figura, "qualitativo_normas")


def executar(seed: int = config.SEED_BASE, epocas: int = config.EPOCAS_GERADOR,
             dispositivo=None) -> None:
    """
    Gera as figuras qualitativas (t-SNE e normas) para uma seed.

    Treina VAE e GAN do zero porque a projeção precisa de centenas de imagens
    sintéticas: o CSV das simulações guarda só 16 por cenário, insuficiente
    para o t-SNE. Reutilizável pelo `analisar.py`, que a chama com o mesmo
    orçamento de épocas usado nas simulações.
    """
    if dispositivo is None:
        dispositivo = escolher_dispositivo()
    print(f"Qualitativo: dispositivo {dispositivo}    seed: {seed}    épocas: {epocas}")

    extrator = ExtratorResNet(dispositivo)
    imagens_treino, rotulos_treino = carregar_split("train")
    idx_min, _ = sortear_indices_escassez(rotulos_treino, seed)
    imagens_minoria = imagens_treino[idx_min]
    emb_real = extrator.extrair(imagens_minoria)

    embeddings = {}
    for nome, (treinar, gerar) in GERADORES.items():
        fixar_sementes(seed)
        modelo, _ = treinar(imagens_minoria, dispositivo, epocas=epocas)
        embeddings[nome] = extrator.extrair(gerar(modelo, N_PROJECAO, dispositivo))
        print(f"  {nome}: {N_PROJECAO} imagens sintéticas geradas")

    print("Projetando com t-SNE (pode levar alguns segundos)...")
    projetar_tsne(emb_real, embeddings["VAE"], embeddings["GAN"], seed)
    comparar_normas(emb_real, embeddings["VAE"], embeddings["GAN"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Análise qualitativa dos embeddings")
    parser.add_argument("--seed", type=int, default=config.SEED_BASE)
    parser.add_argument("--epocas", type=int, default=config.EPOCAS_GERADOR)
    args = parser.parse_args()

    executar(seed=args.seed, epocas=args.epocas)

    print(f"\nFiguras em {config.PASTA_FIGURAS}:")
    print("  qualitativo_tsne.pdf/.png     projeção 2D das três nuvens")
    print("  qualitativo_normas.pdf/.png   distribuição da norma do embedding")
    print("\nLeitura: se as nuvens sintéticas aparecerem como ilhas separadas da")
    print("nuvem real, o balanceamento é nominal e não efetivo — o classificador")
    print("isola as sintéticas em vez de aprender com elas.")


if __name__ == "__main__":
    main()
