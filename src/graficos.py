"""
Gráficos para o paper (formato IEEE, coluna de 3,5 polegadas).

Todos os gráficos são salvos em PDF (vetorial, para o LaTeX) e PNG (para a
apresentação). As cores são fixas por cenário em todas as figuras, para o
leitor associar azul = VAE e laranja = GAN sem precisar reler a legenda.
"""

import matplotlib

matplotlib.use("Agg")   # gera arquivos sem abrir janela (funciona em qualquer máquina)
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import config

CORES = {"SEM_AUMENTO": "#8a8984", "VAE": "#2a78d6", "GAN": "#eb6834"}
NOMES = {"SEM_AUMENTO": "Sem aumento", "VAE": "VAE (H0)", "GAN": "GAN (H1)"}
TINTA_PRIMARIA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
GRADE = "#e4e3df"

LARGURA_COLUNA = 3.5   # polegadas: largura de uma coluna IEEE

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": TINTA_SECUNDARIA,
    "axes.labelcolor": TINTA_PRIMARIA,
    "xtick.color": TINTA_SECUNDARIA,
    "ytick.color": TINTA_SECUNDARIA,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRADE,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "lines.linewidth": 1.5,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


def _salvar(figura, nome: str) -> None:
    config.PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    for extensao in ("pdf", "png"):
        figura.savefig(config.PASTA_FIGURAS / f"{nome}.{extensao}")
    plt.close(figura)


def boxplot_cenarios(tabela, metrica: str, rotulo: str, nome: str) -> None:
    """Distribuição das 30 simulações por cenário: caixa + pontos individuais."""
    cenarios = [c for c in ("SEM_AUMENTO", "VAE", "GAN") if c in set(tabela["cenario"])]
    figura, eixo = plt.subplots(figsize=(LARGURA_COLUNA, 2.4))
    rng = np.random.default_rng(0)
    for posicao, cenario in enumerate(cenarios):
        valores = tabela.loc[tabela["cenario"] == cenario, metrica].to_numpy()
        eixo.boxplot(
            valores, positions=[posicao], widths=0.5, showfliers=False,
            medianprops={"color": TINTA_PRIMARIA, "linewidth": 1.2},
            boxprops={"color": TINTA_SECUNDARIA, "linewidth": 0.8},
            whiskerprops={"color": TINTA_SECUNDARIA, "linewidth": 0.8},
            capprops={"color": TINTA_SECUNDARIA, "linewidth": 0.8},
        )
        # Pontos com leve deslocamento horizontal para não se sobreporem
        deslocamento = rng.uniform(-0.12, 0.12, size=len(valores))
        eixo.scatter(posicao + deslocamento, valores, s=12, color=CORES[cenario],
                     edgecolor="white", linewidth=0.5, zorder=3)
    eixo.set_xticks(range(len(cenarios)), [NOMES[c] for c in cenarios])
    eixo.set_ylabel(rotulo)
    eixo.grid(axis="x", visible=False)
    _salvar(figura, nome)


def linhas_pareadas(tabela, metrica: str, rotulo: str, nome: str) -> None:
    """Cada linha liga o resultado VAE e GAN da MESMA simulação (mesma seed)."""
    pivo = tabela.pivot(index="seed", columns="cenario", values=metrica)
    figura, eixo = plt.subplots(figsize=(LARGURA_COLUNA, 2.4))
    for _, linha in pivo.iterrows():
        cor = CORES["VAE"] if linha["VAE"] > linha["GAN"] else CORES["GAN"]
        eixo.plot([0, 1], [linha["VAE"], linha["GAN"]], color=cor, alpha=0.55, linewidth=0.9)
    eixo.scatter(np.zeros(len(pivo)), pivo["VAE"], s=12, color=CORES["VAE"], zorder=3)
    eixo.scatter(np.ones(len(pivo)), pivo["GAN"], s=12, color=CORES["GAN"], zorder=3)
    vitorias_vae = int((pivo["VAE"] > pivo["GAN"]).sum())
    eixo.set_title(f"VAE melhor em {vitorias_vae} de {len(pivo)} simulações", color=TINTA_SECUNDARIA)
    eixo.set_xticks([0, 1], [NOMES["VAE"], NOMES["GAN"]])
    eixo.set_xlim(-0.3, 1.3)
    eixo.set_ylabel(rotulo)
    eixo.grid(axis="x", visible=False)
    _salvar(figura, nome)


def histograma_diferencas(diferencas: np.ndarray, rotulo: str, nome: str) -> None:
    """Distribuição de (VAE menos GAN). Se H0 for verdade, ela fica centrada em zero."""
    figura, eixo = plt.subplots(figsize=(LARGURA_COLUNA, 2.2))
    eixo.hist(diferencas, bins=10, color=CORES["VAE"], edgecolor="white", linewidth=1.0)
    eixo.axvline(0, color=TINTA_SECUNDARIA, linestyle="--", linewidth=1.0, label="Diferença zero (H0)")
    eixo.axvline(diferencas.mean(), color=TINTA_PRIMARIA, linewidth=1.5,
                 label=f"Média = {diferencas.mean():+.4f}")
    eixo.set_xlabel(f"{rotulo}: VAE menos GAN")
    eixo.set_ylabel("Simulações")
    eixo.legend(frameon=False)
    _salvar(figura, nome)


def convergencia_media(diferencas: np.ndarray, rotulo: str, nome: str) -> None:
    """
    Média acumulada da diferença com IC de 95%, à medida que somamos simulações.
    Mostra visualmente por que 30 repetições dão robustez: a faixa estreita.
    """
    n = np.arange(1, len(diferencas) + 1)
    medias = np.cumsum(diferencas) / n
    meia_largura = np.full(len(n), np.nan)
    for i in range(1, len(n)):
        amostra = diferencas[: i + 1]
        meia_largura[i] = stats.t.ppf(0.975, df=i) * amostra.std(ddof=1) / np.sqrt(i + 1)
    figura, eixo = plt.subplots(figsize=(LARGURA_COLUNA, 2.2))
    eixo.fill_between(n, medias - meia_largura, medias + meia_largura, color=CORES["VAE"], alpha=0.18,
                      linewidth=0, label="IC 95%")
    eixo.plot(n, medias, color=CORES["VAE"], label="Média acumulada")
    eixo.axhline(0, color=TINTA_SECUNDARIA, linestyle="--", linewidth=1.0)
    eixo.set_xlabel("Número de simulações")
    eixo.set_ylabel(f"{rotulo}: VAE menos GAN")
    eixo.legend(frameon=False, loc="best")
    _salvar(figura, nome)


def grade_amostras(reais: np.ndarray, vae: np.ndarray, gan: np.ndarray, nome: str, n: int = 16) -> None:
    """Lado a lado: 16 radiografias reais, 16 do VAE e 16 da GAN."""
    figura, eixos = plt.subplots(1, 3, figsize=(LARGURA_COLUNA * 2, 2.5))   # largura de duas colunas
    for eixo, imagens, titulo in zip(eixos, (reais, vae, gan), ("Reais", "VAE", "GAN")):
        lado = int(np.sqrt(n))
        mosaico = imagens[:n].reshape(lado, lado, 28, 28).transpose(0, 2, 1, 3).reshape(lado * 28, lado * 28)
        eixo.imshow(mosaico, cmap="gray", vmin=0, vmax=1)
        eixo.set_title(titulo)
        eixo.axis("off")
    _salvar(figura, nome)


def curvas_perda(hist_vae: dict, hist_gan: dict, nome: str) -> None:
    """Dois painéis com eixos próprios: as perdas do VAE e da GAN têm escalas diferentes."""
    figura, (eixo_vae, eixo_gan) = plt.subplots(1, 2, figsize=(LARGURA_COLUNA * 2, 2.2))
    eixo_vae.plot(hist_vae["perda_total"], color=CORES["VAE"], label="Total")
    eixo_vae.plot(hist_vae["reconstrucao"], color=TINTA_SECUNDARIA, linewidth=1.0, label="Reconstrução")
    eixo_vae.set_title("VAE")
    eixo_vae.set_xlabel("Época")
    eixo_vae.set_ylabel("Perda por imagem")
    eixo_vae.legend(frameon=False)

    eixo_gan.plot(hist_gan["perda_gerador"], color=CORES["GAN"], label="Gerador")
    eixo_gan.plot(hist_gan["perda_discriminador"], color=TINTA_SECUNDARIA, linewidth=1.0, label="Discriminador")
    eixo_gan.set_title("GAN")
    eixo_gan.set_xlabel("Época")
    eixo_gan.set_ylabel("Perda BCE")
    eixo_gan.legend(frameon=False)
    _salvar(figura, nome)
