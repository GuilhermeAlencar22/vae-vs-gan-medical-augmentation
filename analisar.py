"""
Lê o CSV das simulações, executa o teste de hipótese e gera as figuras.

Uso:
    python analisar.py

Saídas em resultados/:
    relatorio_estatistico.txt   resumo legível (para a apresentação)
    tabela_resultados.tex       tabela pronta para colar no paper IEEE
    figuras/*.pdf e *.png       gráficos
"""

import json

import numpy as np
import pandas as pd

import config
from src import graficos, qualitativo
from src.estatistica import comparar_pareado

METRICAS = {
    "f1_minoria": "F1 (classe normal)",
    "auc": "AUC",
    "acuracia_balanceada": "Acurácia balanceada",
    "revocacao_minoria": "Revocação (normal)",
    "fd_resnet": "FD ResNet",
}


def main():
    tabela = pd.read_csv(config.ARQUIVO_CSV)
    pivo = {m: tabela.pivot(index="seed", columns="cenario", values=m) for m in METRICAS}
    n = len(pivo["f1_minoria"])
    if n < 3:
        # Shapiro Wilk e desvio padrão precisam de pelo menos 3 observações
        print(f"Só {n} simulação(ões) no CSV. Rode pelo menos 3 antes de analisar.")
        return

    linhas_relatorio = [
        "ESTUDO COMPARATIVO: VAE (H0) vs GAN (H1)",
        f"Simulações: {n}    alfa = {config.ALFA}",
        "Pipeline: gerador + ResNet18 congelada + ensemble de votação suave",
        "",
    ]

    # Descritivo por cenário
    linhas_relatorio.append("MÉDIA (DESVIO PADRÃO) POR CENÁRIO")
    descritivo = tabela.groupby("cenario")[list(METRICAS)].agg(["mean", "std"])
    for cenario in ("SEM_AUMENTO", "VAE", "GAN"):
        partes = []
        for metrica, rotulo in METRICAS.items():
            media, dp = descritivo.loc[cenario, (metrica, "mean")], descritivo.loc[cenario, (metrica, "std")]
            if not np.isnan(media):
                partes.append(f"{rotulo} {media:.4f} ({dp:.4f})")
        linhas_relatorio.append(f"  {cenario:<12} " + " | ".join(partes))
    linhas_relatorio.append("")

    # Testes pareados VAE vs GAN
    resultados = {}
    for metrica, rotulo in METRICAS.items():
        resultados[metrica] = comparar_pareado(pivo[metrica]["VAE"].to_numpy(), pivo[metrica]["GAN"].to_numpy())

    principal = resultados[config.METRICA_PRINCIPAL]
    linhas_relatorio += [
        f"TESTE DE HIPÓTESE PRINCIPAL ({METRICAS[config.METRICA_PRINCIPAL]})",
        "  H0: média(VAE menos GAN) = 0",
        "  H1: média(VAE menos GAN) diferente de 0",
        f"  Shapiro Wilk das diferenças: p = {principal['p_shapiro']:.4f}",
        f"  Teste aplicado: {principal['teste']}",
        f"  Estatística = {principal['estatistica']:.4f}    p valor = {principal['p_valor']:.6f}",
        f"  Diferença média = {principal['media_diferenca']:+.4f}  "
        f"IC95% [{principal['ic95_inferior']:+.4f}, {principal['ic95_superior']:+.4f}]",
        f"  d de Cohen (dz) = {principal['cohen_dz']:+.3f}  (efeito {principal['magnitude_efeito']})",
        "",
    ]
    if principal["rejeita_h0"]:
        vencedor = "VAE" if principal["media_diferenca"] > 0 else "GAN"
        linhas_relatorio.append(f"  CONCLUSÃO: rejeitamos H0. O cenário com {vencedor} teve desempenho "
                                f"significativamente superior.")
    else:
        linhas_relatorio.append("  CONCLUSÃO: não rejeitamos H0. Não há evidência estatística de "
                                "diferença entre VAE e GAN.")
    linhas_relatorio += ["", "MÉTRICAS SECUNDÁRIAS (exploratórias, sem correção para comparações múltiplas)"]
    for metrica, rotulo in METRICAS.items():
        if metrica == config.METRICA_PRINCIPAL:
            continue
        r = resultados[metrica]
        linhas_relatorio.append(f"  {rotulo:<22} dif = {r['media_diferenca']:+.4f}  {r['teste']:<10} "
                                f"p = {r['p_valor']:.6f}  dz = {r['cohen_dz']:+.3f}")

    relatorio = "\n".join(linhas_relatorio)
    print(relatorio)
    (config.PASTA_RESULTADOS / "relatorio_estatistico.txt").write_text(relatorio, encoding="utf-8")
    with open(config.PASTA_RESULTADOS / "testes.json", "w", encoding="utf-8") as arquivo:
        json.dump(resultados, arquivo, indent=2, ensure_ascii=False, default=float)

    escrever_tabela_latex(descritivo, resultados)
    gerar_figuras(tabela, pivo)
    print(f"\nFiguras em {config.PASTA_FIGURAS}")


def escrever_tabela_latex(descritivo, resultados) -> None:
    linhas = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Média (desvio padrão) em 30 simulações e teste pareado VAE vs GAN}",
        r"\label{tab:resultados}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Métrica & Sem aumento & VAE & GAN & $p$ \\",
        r"\hline",
    ]
    for metrica, rotulo in METRICAS.items():
        celulas = []
        for cenario in ("SEM_AUMENTO", "VAE", "GAN"):
            media = descritivo.loc[cenario, (metrica, "mean")]
            dp = descritivo.loc[cenario, (metrica, "std")]
            celulas.append("--" if np.isnan(media) else f"{media:.3f} ({dp:.3f})")
        p = resultados[metrica]["p_valor"]
        texto_p = "$<$0.001" if p < 0.001 else f"{p:.3f}"
        linhas.append(f"{rotulo} & " + " & ".join(celulas) + f" & {texto_p} \\\\")
    linhas += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (config.PASTA_RESULTADOS / "tabela_resultados.tex").write_text("\n".join(linhas), encoding="utf-8")


def gerar_figuras(tabela, pivo) -> None:
    graficos.boxplot_cenarios(tabela, "f1_minoria", "F1 (classe normal)", "boxplot_f1")
    graficos.boxplot_cenarios(tabela, "auc", "AUC", "boxplot_auc")
    graficos.linhas_pareadas(tabela, "f1_minoria", "F1 (classe normal)", "pareado_f1")
    diferencas = (pivo["f1_minoria"]["VAE"] - pivo["f1_minoria"]["GAN"]).to_numpy()
    graficos.histograma_diferencas(diferencas, "F1", "histograma_diferencas_f1")
    graficos.convergencia_media(diferencas, "F1", "convergencia_f1")

    arquivo_amostras = config.PASTA_RESULTADOS / "amostras_simulacao_0.npz"
    if arquivo_amostras.exists():
        amostras = np.load(arquivo_amostras)
        graficos.grade_amostras(amostras["reais"], amostras["vae"], amostras["gan"], "amostras_geradas")
    arquivo_historico = config.PASTA_RESULTADOS / "historico_simulacao_0.json"
    if arquivo_historico.exists():
        with open(arquivo_historico) as arquivo:
            historicos = json.load(arquivo)
        graficos.curvas_perda(historicos["VAE"], historicos["GAN"], "curvas_perda")

    # Figuras qualitativas (t-SNE e normas), equivalente ao Grad-CAM do slide de
    # referência. Retreina VAE e GAN da seed base no MESMO orçamento de épocas do
    # CSV, para a projeção casar com os números do experimento.
    epocas = int(tabela["epocas"].iloc[0]) if "epocas" in tabela.columns else config.EPOCAS_GERADOR
    qualitativo.executar(seed=config.SEED_BASE, epocas=epocas)


if __name__ == "__main__":
    main()
