# Paper — instruções de compilação

## Compilar no Overleaf (recomendado)

1. Acesse [overleaf.com](https://www.overleaf.com) → *New Project* → *Upload Project*
2. Suba o conteúdo desta pasta: `paper.tex`, `referencias.bib` e `figuras/`
3. Em *Menu > Compiler*, selecione **pdfLaTeX**
4. Compile duas vezes (a segunda resolve as citações do BibTeX)

O documento usa a classe `IEEEtran`, já disponível no Overleaf. Não é preciso
instalar nada.

## Compilar localmente

Requer uma distribuição LaTeX (MacTeX no macOS, TeX Live no Linux):

```bash
cd paper
pdflatex paper
bibtex paper
pdflatex paper
pdflatex paper
```

As três passagens são necessárias: a primeira gera as referências cruzadas, o
`bibtex` resolve a bibliografia e as duas últimas fixam numeração e citações.

## Estado atual

O documento **compila sem erros** e produz 5 páginas, com 3 tabelas e 7
figuras, todas referenciadas no texto. Verificado com Tectonic em 2026-10-21.

As figuras da execução oficial (30 seeds × 200 épocas) já estão versionadas em
`figuras/` e inseridas no texto. O diagrama do pipeline (Fig. 1) é desenhado em
TikZ dentro do próprio `.tex`, sem depender de arquivo externo.

## Figuras

As figuras do experimento são geradas em `resultados/figuras/` ao rodar
`python analisar.py`. Caso a execução seja repetida, atualize-as com:

```bash
cp ../resultados/figuras/*.pdf figuras/
```

Para inserir uma figura nova no texto:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{boxplot_f1}
  \caption{Distribuição do F1 da classe minoritária nas 30 simulações.}
  \label{fig:boxplot}
\end{figure}
```

Use `\columnwidth` para figura de uma coluna e `figure*` com `\textwidth` para
figura que ocupe a largura da página.

### Figuras disponíveis

| Arquivo | Conteúdo | Sugestão de uso |
|---|---|---|
| `boxplot_f1` | F1 por cenário, 30 simulações | Resultados |
| `boxplot_auc` | AUC por cenário | Resultados (opcional) |
| `pareado_f1` | Linhas ligando VAE e GAN da mesma seed | Resultados — ilustra o pareamento |
| `histograma_diferencas_f1` | Distribuição de VAE − GAN | Resultados |
| `convergencia_f1` | Média acumulada com IC 95% | Metodologia — justifica as 30 repetições |
| `amostras_geradas` | Grade: reais, VAE, GAN | Resultados — usar `figure*` |
| `curvas_perda` | Perdas de treino dos dois geradores | Apêndice ou metodologia |
| `qualitativo_tsne` | Projeção t-SNE das três nuvens | Discussão — evidência da separabilidade |
| `qualitativo_normas` | Histograma das normas | Discussão |

## O diagrama do pipeline (já pronto)

A Fig. 1 é desenhada em TikZ diretamente no `paper.tex` (requer a biblioteca
`calc`, já declarada no preâmbulo). Reproduz este fluxo:

```
Conjunto PneumoniaMNIST  →  escassez simulada (300 / 1.500)
                              ↓
          ┌───────────────────┴───────────────────┐
   Solução principal                     Solução contrafactual
   VAE + 1.200 sintéticas                DCGAN + 1.200 sintéticas
          └───────────────────┬───────────────────┘
                              ↓
              ResNet18 congelada (512 características)
                              ↓
          Ensemble de votação suave (LR + RF + GB)
                              ↓
        Avaliação no teste oficial (F1, AUC, FD)
                              ↓
      30 seeds pareadas  →  teste t pareado bilateral
```

Para alterá-lo, edite o bloco `tikzpicture` na seção *Pipeline* do `paper.tex`.
Não há arquivo externo a manter.

## Números a conferir antes de submeter

Os valores no texto vieram da execução oficial de 30 seeds × 200 épocas.
Confirme contra `resultados/relatorio_estatistico.txt` antes de entregar —
especialmente se a execução for repetida.

- Tabela `tab:resultados` ↔ `resultados/tabela_resultados.tex`
- Diferença média, IC, p e dz ↔ `relatorio_estatistico.txt`
- Tabela `tab:separabilidade` ↔ saída de `python -m src.diagnostico`
