# VAE vs GAN para aumento de dados em radiografias de tórax

Estudo comparativo pareado com 30 simulações, aplicado à classificação de
pneumonia em radiografias com classe minoritária escassa.

**2ª Avaliação · Projeto + Paper**

---

## 1. Visão geral

O problema é classificar radiografias de tórax entre **normal** e **pneumonia**
quando há poucos exemplos rotulados da classe normal — a situação real de um
hospital que acabou de começar a rotular seu acervo.

A estratégia é gerar imagens sintéticas da classe escassa para balancear o
treino. A pergunta do estudo é **qual gerador faz isso melhor**: um VAE ou uma
GAN. Tudo o mais no pipeline é mantido idêntico, de modo que a diferença
medida só possa vir da troca do gerador.

### As três técnicas

| | Papel | Cenário H0 | Cenário H1 (contrafactual) |
|---|---|---|---|
| **A** | Geração sintética | **VAE convolucional** | **DCGAN** |
| **B** | Transfer Learning | ResNet18 (ImageNet), congelada | *idêntico* |
| **C** | Ensemble Learning | Votação suave: Reg. Logística + Random Forest + Gradient Boosting | *idêntico* |

Só a técnica **A** muda entre os cenários. Um terceiro cenário, `SEM_AUMENTO`,
não entra no teste de hipótese e serve de linha de base: ele responde "gerar
dados ajuda?" antes de respondermos "qual gerador é melhor?".

### Hipóteses

Sobre o **F1 da classe normal** no conjunto de teste oficial do MedMNIST:

- **H0:** média(F1_VAE − F1_GAN) = 0 — trocar o VAE pela GAN não muda nada.
- **H1:** média(F1_VAE − F1_GAN) ≠ 0 — há diferença de desempenho.

O teste é **bilateral**, porque não havia como saber de antemão qual gerador
venceria, e **pareado**, porque a simulação *i* usa a mesma seed nos dois
cenários: as mesmas imagens reais, o mesmo ensemble, o mesmo conjunto de teste.

### Desenho experimental

```
30 seeds (42 … 71)
  └── para cada seed:
        sorteia 300 normais + 1500 pneumonias reais   (escassez simulada)
        ├── SEM_AUMENTO : ensemble sobre os dados reais desbalanceados
        ├── VAE         : + 1200 normais sintéticas do VAE   → balanceado 1:1
        └── GAN         : + 1200 normais sintéticas da DCGAN → balanceado 1:1
                ↓
   Shapiro-Wilk nas 30 diferenças → teste t pareado ou Wilcoxon
                ↓
   p valor, d de Cohen (dz) e IC de 95%
```

---

## 2. Dataset

**PneumoniaMNIST** (MedMNIST v2, Yang et al., 2023): radiografias de tórax
pediátricas, 28×28, tons de cinza.

| Split | Imagens | Normal (classe 0) | Pneumonia (classe 1) |
|---|---|---|---|
| `train` | 4 708 | 1 214 (25,8 %) | 3 494 (74,2 %) |
| `test` | 624 | 234 (37,5 %) | 390 (62,5 %) |

O download é automático na primeira execução, para a pasta `dados/`.

**Por que criar escassez artificial?** Com as 1 214 normais disponíveis, o
aumento sintético faria pouca diferença. Reduzindo para 300 construímos o
cenário em que a geração de dados é de fato necessária.

**Por que F1 da classe normal e não acurácia?** O teste tem 62,5 % de
pneumonia, então um modelo que responde "pneumonia" para tudo já alcança
62,5 % de acurácia sem aprender nada. O F1 da classe normal só é alto se o
modelo reconhecer bem justamente a classe rara.

---

## 3. Estrutura do repositório

```
.
├── config.py               todos os hiperparâmetros num lugar só
├── simular.py              executa as 30 simulações e grava o CSV
├── analisar.py             teste de hipótese, tabela LaTeX e figuras
├── requirements.txt
├── src/
│   ├── dados.py            carrega o PneumoniaMNIST e sorteia a escassez
│   ├── vae.py              técnica A (H0): VAE convolucional
│   ├── gan.py              técnica A (H1): DCGAN
│   ├── extrator.py         técnica B: ResNet18 congelada
│   ├── ensemble.py         técnica C: votação suave
│   ├── metricas.py         F1, AUC, acurácia balanceada, distância de Fréchet
│   ├── estatistica.py      teste pareado, d de Cohen, IC de 95%
│   ├── graficos.py         figuras no tamanho de coluna IEEE
│   ├── diagnostico.py      verifica se o aumento sintético é efetivo
│   ├── qualitativo.py      projeção t-SNE dos embeddings (análise visual)
│   └── utilidades.py       seeds e escolha de GPU
├── paper/
│   ├── paper.tex           artigo em IEEE coluna dupla
│   ├── referencias.bib     bibliografia
│   └── README.md           como compilar (Overleaf ou local)
├── docs/
│   ├── APRESENTACAO.md     roteiro dos 7 blocos da apresentação
│   ├── STATUS_ATUAL.md     progresso das tarefas
│   ├── INSTRUCOES_LEO.md   passo a passo para rodar a execução oficial
│   ├── GUIA_DEFESA.md      perguntas prováveis do professor, com respostas
│   ├── PLANO_ENTREGA.md    divisão de tarefas e ordem das atividades
│   └── adr/                decisões de arquitetura registradas
└── resultados/             gerado ao rodar (não versionado)
```

Cada módulo tem uma responsabilidade só, e os comentários explicam **por que**
cada camada, ativação e hiperparâmetro está ali — não apenas o que o código faz.

---

## 4. Como executar

> **Vai rodar a execução oficial?** Siga `docs/INSTRUCOES_LEO.md`, que traz o
> passo a passo testado num clone limpo, o tempo esperado e o que fazer em
> cada erro conhecido.

### Pré-requisitos

Python 3.11 ou mais novo. No VS Code, a extensão **Python** da Microsoft.

### Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No VS Code, `Cmd+Shift+P` → **Python: Select Interpreter** → escolha o `.venv`.

### Teste rápido (cerca de 2 minutos)

Confirma que tudo funciona e baixa o dataset e os pesos da ResNet:

```bash
python simular.py --simulacoes 3 --epocas 5 --reiniciar
python analisar.py
```

Três simulações, e não duas, porque o Shapiro-Wilk e o desvio padrão precisam
de pelo menos três observações para rodar.

### Execução oficial

As 30 simulações com 200 épocas por gerador:

```bash
caffeinate -i python simular.py --reiniciar
python analisar.py
```

`caffeinate -i` impede o Mac de dormir durante a execução (omita em Linux ou
Windows). Leva algo em torno de uma hora num MacBook com chip M, que usa a GPU
via MPS automaticamente.

**Retomada.** O CSV é gravado ao fim de cada simulação. Se a execução for
interrompida, rode de novo **sem** `--reiniciar` e ela continua de onde parou.
O script se recusa a misturar no mesmo CSV simulações treinadas com números de
épocas diferentes, porque isso invalidaria a comparação.

### Diagnóstico do aumento sintético

```bash
python -m src.diagnostico --epocas 50     # evidência numérica
python -m src.qualitativo --epocas 50     # evidência visual (t-SNE)
```

Medem se as imagens sintéticas funcionam de fato como exemplos extras da
classe normal. Ver a seção 6.

---

## 5. Saídas

Tudo em `resultados/`:

| Arquivo | Uso |
|---|---|
| `simulacoes.csv` | números brutos, uma linha por cenário por simulação |
| `relatorio_estatistico.txt` | conclusão do teste, para a apresentação |
| `tabela_resultados.tex` | tabela pronta para colar no paper |
| `testes.json` | todas as estatísticas, para quem quiser reprocessar |
| `figuras/*.pdf` | figuras vetoriais para o LaTeX |
| `figuras/*.png` | figuras para os slides |

As figuras geradas são o boxplot por cenário, as linhas pareadas seed a seed,
o histograma das diferenças, a convergência da média acumulada com IC, a grade
de imagens reais vs VAE vs GAN, e as curvas de perda dos dois geradores.

---

## 6. Resultados oficiais e achados

Execução oficial: **30 seeds × 200 épocas**, pipeline completo.

### 6.1 Rejeitamos H0: a DCGAN supera o VAE

| Métrica | Sem aumento | VAE (H0) | GAN (H1) |
|---|---|---|---|
| F1 (classe normal) | 0,710 | 0,685 | **0,717** |
| AUC | **0,939** | 0,928 | 0,919 |
| Acurácia balanceada | 0,775 | 0,759 | **0,778** |
| Revocação (normal) | 0,570 | 0,534 | **0,586** |
| FD ResNet (menor = mais fiel) | — | 223,3 | **160,2** |

Teste t pareado (Shapiro p = 0,69, normalidade não rejeitada):
diferença média VAE−GAN = **−0,032**, IC95% [−0,037; −0,027],
**p < 10⁻⁶**, d de Cohen dz = **−2,40** (efeito grande).

### 6.2 O efeito é grande, o ganho é pequeno

Esta é a leitura que o paper defende. O dz de 2,40 é enorme e o p é minúsculo,
mas isso decorre do desenho pareado, que remove a variância do sorteio e torna
detectável uma diferença sistemática pequena.

Em termos absolutos: trocar VAE por GAN rende **0,032** de F1, e a GAN supera
**não usar aumento nenhum** por apenas **0,007** — enquanto o VAE *degrada* a
linha de base em 0,025. A AUC, aliás, favorece a linha de base sobre os dois
geradores.

Significância estatística e relevância prática divergem aqui, e reportar só o
valor p sugeriria uma conclusão bem mais entusiasmada do que os dados sustentam.

### 6.3 Por quê: o balanceamento é nominal, não efetivo

Uma regressão logística distingue características sintéticas de reais com
acurácia quase perfeita:

| Gerador | Separabilidade | Norma média | % da real |
|---|---|---|---|
| Reais | — | 34,4 | 100,0% |
| VAE | 0,997 | 27,3 | 79,4% |
| GAN | 0,986 | 34,6 | 100,5% |

Até uma fronteira **linear** separa as nuvens. Do ponto de vista do
classificador, as imagens sintéticas não são exemplos adicionais da classe
normal: são um agrupamento próprio, que as árvores isolam com poucos cortes.

A norma explica a diferença entre os geradores: a GAN reproduz a magnitude de
ativação das reais (100,5%), o VAE fica bem abaixo (79,4%), compatível com a
suavização da perda de reconstrução pixel a pixel. Por isso a GAN, embora
igualmente separável, degrada menos o classificador.

**Recomendação metodológica:** medir a separabilidade antes de atribuir ganho a
aumento generativo. É verificação barata — basta um classificador linear — e
distingue balanceamento nominal de efetivo.

> **Atenção.** Versões anteriores deste README afirmavam o oposto em três
> pontos (FD favorecendo o VAE, "ilha separada" no t-SNE, GAN com textura
> pobre). Aquelas conclusões vieram de execuções de verificação com poucas
> épocas e **não se sustentaram** a 200 épocas. Ver
> `docs/adr/ADR-0003-conclusoes-de-treinos-curtos-nao-se-sustentaram.md`.

## 7. Notas de reprodutibilidade

- Cada simulação fixa as quatro fontes de aleatoriedade: `random`, NumPy,
  PyTorch CPU e PyTorch GPU. O sorteio das imagens usa um
  `np.random.default_rng(seed)` próprio, de modo a não depender de quantos
  números aleatórios o resto do código já consumiu.
- `fixar_sementes(seed)` é chamada antes de **cada** gerador, para que VAE e
  GAN partam do mesmo estado aleatório.
- Os embeddings das imagens reais são calculados **uma única vez** e guardados
  em `dados/embeddings_resnet18_112.npz`. A ResNet está congelada, então esses
  vetores nunca mudam — é o que torna 30 simulações viáveis em cerca de uma hora.
- No backend MPS algumas operações não são totalmente determinísticas, então a
  mesma seed pode variar na terceira ou quarta casa decimal. Isso não afeta a
  conclusão do teste, mas vale citar como limitação no paper.

---

## 8. Defesa do código

`docs/GUIA_DEFESA.md` traz um roteiro de apresentação de cerca de 8 minutos e
mais de 50 perguntas prováveis com respostas, organizadas por arquivo: por que
`logvar` e não a variância, por que `.detach()`, por que LeakyReLU no
discriminador, por que teste pareado, o que o p valor significa exatamente.

Para depurar ao vivo, `.vscode/launch.json` já traz duas configurações. Na aba
*Run and Debug*, escolha **Teste rápido**, coloque um breakpoint em
`src/vae.py` (por exemplo dentro de `reparametrizar`) e inspecione o formato de
cada tensor durante a execução.

---

## 9. Referências

1. Kingma, D. P., Welling, M. *Auto-Encoding Variational Bayes.* ICLR, 2014.
2. Radford, A., Metz, L., Chintala, S. *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.* ICLR, 2016.
3. He, K., Zhang, X., Ren, S., Sun, J. *Deep Residual Learning for Image Recognition.* CVPR, 2016.
4. Yang, J. et al. *MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification.* Scientific Data, 2023.
5. Heusel, M. et al. *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.* NeurIPS, 2017.
6. Goodfellow, I. et al. *Generative Adversarial Nets.* NeurIPS, 2014.
