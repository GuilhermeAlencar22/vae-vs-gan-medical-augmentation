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
├── docs/
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

## 6. Achados de validação do pipeline

Estes resultados vêm da verificação do código e **devem entrar na discussão do
paper**. São eles que mostram leitura crítica dos próprios resultados.

### 6.1 O balanceamento 1:1 é nominal, não efetivo

Ao somar 1 200 imagens sintéticas às 300 reais, o treino fica numericamente
balanceado em 1 500 × 1 500. O desempenho, porém, **não melhora** — e de fato
piora um pouco em relação à linha de base.

O diagnóstico (`src/diagnostico.py`) explica o motivo. Treinando uma regressão
logística para distinguir embedding real de embedding sintético:

| Gerador | Separabilidade real vs sintético | Norma média do embedding | % da norma real |
|---|---|---|---|
| Real (normal) | — | 34,40 | 100 % |
| VAE | **1,0000** | 22,53 | 65,5 % |
| GAN | **1,0000** | 30,99 | 90,1 % |

Separabilidade de 1,0000 significa que até uma fronteira **linear** separa as
duas nuvens sem erro algum. Do ponto de vista do classificador, as imagens
sintéticas não são "mais exemplos de normal": são um terceiro agrupamento, que
as árvores isolam com um único corte e que portanto não ensina nada sobre as
radiografias normais de verdade.

A projeção t-SNE (`figuras/qualitativo_tsne.pdf`) mostra isso visualmente: as
três nuvens aparecem como ilhas praticamente disjuntas, sem sobreposição entre
o sintético e o real.

A norma menor do embedding sintético é a pista de **por que** isso acontece: a
ResNet responde com ativações mais fracas a texturas pobres. As imagens geradas
parecem radiografias a olho nu, mas não reproduzem a estatística de textura
fina que a rede usa para descrever as reais.

É um resultado honesto e publicável: **aumento de dados no espaço de pixels com
geradores treinados em 300 imagens de 28×28 não transfere para o espaço de
embeddings de uma rede pré-treinada.**

### 6.2 O limiar fixo de 0,5 enviesa a métrica

O ensemble decide pelo limiar padrão de 0,5 sobre um treino desbalanceado em
1 500 × 300. Isso travava a revocação da classe normal em torno de 0,53, com
precisão 0,97: o modelo só acertava os normais mais óbvios. Varrendo o limiar,
o F1 alcançável na seed 42 chegava a **0,82**, contra **0,68** medidos.

Como isso comprime os dois cenários contra o mesmo teto e torna o teste
insensível à diferença que ele quer detectar, o ensemble passou a usar
`class_weight="balanced"` nos três modelos (ver `config.PESO_CLASSE`). A
correção é **idêntica nos três cenários**, então não favorece VAE nem GAN: é
uma correção do instrumento de medida, não do tratamento. O F1 da linha de base
subiu de 0,683 para 0,727 na seed 42.

### 6.3 A diferença aparece na fidelidade, não no F1

Nas simulações de verificação, o F1 não distingue os dois geradores, mas a
**distância de Fréchet** distingue com folga: o VAE gera imagens
consistentemente mais próximas das reais que a DCGAN (dz ≈ −5,8, p < 0,01).

Ou seja, o VAE ganha em fidelidade e isso **não se converte** em ganho de
classificação — exatamente o que a seção 6.1 explica. Vale reportar as duas
métricas no paper: a conclusão interessante está na diferença entre elas.

> As execuções de verificação usaram poucas épocas e poucas seeds, apenas para
> validar o pipeline. Os números oficiais do paper saem da execução completa.

---

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
