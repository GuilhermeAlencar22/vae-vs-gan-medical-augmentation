# Status do projeto — o que foi feito até agora

**Atualizado em:** 2026-10-21
**Máquina:** MacBook Pro do Leo (GPU `mps`, Python 3.12)

Registro do progresso das tarefas da 2ª avaliação. Para o plano completo, ver
[PLANO_ENTREGA.md](PLANO_ENTREGA.md).

---

## Resumo rápido

| Fase | Tarefa | Situação |
|---|---|---|
| 1 | T1 · Revisão e validação do código (Leo) | ✅ Concluída (guia precisa de correção, ver abaixo) |
| 1 | T2 · Camada qualitativa (Guilherme) | ✅ Concluída e integrada no `analisar.py` |
| 1 | T3 · Execução oficial (~40 min) | ✅ Concluída — 30/30 seeds a 200 épocas |
| 2 | T4 · Análise e mensagem central | ✅ Decidida — ver abaixo |
| 3 | T5 · Paper (`paper/paper.tex`) | ✅ Escrito — falta diagrama + figuras |
| 3 | T6 · Apresentação (`docs/APRESENTACAO.md`) | ✅ Roteiro pronto — falta montar slides |
| 4 | T7 · Ensaio / T8 · Entrega | ⬜ Não iniciadas |

---

## O que foi executado

### Ambiente
- venv criado com **Python 3.12** (o `python3` padrão da máquina é 3.9, não serve).
- Dependências de `requirements.txt` instaladas; GPU detectada (`Dispositivo: mps`).
- Dataset PneumoniaMNIST e pesos da ResNet18 baixados (cache local).

### T1 — Revisão do código (Leo)
- Teste rápido rodado de ponta a ponta (`simular.py --simulacoes 3 --epocas 5` +
  `analisar.py`): passou.
- Diagnósticos rodados e entendidos (`src.diagnostico`, `src.qualitativo`).
- Guia de defesa escrito em [ESTUDO_T1_LEO.md](ESTUDO_T1_LEO.md), item por item,
  com as pegadinhas prováveis.
- **Atenção:** parte do guia (FD e separabilidade) foi escrita com base na
  narrativa antiga, que os números oficiais contradizem. Precisa de correção.

### T2 — Camada qualitativa (Guilherme)
- `src/qualitativo.py` refatorado: lógica extraída para a função `executar(seed,
  epocas, dispositivo)`, reutilizável. O CLI `python -m src.qualitativo` segue
  funcionando.
- `analisar.py` passou a chamar `qualitativo.executar(...)` ao final de
  `gerar_figuras`, usando o mesmo nº de épocas do CSV. Agora as figuras t-SNE e
  de normas saem **junto** com as demais numa só execução de `python analisar.py`
  (era o último item aberto da T2).

### T3 — Execução oficial
- `caffeinate -i python simular.py --reiniciar`: 30 seeds × 3 cenários a 200
  épocas, ~77 s por seed, concluída em ~40 min.
- `python analisar.py`: gerou o relatório, a tabela LaTeX e todas as figuras
  (incluindo as qualitativas).
- Arquivos para o grupo: `resultados/simulacoes.csv` e
  `resultados/relatorio_estatistico.txt` (pasta `resultados/` está no
  `.gitignore`).

---

## Resultado oficial (30 seeds, 200 épocas)

> **REJEITAMOS H0.** A GAN teve desempenho significativamente superior ao VAE.
> F1 (classe normal): diferença média VAE−GAN = **−0,0323**,
> IC95% [−0,0373, −0,0273], **p < 0,000001**, d de Cohen dz = **−2,40**
> (efeito grande). Shapiro p = 0,69 → teste t pareado.

| Métrica | Sem aumento | VAE (H0) | GAN (H1) |
|---|---|---|---|
| F1 (classe normal) | 0,710 | 0,685 | **0,717** |
| AUC | 0,939 | **0,928** | 0,919 |
| Acurácia balanceada | 0,775 | 0,759 | **0,778** |
| Revocação (normal) | 0,570 | 0,534 | **0,586** |
| FD ResNet (menor = mais fiel) | — | 223,3 | **160,2** |

Diagnóstico de separabilidade (200 épocas, seed 42):

| Gerador | Separabilidade | Norma sint. | % da real | FD ResNet |
|---|---|---|---|---|
| VAE | 0,9973 | 27,33 | 79,4% | 219,5 |
| GAN | 0,9860 | 34,58 | 100,5% | 150,9 |

---

## Mudança de narrativa (importante para T4/T5)

O resultado oficial a 200 épocas **contradiz** três pontos assumidos nos
documentos pré-escritos (que se baseavam em treinos curtos):

1. **FD favorecia o VAE** → **invertido**: a GAN é mais fiel (FD 160 vs 223).
2. **Sintético formava "ilha separada" no t-SNE** → com 200 épocas, a nuvem da
   **GAN se mistura com as reais**; a separação nítida só aparecia com poucas
   épocas.
3. **GAN com "textura pobre" (norma baixa)** → agora a norma da GAN é 100,5% da
   real: reproduz a textura.

**Permanece verdade:** separabilidade linear ainda alta, e o **VAE** continua
sem ajudar (piora vs linha de base). O achado "aumento nominal, não efetivo"
vale **só para o VAE**, não para a GAN.

Isto corresponde ao **cenário 3** do plano: "Rejeita H0; um gerador vence;
reportar dz e discutir se é relevante na prática." A diferença é
estatisticamente grande, mas pequena em termos absolutos (+0,032 no F1) — é o
ponto que a discussão do paper deve problematizar.

---

## Próximos passos

1. **T4 (Leo + Guilherme):** alinhar a mensagem central com os números oficiais
   — a conclusão mudou (GAN vence, GAN mais fiel). Decisão bloqueante antes de
   paper e slides.
2. Atualizar [ESTUDO_T1_LEO.md](ESTUDO_T1_LEO.md) e a seção "perguntas difíceis"
   de [GUIA_DEFESA.md](GUIA_DEFESA.md) com a narrativa correta (cenário 3).
3. Versionar os resultados: copiar as figuras finais para `docs/` (pois
   `resultados/` está no `.gitignore`) e commitar numa branch.
4. T5 (paper) e T6 (apresentação) — só depois de T4.


---

## Atualização 2026-10-21 (Guilherme)

### Conferência independente dos resultados

Rodei a execução oficial na minha máquina para cruzar com a do Leo. Interrompi
em 5 seeds (a 200 épocas ela leva ~5 min/seed aqui, contra 77 s no MacBook Pro
dele), mas o suficiente para confirmar a direção:

| Cenário | Leo (30 seeds) | Guilherme (5 seeds) |
|---|---|---|
| Sem aumento | 0,710 | 0,703 |
| VAE | 0,685 | 0,684 |
| GAN | **0,717** | **0,706** |
| GAN − VAE | +0,032 | +0,022 |

Mesma ordenação e mesma magnitude. **Os números do Leo se sustentam.** O CSV
parcial ficou na minha máquina e é retomável sem `--reiniciar`.

### T4 — mensagem central decidida

**"A GAN vence, mas o ganho é pequeno."** A diferença é estatisticamente
enorme (dz = 2,40, p < 10⁻⁶) porque o desenho pareado remove a variância do
sorteio, mas em valor absoluto são 0,032 de F1 — e a GAN supera *não usar
aumento nenhum* por apenas 0,007, enquanto o VAE degrada a linha de base.
A tensão entre significância estatística e relevância prática é o fio condutor
do paper.

### T5 e T6 entregues

- `paper/paper.tex` — artigo completo em IEEEtran, 10 referências,
  `paper/README.md` com instruções de compilação no Overleaf.
- `docs/APRESENTACAO.md` — roteiro dos 7 blocos, um por apresentador, com
  tempo, conteúdo e perguntas prováveis de cada parte.

### Documentos corrigidos

A narrativa antiga (escrita com treinos curtos) foi corrigida no README
(seção 6), no `GUIA_DEFESA.md` e na nota do ADR-0002. A lição ficou registrada
em `adr/ADR-0003-conclusoes-de-treinos-curtos-nao-se-sustentaram.md`.

### Pendências

1. **`ESTUDO_T1_LEO.md` não está no repositório** — o `STATUS_ATUAL.md` o
   referencia, mas o arquivo não foi commitado. Leo, pode subir?
2. **Diagrama do pipeline** para a Fig. 1 do paper (hoje é um `\framebox`
   marcador). Esboço em `paper/README.md`.
3. **Copiar as figuras** de `resultados/figuras/*.pdf` para `paper/figuras/`.
4. **Montar os slides** a partir do roteiro.
5. **T7 ensaio** com os 7 apresentadores.


---

## Atualização 2026-10-21, segunda rodada (Guilherme)

### Pendências fechadas

As duas que estavam comigo, depois do commit do Leo:

1. **Diagrama do pipeline (Fig. 1).** Era um `\framebox` marcador. Agora é um
   diagrama TikZ desenhado no próprio `.tex` — sem arquivo externo nem
   ferramenta de diagramação para manter. Segue a estrutura do slide de
   referência: dados → escassez → dois cenários lado a lado → classificador
   comum → teste pareado.

2. **Figuras inseridas no texto.** O Leo commitou as 9 figuras, mas o
   `paper.tex` não referenciava nenhuma: compilaria com zero gráficos, o que
   não atende a exigência da atividade. Seis entraram no texto e **todas são
   citadas** no corpo (float não referenciado é apontado em revisão).

### O paper compila

Verificado de ponta a ponta com Tectonic:

| | |
|---|---|
| Páginas | 5 |
| Tabelas | 3 |
| Figuras | 7 (1 diagrama + 6 da execução oficial) |
| Referências indefinidas | nenhuma |
| Bibliografia | 10 entradas, todas citadas |

Restam apenas avisos cosméticos de espaçamento vertical, normais em coluna
dupla. No caminho apareceu um erro real — o TikZ exigia
`\usetikzlibrary{calc}` para a sintaxe de coordenada média — já corrigido.

### Situação das tarefas

| Fase | Tarefa | Situação |
|---|---|---|
| 1 | T1 · Revisão do código (Leo) | ✅ `ESTUDO_T1_LEO.md` |
| 1 | T2 · Camada qualitativa | ✅ |
| 1 | T3 · Execução oficial | ✅ 30/30 seeds, conferida em execução independente |
| 2 | T4 · Mensagem central | ✅ "GAN vence, ganho pequeno" |
| 3 | T5 · Paper | ✅ **compila, 5 páginas** — falta só revisar o texto |
| 3 | T6 · Apresentação | 🟡 Roteiro pronto; **falta montar os slides** |
| 4 | T7 · Ensaio | ⬜ |
| 4 | T8 · Entrega | ⬜ |

### O que falta, em ordem

1. **Montar os slides** a partir de `APRESENTACAO.md` (7 blocos, um por
   apresentador). É a maior pendência.
2. **Revisar o texto do paper** — os dois lendo, à procura de afirmação que não
   se sustente nos números.
3. **Compilar o PDF final** no Overleaf (subir `paper/` inteiro) e conferir as
   figuras na diagramação final.
4. **T7:** ensaio cronometrado com os 7, cada um respondendo 3 perguntas
   sorteadas do `GUIA_DEFESA.md`.
5. **T8:** entregar paper em PDF, slides em PDF e o link do repositório.

### Observação sobre o histórico

O primeiro commit do Leo tem mensagem "Update". O conteúdo é ótimo, mas o
professor pode olhar o histórico — vale combinar o padrão de mensagens
(`docs:`, `feat:`, `fix:` + descrição em português) daqui em diante.
