# Plano de entrega — 2ª Avaliação

**Codando:** 2 pessoas (Guilherme e Leo) · **Apresentando:** 7 pessoas
**Artefatos:** código + paper (IEEE coluna dupla) + apresentação

Este plano cobre os passos que o slide de referência (projeto Beans/MobileNetV2)
demonstra serem o padrão esperado pelo professor, adaptados ao nosso pipeline.

---

## Mapa: slide de referência → nosso projeto

| Etapa no slide de referência | Equivalente aqui | Situação |
|---|---|---|
| Conjunto de dados (3 classes, splits) | PneumoniaMNIST, 2 classes, split oficial | **Pronto** |
| Pré-processamento (segmentação HSV, resize) | Normalização [0,1], resize 28→112, 3 canais | **Pronto** |
| Solução principal vs contrafactual | VAE (H0) vs DCGAN (H1) | **Pronto** |
| Cabeça/classificador comum às duas soluções | ResNet18 congelada + ensemble (idêntico nos dois) | **Pronto** |
| Treino com hiperparâmetros declarados | 200 épocas, Adam, lote 32/64 — em `config.py` | **Pronto** |
| Avaliação no teste (F1 macro, acurácia) | F1 minoritária, AUC, acurácia balanceada, FD | **Pronto** |
| 30 pares / 60 treinamentos | 30 seeds × 3 cenários = 90 treinos | **Falta rodar oficial** |
| Teste t pareado bilateral (IC 95%, p) | Shapiro → t pareado ou Wilcoxon + dz + IC | **Pronto** |
| **Grad-CAM (análise qualitativa)** | **Análise qualitativa dos embeddings** | **FALTA — T4** |

Duas lacunas reais: a execução oficial e a camada qualitativa.

---

## Ordem das atividades

As tarefas estão em ordem de dependência. **T1 e T2 são bloqueantes**: nada de
paper ou apresentação começa antes delas, porque sem números reais tudo vira
retrabalho.

```
FASE 1  T1 revisão ──┐
        T2 qualitativa ┴─→ T3 EXECUÇÃO OFICIAL (~70 min, sozinho na máquina)
                               │
FASE 2                         ├─→ T4 análise dos resultados
                               │        │
FASE 3                         │        ├─→ T5 paper
                               │        └─→ T6 apresentação
FASE 4                         └─────────→ T7 ensaio + T8 entrega
```

---

## FASE 1 — Antes de rodar o oficial

### T1 · Revisão e validação do código — **Leo** (principal) + Guilherme

Você pediu revisar antes de gastar a execução. Esta é a tarefa.

Leo deve conseguir **explicar** cada item, não só confirmar que roda — o
professor pergunta a qualquer um dos dois:

- [ ] Ler `docs/GUIA_DEFESA.md` inteiro e marcar o que não souber responder
- [ ] `config.py`: conferir que os dois geradores têm o mesmo orçamento (épocas, lote)
- [ ] `src/vae.py`: reparametrização, por que `logvar`, por que BCE com `reduction="sum"`
- [ ] `src/gan.py`: `.detach()`, LeakyReLU, label smoothing 0,9, betas (0.5, 0.999)
- [ ] `src/extrator.py`: por que congelada, por que 112 px, por que 3 canais
- [ ] `src/ensemble.py`: por que `class_weight="balanced"` (ver ADR-0002)
- [ ] `src/estatistica.py`: Shapiro → t ou Wilcoxon, bilateral, dz, IC
- [ ] Rodar o teste rápido e confirmar que passa de ponta a ponta:
      `python simular.py --simulacoes 3 --epocas 5 --reiniciar && python analisar.py`
- [ ] Rodar `python -m src.diagnostico --epocas 50` e entender a separabilidade 1,0000

**Saída:** lista de dúvidas para o Guilherme responder antes de T3.

### T2 · Camada qualitativa (equivalente ao Grad-CAM) — **Guilherme**

A única lacuna de escopo frente ao slide de referência.

- [ ] `src/qualitativo.py`: projeção 2D (t-SNE) dos embeddings reais vs VAE vs GAN
- [ ] Figura mostrando visualmente que o sintético forma cluster separado
- [ ] Grade de imagens reais vs VAE vs GAN lado a lado (já existe, integrar)
- [ ] Integrar em `analisar.py` para gerar junto com as demais figuras

**Por que isso importa:** é a prova *visual* do achado central (separabilidade
1,0000). O número convence o revisor; a figura convence a plateia.

### T3 · Execução oficial — **Guilherme** (quem tem o Mac com GPU)

Só começa quando T1 e T2 estiverem fechadas.

- [ ] Fechar apps pesados; a máquina fica dedicada
- [ ] `caffeinate -i python simular.py --reiniciar`  (**~70 min**, medido)
- [ ] `python analisar.py`
- [ ] `python -m src.qualitativo` (de T2)
- [ ] Commit dos resultados numa branch (`resultados/` está no `.gitignore`;
      copiar as figuras finais para `docs/` para versionar)
- [ ] Mandar `simulacoes.csv` + `relatorio_estatistico.txt` para o grupo

> Tempo medido nesta máquina: ensemble ~20 s × 3 + geradores ~47 s = ~111 s por
> seed. 30 seeds ≈ 56 min, ~70 min com folga. Se interromper, rode de novo
> **sem** `--reiniciar` que ele continua de onde parou.

---

## FASE 2 — Com os números na mão

### T4 · Análise e interpretação — **Guilherme e Leo juntos**

- [ ] Ler `relatorio_estatistico.txt`: rejeita ou não rejeita H0?
- [ ] Anotar: diferença média, IC 95%, p, dz, qual teste o Shapiro escolheu
- [ ] Conferir se o achado da separabilidade se mantém com 200 épocas
- [ ] Conferir se o FD continua favorecendo o VAE (p < 0,01 nas verificações)
- [ ] Decidir a mensagem central do paper a partir do que os dados mostrarem

**Três cenários possíveis, todos defensáveis** — decidam qual é o de vocês:

1. *Não rejeita H0 e aumento não ajuda.* A mensagem é o porquê: balanceamento
   nominal vs efetivo, com a separabilidade como evidência.
2. *Não rejeita H0 no F1, mas FD distingue.* Fidelidade não se converte em
   desempenho — o contraste entre as métricas é a contribuição.
3. *Rejeita H0.* Um gerador vence; reportar dz e discutir se é relevante na prática.

---

## FASE 3 — Artefatos (podem ser paralelos)

### T5 · Paper IEEE coluna dupla — **Guilherme** (escrita) + Leo (revisão)

Template: IEEEtran no Overleaf.

- [ ] Abstract · Introdução · Trabalhos relacionados
- [ ] Metodologia — **espelhar o slide de referência**: dados e splits,
      pré-processamento, as 3 técnicas, principal vs contrafactual, protocolo
      das 30 simulações pareadas, hiperparâmetros em tabela
- [ ] Resultados — colar `tabela_resultados.tex`, inserir as figuras em PDF
- [ ] Discussão — o achado da separabilidade, fidelidade vs desempenho
- [ ] Limitações — 28×28, MPS não determinístico, ResNet congelada, 2 classes
- [ ] Conclusão · Referências (as 6 do README)

### T6 · Apresentação — **Leo** (montagem) + Guilherme (dados e figuras)

**7 pessoas apresentam.** Montar como 7 blocos, cada pessoa dona do seu:

| # | Bloco | Conteúdo |
|---|---|---|
| 1 | Problema e motivação | Escassez da classe minoritária em contexto clínico |
| 2 | Dados | PneumoniaMNIST, splits, escassez simulada, por que F1 e não acurácia |
| 3 | Técnica A — geradores | VAE vs DCGAN, o contrafactual |
| 4 | Técnicas B e C | ResNet18 congelada + ensemble de votação suave |
| 5 | Desenho experimental | 30 seeds pareadas, hipóteses, escolha do teste |
| 6 | Resultados | Tabela, boxplot, linhas pareadas, FD |
| 7 | Discussão e conclusão | Separabilidade, limitações, trabalhos futuros |

- [ ] Slide de metodologia com **fluxograma**, no padrão do slide de referência
- [ ] Figuras em PNG (já geradas em `resultados/figuras/`)
- [ ] Cada apresentador estuda o bloco **e** o `GUIA_DEFESA.md` da sua parte

---

## FASE 4 — Fechamento

### T7 · Ensaio e preparação para arguição — **todos os 7**

- [ ] Ensaio cronometrado, na ordem real
- [ ] Rodar o código ao vivo uma vez (teste rápido, 2 min) para garantir que funciona
- [ ] Simular a arguição: cada apresentador responde 3 perguntas do `GUIA_DEFESA.md`
      sorteadas pelos colegas
- [ ] Preparar o ponto crítico: *"por que o aumento sintético não melhorou?"* —
      todo mundo precisa saber responder, é a pergunta mais provável

### T8 · Entrega — **Guilherme**

- [ ] Paper em PDF (IEEE coluna dupla, com as figuras)
- [ ] Repositório atualizado e público/acessível
- [ ] Slides exportados em PDF
- [ ] README com instruções que funcionam numa máquina limpa

---

## Divisão resumida

| | Guilherme | Leo |
|---|---|---|
| **Fase 1** | T2 camada qualitativa · T3 execução oficial | T1 revisão e validação do código |
| **Fase 2** | T4 análise — juntos | T4 análise — juntos |
| **Fase 3** | T5 paper (escrita) | T6 apresentação (montagem) |
| **Fase 4** | T8 entrega | T7 conduzir o ensaio |

**Critério do professor:** ele lê o código e pergunta por que cada camada,
ativação, biblioteca e hiperparâmetro está ali. Dividir tarefas não divide a
responsabilidade de saber: os dois precisam dominar o código inteiro, e os 7
apresentadores precisam dominar o próprio bloco.
