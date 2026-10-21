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
| 2 | T4 · Análise e mensagem central | ⬜ Pendente — **decisão Leo + Guilherme** |
| 3 | T5 · Paper / T6 · Apresentação | ⬜ Não iniciadas |
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
