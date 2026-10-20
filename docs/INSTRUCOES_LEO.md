# Instruções de execução — Leo

Passo a passo testado num clone limpo. Se algo divergir do descrito aqui,
pare e avise antes de continuar.

**Sua máquina:** MacBook Pro. O código detecta a GPU (MPS) sozinho.

---

## 1. Clonar e instalar (uma vez, ~5 min)

```bash
git clone https://github.com/GuilhermeAlencar22/vae-vs-gan-medical-augmentation.git
cd vae-vs-gan-medical-augmentation

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requer Python 3.11+. Confirme com `python3 --version`.

No VS Code: *File > Open Folder*, depois `Cmd+Shift+P` →
**Python: Select Interpreter** → escolha o `.venv`.

> O dataset (PneumoniaMNIST, 4 MB) e os pesos da ResNet18 (45 MB) baixam
> sozinhos na primeira execução. Não estão no repositório de propósito.

---

## 2. Teste rápido (~2 min) — confirme que tudo funciona

```bash
python simular.py --simulacoes 3 --epocas 5 --reiniciar
python analisar.py
```

**Esperado:** três linhas `[01/3]`, `[02/3]`, `[03/3]`, cada uma mostrando
`SEM_AUMENTO`, `VAE` e `GAN` com um F1 em torno de 0,6–0,7. Depois o
`analisar.py` imprime o relatório e grava figuras em `resultados/figuras/`.

São 3 simulações e não 2 porque o Shapiro-Wilk precisa de pelo menos três
observações.

**Se der erro aqui, pare e me chame.** Não siga para o passo 4.

---

## 3. Revisão do código (sua tarefa T1 — antes da execução oficial)

A execução oficial leva ~1h, então vale revisar antes. O professor vai abrir o
código e perguntar **por que** cada escolha está ali — e ele pode perguntar a
qualquer um de nós dois.

Leia `docs/GUIA_DEFESA.md` inteiro e marque o que não souber responder.
Percorra nesta ordem:

- [ ] `config.py` — os dois geradores têm o mesmo orçamento de treino?
- [ ] `src/dados.py` — como a escassez é criada e por que a mesma seed pareia
- [ ] `src/vae.py` — reparametrização, por que `logvar`, por que BCE com `sum`
- [ ] `src/gan.py` — `.detach()`, LeakyReLU, label smoothing 0,9, betas (0.5, 0.999)
- [ ] `src/extrator.py` — por que congelada, por que 112 px, por que 3 canais
- [ ] `src/ensemble.py` — por que `class_weight="balanced"` (ver `docs/adr/ADR-0002`)
- [ ] `src/estatistica.py` — Shapiro → t pareado ou Wilcoxon, bilateral, dz, IC

Rode também os dois diagnósticos e entenda o que mostram:

```bash
python -m src.diagnostico --epocas 50     # separabilidade real vs sintético
python -m src.qualitativo --epocas 50     # projeção t-SNE das três nuvens
```

Os dois sustentam o achado central: o aumento sintético **não** melhora o
desempenho, porque as imagens geradas formam um agrupamento próprio em vez de
se misturarem às reais. Veja a seção 6 do README.

**Saída desta etapa:** sua lista de dúvidas para eu responder.

---

## 4. Execução oficial (~70 min)

Só depois do passo 3.

```bash
caffeinate -i python simular.py --reiniciar
python analisar.py
python -m src.qualitativo
```

`caffeinate -i` impede o Mac de dormir no meio. Deixe a máquina dedicada.

**Se interromper** (fechou o terminal, acabou a bateria): rode de novo **sem**
`--reiniciar` que ele continua da seed onde parou. O CSV é gravado ao fim de
cada simulação.

> **Nunca** misture orçamentos de treino no mesmo CSV. Se você rodou o teste
> rápido com `--epocas 5`, a execução oficial precisa do `--reiniciar`. O
> script detecta a mistura e se recusa a continuar, com instrução na tela.

### Tempo esperado

~111 s por seed × 30 ≈ **56 min**, até ~70 min com folga. O gargalo é o
ensemble (~20 s × 3 por seed), não o treino dos geradores (~47 s para as 200
épocas das duas redes somadas).

---

## 5. Me mandar os resultados

Quando terminar, envie estes dois arquivos:

```
resultados/simulacoes.csv
resultados/relatorio_estatistico.txt
```

Com eles eu escrevo o paper no template IEEE e monto a apresentação com os
números reais.

Manda também a conclusão que apareceu na tela: **rejeitamos ou não rejeitamos
H0?** É o que define a mensagem central do paper.

---

## Se algo der errado

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| `ModuleNotFoundError` | venv não ativado | `source .venv/bin/activate` |
| `command not found: python` | usar `python3` | ou ativar o venv |
| "já tem N seed(s) com outro número de épocas" | CSV misturado | rode com `--reiniciar` |
| "Só N simulação(ões) no CSV" | menos de 3 seeds | rode pelo menos 3 |
| Download travado | rede | apague `dados/` e rode de novo |
| Muito lento | rodando em CPU | confira se a 1ª linha diz `Dispositivo: mps` |

Qualquer saída que não bata com o descrito aqui, me chame antes de seguir.
