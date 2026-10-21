# Estudo da T1 — Leo

Guia de revisão do código, item por item, ancorado no código real. Para cada
ponto: onde está, o que responder e a pegadinha provável. Leia uma vez e depois
tente responder cada pergunta sem olhar.

Estado do ambiente validado: venv com Python 3.12, GPU detectada (`mps`), teste
rápido e os dois diagnósticos rodaram de ponta a ponta.

> **Nota (pós-execução oficial):** os números de FD e separabilidade abaixo
> foram escritos com base em treinos curtos. A execução oficial (200 épocas)
> inverteu parte dessa leitura — ver [ADR-0003](adr/ADR-0003-conclusoes-de-treinos-curtos-nao-se-sustentaram.md)
> e [STATUS_ATUAL.md](STATUS_ATUAL.md). Os itens 1 a 7 (o *porquê* de cada
> escolha de código) seguem válidos; a interpretação dos resultados é a do
> cenário 3 (rejeita H0, GAN vence com ganho pequeno).

---

## 1. `config.py` — os dois geradores têm o mesmo orçamento?

**Sim.** `EPOCAS_GERADOR = 200` e `TAMANHO_LOTE = 64` (config.py:46-47) são
compartilhados por VAE e GAN. Garante justiça: se a GAN treinasse mais, uma
vitória dela poderia ser só computação extra.

**Pegadinha:** `DIM_LATENTE` (16 vs 64) e `LR` diferem. São internos de cada
arquitetura — a dimensão latente ótima e o `lr` recomendado (DCGAN: 2e-4)
diferem por design. O que precisa ser igual é o *orçamento* (épocas, lote, nº
de imagens reais, nº de sintéticas), não os hiperparâmetros intrínsecos.
Igualar o `lr` sabotaria uma das redes.

---

## 2. `src/dados.py` — escassez e pareamento

- **Escassez** (dados.py:49-50): sorteia 300 normais + 1500 pneumonias, sem
  reposição. Recria o caso real (hospital com poucos exames da classe rara),
  onde gerar dados faz sentido. Com o dataset inteiro o aumento seria
  irrelevante.
- **Pareamento** (dados.py:46): `np.random.default_rng(seed)` cria um gerador
  próprio, que depende só da seed — não de quantos números aleatórios o resto
  do código já consumiu. VAE e GAN da simulação i recebem a mesma seed, logo
  treinam com as mesmas 300 imagens. É isso que torna o desenho pareado.

---

## 3. `src/vae.py` — reparametrização, `logvar`, BCE com `sum`

- **Duas cabeças** (vae.py:41-42): `fc_mu` e `fc_logvar`. O VAE mapeia cada
  imagem para uma distribuição gaussiana, não um ponto — espaço latente
  contínuo, permite sortear `z ~ N(0,I)` para gerar.
- **Por que `logvar`** (vae.py:72): variância tem que ser positiva, mas uma
  camada linear produz qualquer real. Prevendo o log, qualquer saída é válida e
  recupera-se `sigma = exp(0.5*logvar)`, sempre positivo.
- **Reparametrização** (vae.py:62-74): não dá para derivar através de um
  sorteio. Reescreve-se `z = mu + sigma*epsilon`, com `epsilon ~ N(0,1)`. O
  sorteio fica isolado em `epsilon` (não depende dos pesos); o gradiente flui
  por `mu` e `sigma`.
- **BCE com `reduction="sum"`** (vae.py:102): a KL é soma sobre as 16 dimensões
  latentes. Se a BCE fosse média dos 784 pixels, ficaria ~784x menor e a KL
  dominaria -> imagens borradas demais. Somar mantém os dois termos na mesma
  escala. Depois divide por `n` (lote) só para não depender do tamanho do lote.

**Pegadinha "por que o VAE sai borrado?":** a perda pixel-a-pixel, em dúvida
entre duas posições de uma borda, minimiza o erro médio produzindo a média das
duas -> borda borrada. É inerente ao VAE, não um bug.

---

## 4. `src/gan.py` — `.detach()`, LeakyReLU, label smoothing, betas

- **`.detach()`** (gan.py:131-132): ao treinar o D, só os pesos dele mudam.
  `falsas.detach()` corta o grafo para o gradiente do D não voltar ao G. No
  passo 2 (gan.py:141) usa-se `falsas` sem detach, porque aí o gradiente deve
  chegar ao G.
- **LeakyReLU no D, ReLU no G** (gan.py:58): o G só aprende pelo gradiente que
  atravessa o D. Com ReLU no D, neurônios negativos devolveriam gradiente zero e
  cortariam esse sinal. LeakyReLU(0,2) deixa passar uma fração no negativo.
- **Label smoothing 0,9** (gan.py:123): alvo das reais é 0,9 em vez de 1,0, para
  impedir o D de ficar confiante demais (D saturado manda gradiente ~zero ao G).
- **`betas=(0.5, 0.999)`** (config.py:57, gan.py:108-109): o momento padrão 0,9
  acumula demais a direção dos gradientes passados; num jogo onde o alvo muda a
  cada passo, isso oscila. DCGAN achou que 0,5 estabiliza.

**Bônus:** D termina em `Linear` sem Sigmoid (gan.py:66) porque usa
`BCEWithLogitsLoss` (gan.py:110), que aplica a sigmoid internamente com o log
(mais estável perto de 0/1). `Tanh` na saída do G (gan.py:42): saída [-1,1]
centrada em zero ajuda o treino; por isso as reais viram [-1,1] (gan.py:99).

---

## 5. `src/extrator.py` — congelada, 112 px, 3 canais

- **Congelada** (extrator.py:43-44): (1) overfitting — ajustar 11M de pesos com
  300 imagens; (2) custo — 30x3 = 90 fine-tunings; congelada, extrai os
  embeddings reais uma vez (cache); (3) controle — com a ResNet fixa, só o
  gerador varia entre cenários.
- **`fc = nn.Identity()`** (extrator.py:42): a `fc` classifica as 1000 classes
  do ImageNet; trocada por identidade, a saída vira o vetor de 512 (o embedding).
- **28 -> 112 px** (extrator.py:53): a ResNet reduz a resolução por 32; com 28 px
  a imagem chegaria à última conv como 1 pixel. Os filtros foram treinados em
  224x224; 112 dá escala parecida pela metade do custo de 224.
- **3 canais** (extrator.py:56): a ResNet espera RGB; a radiografia é cinza ->
  repete o canal 3x (mantém a informação sem inventar cor).

**Pegadinha "por que `eval()` se não treina?"** (extrator.py:46): o BatchNorm se
comporta diferente em treino/avaliação. Em `eval()` usa as estatísticas fixas do
ImageNet, deixando cada embedding determinístico. Normalização (extrator.py:33-34,57)
com média/desvio do ImageNet porque a rede foi treinada assim.

---

## 6. `src/ensemble.py` — `class_weight="balanced"` (a pergunta mais perigosa)

- **O problema** (config.py:75-84): treino 1500 pneumonias x 300 normais, decisão
  em limiar fixo 0,5. Sem peso, a revocação da classe normal travava em ~0,53 com
  precisão 0,97 — só acertava os normais óbvios. Varrendo o limiar dava F1 0,82,
  mas os medidos ficavam em 0,68. Isso comprimia os dois cenários contra o mesmo
  teto e tornava o teste insensível à diferença que ele quer detectar.
- **A correção**: `class_weight="balanced"` recoloca as classes no mesmo patamar
  da perda (F1 0,683 -> 0,727 na seed 42).
- **A frase que blinda**: é aplicado de forma IDÊNTICA nos três cenários, então
  não favorece VAE nem GAN — corrige o INSTRUMENTO DE MEDIDA, não o TRATAMENTO.

Outros pontos: três famílias que erram diferente (ensemble.py:4-12); votação
suave = média de probabilidades (dá peso à confiança e fornece prob. para a AUC,
ensemble.py:14-16); `StandardScaler` só na Logística (ensemble.py:33-36) porque
árvores só comparam com limiares.

---

## 7. `src/estatistica.py` — Shapiro -> t/Wilcoxon, bilateral, dz, IC

Fluxo em estatistica.py:47-69:

1. **Shapiro-Wilk nas 30 diferenças** (L50): testa normalidade das diferenças
   (não das amostras), pois o t pareado é um t de uma amostra sobre as diferenças.
2. **Normal -> t pareado; senão -> Wilcoxon** (L53-58). Wilcoxon usa postos, não
   exige normalidade.
3. **Bilateral**: não se sabia qual gerador venceria; fixar a direção depois dos
   dados inflaria a chance de significância.
4. **`dz`** (L64): p-valor diz se há diferença; dz diz quão grande em desvios
   (0,2 peq / 0,5 méd / 0,8 grande).
5. **IC 95%** (L67-69): se não rejeitar H0, o IC mostra a diferença máxima
   compatível com os dados. Não rejeitar H0 nao prova H0.

**Pegadinha métricas secundárias**: são exploratórias porque cada teste a 5% tem
5% de falso positivo; cinco testes inflam o erro. A decisão apoia-se só na
métrica principal (`f1_minoria`), definida antes de rodar (config.py:91).

No resultado oficial: Shapiro p=0,69 -> t pareado, p<0,000001 -> **rejeita H0**
(GAN > VAE, dz=-2,40).

---

## As 3 perguntas difíceis (decorar) — atualizadas com os números oficiais

1. **"O aumento melhorou?"** Marginalmente, e só com a GAN: GAN 0,717 vs linha de
   base 0,710 (+0,007); o VAE piora (0,685). O ganho prático é pequeno.
2. **"VAE ou GAN é melhor?"** A GAN, de forma estatisticamente significativa
   (p<0,000001, dz=-2,40). O dz é enorme porque o pareamento remove a variância
   do sorteio; em valor absoluto são 0,032 de F1.
3. **"Fidelidade (FD) e desempenho concordam?"** Nesta execução sim: a GAN é mais
   fiel (FD 160 vs 223) e melhor no F1. Mas atenção ao AUC, onde o VAE vence
   (0,928 vs 0,919) — a discussão tem que tratar essa divergência entre métricas.

**Sobre a separabilidade (0,997 VAE / 0,986 GAN):** ainda alta — um classificador
linear separa sintético de real. Mas, diferente dos treinos curtos, a 200 épocas
a GAN já é fiel o bastante para ajudar marginalmente. A separabilidade vira
recomendação metodológica (medir antes de confiar no aumento), não "aumento é
inútil".

---

## Lista de dúvidas para o Guilherme (saída da T1)

1. **Mensagem central** (T4): fechada — cenário 3, "GAN vence, ganho pequeno".
2. **MPS não-determinístico**: a mesma seed varia na 3a/4a casa decimal; citar
   como limitação no paper. Os dois devem dar a mesma resposta se perguntados.
3. **`DIM_LATENTE` e `LR` diferentes entre VAE/GAN**: alinhar o argumento de que
   isso não quebra a justiça da comparação (ponto 1 da seção config). É a objeção
   mais fina que o professor pode levantar.
