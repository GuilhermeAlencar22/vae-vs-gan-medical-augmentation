# Apresentação — roteiro para 7 apresentadores

Resultado oficial: **rejeitamos H0, a DCGAN venceu** — mas o ganho prático é
pequeno. Essa tensão é o fio condutor da apresentação.

Cada pessoa é dona de um bloco, **mas o professor pergunta a qualquer um**.
Todos devem ler `GUIA_DEFESA.md` inteiro, não só a própria parte.

---

## Bloco 1 · Problema e motivação

**Slides:** 2 · **Tempo:** ~1,5 min

- Triagem de pneumonia por radiografia; serviço que começa a rotular tem poucos
  exames normais confirmados.
- Classificador treinado assim favorece a classe majoritária.
- **O gancho:** a acurácia esconde o problema. Com 62,5% de pneumonia no teste,
  responder "pneumonia" para tudo já dá 62,5% de acurácia sem aprender nada.
- Pergunta do trabalho: gerar imagens sintéticas da classe rara resolve? E qual
  gerador é melhor?

**Prepare-se para:** *por que esse problema importa?* · *por que não usar
acurácia?*

---

## Bloco 2 · Dados e desenho da escassez

**Slides:** 2 · **Tempo:** ~1,5 min

- PneumoniaMNIST (MedMNIST v2): 28×28, cinza, split oficial preservado.
- Treino 4.708 (25,8% normal) · Teste 624 (37,5% normal).
- **Escassez simulada:** sorteamos 300 normais e 1.500 pneumonias (1:5). Com as
  1.214 normais disponíveis, o aumento teria pouco a acrescentar — criamos o
  regime em que ele é de fato necessário.
- Cada gerador sintetiza 1.200 normais → 1.500 por classe.
- Métrica principal definida **antes** de rodar: F1 da classe minoritária.

**Prepare-se para:** *por que criar escassez artificial?* · *por que 300?* ·
*por que o teste nunca muda?*

---

## Bloco 3 · Técnica A — os geradores (o contrafactual)

**Slides:** 3 · **Tempo:** ~2,5 min

- **VAE:** comprime a imagem numa *distribuição* (μ, σ²), não num ponto. O
  termo KL mantém o espaço latente contínuo, o que permite amostrar z ~ N(0,I).
  Truque da reparametrização: z = μ + σ·ε, porque não há gradiente através de
  um sorteio.
- **DCGAN:** gerador e discriminador competem; sem perda de reconstrução.
  Escolhas do artigo: LeakyReLU no discriminador, BatchNorm, Tanh na saída,
  label smoothing 0,9, betas (0.5, 0.999).
- **Este é o contrafactual:** a única coisa que muda entre H0 e H1.

**Prepare-se para:** *por que `logvar` e não a variância?* · *para que serve o
`.detach()`?* · *por que LeakyReLU no discriminador?* · *o que é mode collapse?*

---

## Bloco 4 · Técnicas B e C — classificador comum

**Slides:** 2 · **Tempo:** ~2 min

- **Transfer Learning:** ResNet18 do ImageNet, `fc` → Identity, saída de 512
  características. **Congelada** por três razões: 300 imagens não sustentam o
  ajuste de 11 milhões de pesos; 90 ajustes finos seriam proibitivos; e com ela
  fixa a única variação entre cenários é o gerador.
- Imagens 28→112 px e canal replicado 3×, para aproximar o que a rede viu.
- **Ensemble:** votação suave entre regressão logística, random forest e
  gradient boosting. Famílias diferentes erram diferente; a média cancela parte
  dos erros. Suave e não dura porque usa a confiança e fornece probabilidade
  (necessária para a AUC).
- `class_weight="balanced"` nos três, **idêntico nos três cenários**.

**Prepare-se para:** *por que congelar?* · *por que 112 px?* · *por que repetir
o canal 3×?* · *por que o peso de classe não favorece ninguém?*

---

## Bloco 5 · Desenho experimental e hipóteses

**Slides:** 2 · **Tempo:** ~2 min

- 30 simulações, seeds 42–71. A seed controla sorteio, inicialização, ordem dos
  lotes e aleatoriedade do ensemble.
- **Pareado:** os dois cenários compartilham a seed, logo veem as mesmas
  imagens reais e o mesmo teste. É como medir o mesmo paciente antes e depois.
- H0: média(F1_VAE − F1_GAN) = 0 · H1: ≠ 0. **Bilateral**, porque não sabíamos
  quem venceria — fixar direção depois de ver os dados seria desonesto.
- Shapiro-Wilk nas diferenças → t pareado (se normal) ou Wilcoxon.
- Orçamento idêntico: 200 épocas para os dois geradores.

**Prepare-se para:** *por que 30?* · *por que pareado e não independente?* ·
*por que testar normalidade das diferenças e não das amostras?*

---

## Bloco 6 · Resultados

**Slides:** 3 · **Tempo:** ~2,5 min

| Métrica | Sem aumento | VAE | GAN |
|---|---|---|---|
| F1 (normal) | 0,710 | 0,685 | **0,717** |
| AUC | **0,939** | 0,928 | 0,919 |
| FD ResNet ↓ | — | 223,3 | **160,2** |

- **Rejeitamos H0:** diferença −0,032, IC95% [−0,037; −0,027], p < 10⁻⁶,
  dz = −2,40. Shapiro p = 0,69 → t pareado.
- Figuras: boxplot por cenário · linhas pareadas seed a seed · histograma das
  diferenças · convergência da média com IC.
- **Não pare no p.** Diga em seguida: a GAN supera "não fazer nada" por 0,007.

**Prepare-se para:** *o que o p valor significa exatamente?* · *por que a AUC
favorece a linha de base?*

---

## Bloco 7 · Discussão e conclusão

**Slides:** 3 · **Tempo:** ~3 min

Este é o bloco mais importante. É aqui que o trabalho mostra leitura crítica.

- **Efeito grande, ganho pequeno.** O dz de 2,40 vem do pareamento, que remove
  a variância do sorteio. Em valor absoluto são 0,032 de F1.
- **Por quê:** separabilidade 0,997 (VAE) e 0,986 (GAN). Mostrar o t-SNE e a
  tabela de normas. O balanceamento é **nominal, não efetivo**.
- **A diferença entre geradores:** a GAN reproduz a textura real (norma 100,5%),
  o VAE suaviza (79,4%). Por isso a GAN degrada menos.
- **Honestidade metodológica:** nossas conclusões de treinos curtos se
  inverteram a 200 épocas (ADR-0003). Execução de verificação responde "o
  código roda?", não "qual gerador é melhor?".
- **Recomendação:** medir separabilidade antes de alegar ganho por aumento
  generativo. É barato — basta um classificador linear.
- **Limitações:** 28×28, extratora congelada, duas classes, MPS não
  determinístico.

**Prepare-se para:** *então o trabalho deu em nada?* (não — o achado é que o
ganho prometido não se materializa, e sabemos por quê) · *o que fariam
diferente?*

---

## Demonstração do código (~3 min, ao fim)

O professor avisou que **lê o código**. Deixe pronto:

1. `config.py` — tudo num lugar só, mesmo orçamento para os dois geradores.
2. `src/vae.py` → `reparametrizar` — breakpoint aqui mostra os tensores ao vivo.
3. `src/gan.py` → o `.detach()` no passo do discriminador.
4. `src/estatistica.py` → a escolha entre t pareado e Wilcoxon.
5. `python -m src.diagnostico` — rodar ao vivo e mostrar a separabilidade.

O `.vscode/launch.json` já tem a configuração **Teste rápido** para debugar.

---

## Checklist final

- [ ] Figuras em PNG exportadas de `resultados/figuras/`
- [ ] Slide de metodologia **com fluxograma** (padrão do slide de referência)
- [ ] Ensaio cronometrado na ordem real
- [ ] Cada apresentador respondeu 3 perguntas sorteadas do `GUIA_DEFESA.md`
- [ ] Código rodando numa máquina, pronto para a demonstração
- [ ] Todos sabem responder: *"por que o aumento sintético não ajudou?"*
