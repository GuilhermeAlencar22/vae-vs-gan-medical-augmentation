# ADR-0002: Peso de classe no ensemble e referência local da distância de Fréchet

- **Status:** aceito
- **Data:** 2026-10-14
- **Relacionado:** ADR-0001

## Contexto

Na revisão do código, antes da execução oficial, três problemas apareceram ao
inspecionar os resultados das simulações de verificação.

**1. O limiar fixo de 0,5 enviesava a métrica principal.** O ensemble treinava
com 1 500 pneumonias contra 300 normais e decidia pelo limiar padrão de 0,5. A
revocação da classe normal travava em torno de 0,53 com precisão 0,97: o modelo
só acertava os normais mais óbvios. Varrendo o limiar na seed 42, o F1
alcançável era 0,82 contra 0,68 medidos. O efeito comprimia os dois cenários
contra o mesmo teto e tornava o teste insensível à diferença que ele pretende
detectar.

**2. A referência da distância de Fréchet vazava informação.** O cálculo usava
todas as 1 214 imagens normais do split de treino, inclusive as que o gerador
daquela simulação nunca viu.

**3. O número de épocas não era registrado.** Um teste rápido com `--epocas 5`
deixava linhas no CSV que a execução oficial com 200 épocas pularia na
retomada, misturando dois orçamentos de treino na mesma análise.

## Decisão

**1.** Os três modelos do ensemble passam a usar `class_weight="balanced"`
(`config.PESO_CLASSE`). A correção é aplicada de forma **idêntica aos três
cenários**, então não favorece VAE nem GAN: corrige o instrumento de medida,
não o tratamento.

**2.** A distância de Fréchet passa a usar como referência apenas as imagens
da classe minoritária sorteadas naquela simulação. Também somamos
`epsilon = 1e-6` à diagonal das covariâncias, regularização usada na
implementação de referência do FID, porque estimar uma covariância 512×512 com
algumas centenas de amostras produz matriz singular e `sqrtm` devolvia uma raiz
imprecisa.

**3.** O CSV ganha a coluna `epocas`, e a retomada só considera concluídas as
seeds gravadas com o mesmo orçamento. Detectada mistura, o script para e pede
`--reiniciar`.

## Consequências

- F1 da linha de base subiu de 0,683 para 0,727 na seed 42, com revocação de
  0,53 para 0,59.
- A distância de Fréchet passou a medir fidelidade ao que o gerador realmente
  viu, e a referência deixou de ser idêntica nas 30 simulações, preservando o
  pareamento.
- Os avisos `LinAlgWarning: Matrix is singular` desapareceram.
- Resultados antigos são incompatíveis: um CSV sem a coluna `epocas` é
  rejeitado com instrução de usar `--reiniciar`.

## Nota: o aumento sintético não melhora o desempenho

A investigação revelou um achado que **não** é um defeito do código e que vai
para a discussão do paper. Somar 1 200 imagens sintéticas balanceia o treino em
1 500 × 1 500, mas o desempenho não melhora.

`src/diagnostico.py` mede a causa: uma regressão logística distingue embedding
real de embedding sintético com acurácia **1,0000**, isto é, até uma fronteira
linear separa as duas nuvens sem erro. As imagens sintéticas não funcionam como
exemplos adicionais da classe normal; formam um terceiro agrupamento que as
árvores isolam com um corte só. A norma média do embedding indica o porquê:
34,4 nas reais contra 22,5 (VAE) e 31,0 (GAN) — a ResNet responde com ativações
mais fracas a texturas pobres.

O balanceamento é **nominal, não efetivo**. Mantemos o desenho como está e
reportamos o achado, porque ele é o resultado mais interessante do estudo.
