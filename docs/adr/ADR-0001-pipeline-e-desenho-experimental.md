# ADR-0001: Pipeline de três técnicas e desenho pareado de 30 simulações

- **Status:** aceito
- **Data:** 2026-10-14

## Contexto

A atividade exige comparar dois cenários formados por três técnicas vistas em
aula, trocando apenas uma delas (o contrafactual), e decidir entre a hipótese
nula e a alternativa com 30 simulações.

O grupo já havia trabalhado com AutoEncoders, então o VAE é o ponto de partida
natural para a técnica de geração sintética.

## Decisão

**Problema.** Classificação binária de radiografias de tórax (normal vs
pneumonia) com escassez na classe normal, usando PneumoniaMNIST (MedMNIST v2).

**Três técnicas.**

- **A, geração sintética:** VAE convolucional (cenário H0) ou DCGAN (H1).
- **B, Transfer Learning:** ResNet18 pré-treinada no ImageNet, congelada,
  usada como extratora de embeddings de 512 dimensões.
- **C, Ensemble Learning:** votação suave entre Regressão Logística, Random
  Forest e Histogram Gradient Boosting sobre os embeddings.

**Contrafactual.** Troca apenas da técnica A, espelhando o exemplo dado em
aula (GAN como alternativa ao gerador original).

**Desenho estatístico.** 30 simulações pareadas; a simulação *i* usa a mesma
seed nos dois cenários, o que fixa as mesmas imagens reais, o mesmo ensemble e
o mesmo conjunto de teste. Métrica principal definida *antes* de rodar: F1 da
classe normal. Shapiro-Wilk nas diferenças decide entre teste t pareado e
Wilcoxon. Reportamos p valor, d de Cohen (dz) e IC de 95%.

**Métrica principal.** F1 da classe minoritária, e não acurácia, porque o
conjunto de teste tem 62,5% de pneumonia e um classificador trivial já
alcançaria essa acurácia sem aprender nada.

## Consequências

**Positivas.**

- Congelar a ResNet permite calcular os embeddings das imagens reais uma única
  vez. Cada simulação só treina o gerador e o ensemble, o que viabiliza as 30
  repetições em cerca de uma hora.
- Com a ResNet fixa, a única fonte de variação entre os cenários é o gerador,
  o que isola a causa.
- O pareamento remove a variação do sorteio dos dados e aumenta a sensibilidade
  do teste.

**Negativas e limitações.**

- 28×28 limita a qualidade possível dos geradores. Ver ADR-0002.
- Sem fine tuning, o desempenho absoluto fica abaixo do que a literatura
  reporta para PneumoniaMNIST. Aceitável: o objetivo é comparar geradores, não
  maximizar a métrica.
- No backend MPS algumas operações não são determinísticas, então a mesma seed
  pode variar na terceira ou quarta casa decimal.

## Alternativas consideradas

- **Fine tuning da ResNet em vez de congelar.** Rejeitado: 90 fine tunings
  (30 seeds × 3 cenários), risco de overfitting com 300 imagens e a própria
  rede viraria fonte extra de variação.
- **CNN treinada do zero como classificador.** Rejeitado: não usaria Transfer
  Learning, uma das três técnicas exigidas.
- **Teste t para amostras independentes.** Rejeitado: desperdiça a estrutura
  pareada e é menos sensível.
