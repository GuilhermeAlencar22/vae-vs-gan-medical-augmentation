# ADR-0003: Conclusões tiradas de treinos curtos não se sustentaram a 200 épocas

- **Status:** aceito
- **Data:** 2026-10-21
- **Relaciona-se a:** ADR-0002

## Contexto

Durante a revisão do código (antes da execução oficial), rodamos o pipeline com
orçamentos curtos — 3, 5, 15 e 20 épocas — apenas para validar que tudo
funcionava de ponta a ponta. Os números que apareceram nessas execuções foram
usados para escrever a seção 6 do README, a seção "perguntas difíceis" do
`GUIA_DEFESA.md` e a nota final do ADR-0002.

A execução oficial (30 seeds × 200 épocas, feita pelo Leo em 2026-10-21)
**contradisse três dessas afirmações**.

| Afirmação escrita com treinos curtos | Execução oficial (200 épocas) |
|---|---|
| FD favorece o VAE (mais fiel que a GAN) | **Invertido**: GAN 160,2 vs VAE 223,3 |
| Sintético forma "ilha separada" no t-SNE | A nuvem da **GAN se mistura** com as reais |
| GAN tem "textura pobre" (norma baixa) | Norma da GAN é **100,5%** da real |
| Aumento sintético não ajuda | Vale **só para o VAE**; a GAN empata com a linha de base |
| Não rejeitamos H0 | **Rejeitamos H0** (p < 10⁻⁶, dz = −2,40) |

## Decisão

Corrigimos todos os documentos para refletir os números oficiais, e registramos
aqui a causa do erro para que ele não se repita.

**A causa:** uma GAN a 15 épocas e uma GAN a 200 épocas não são o mesmo modelo.
O treino adversarial precisa de tempo para o gerador alcançar o discriminador;
interrompido cedo, ele produz exatamente os artefatos de textura pobre que
observamos e que nos levaram à conclusão errada. O VAE converge mais rápido,
então sofreu menos com o orçamento curto — e isso criou uma vantagem aparente
que desapareceu com o treino completo.

**A regra que adotamos:** execuções de verificação servem para responder "o
código roda sem quebrar?". Não servem para responder "qual gerador é melhor?".
Nenhuma afirmação sobre resultado entra em documento antes da execução oficial.

## Consequências

- README (seção 6), `GUIA_DEFESA.md` e a nota do ADR-0002 foram reescritos com
  os números de 200 épocas.
- O `STATUS_ATUAL.md` do Leo registra a transição entre as duas narrativas.
- A lição vai para a seção de discussão do paper: é evidência concreta de que
  comparar geradores sob orçamento insuficiente produz conclusão invertida.

## Nota sobre o que permaneceu verdadeiro

A **separabilidade alta** entre características reais e sintéticas se manteve
(0,997 para o VAE e 0,986 para a GAN a 200 épocas). O achado de que o
balanceamento é "nominal e não efetivo" continua válido e é a explicação central
do paper — o que mudou foi a atribuição: ele descreve bem o VAE, e apenas
parcialmente a GAN, que reproduz a textura real mas ainda assim não entrega
ganho relevante sobre a linha de base.
