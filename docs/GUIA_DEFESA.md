# Guia de defesa do código

O professor vai abrir o código e perguntar por que cada coisa está onde está. Este guia reúne as perguntas mais prováveis, organizadas por arquivo. Todo mundo do grupo deve conseguir responder qualquer uma delas sem ler a resposta.

## Roteiro para apresentar o código (cerca de 8 minutos)

1. **`config.py`**: mostrar que todo hiperparâmetro está num lugar só e que os dois geradores têm o mesmo orçamento de treino.
2. **`src/dados.py`**: a escassez simulada e por que a mesma seed garante o pareamento.
3. **`src/vae.py`**: codificador, as duas cabeças (mu e logvar), reparametrização, perda.
4. **`src/gan.py`**: gerador vs discriminador, as escolhas do DCGAN, o `.detach()`.
5. **`src/extrator.py`**: ResNet congelada, por que 112 pixels e 3 canais.
6. **`src/ensemble.py`**: votação suave com três famílias de modelos.
7. **`simular.py`**: o laço principal; destacar que VAE e GAN passam pelo MESMO código.
8. **`src/estatistica.py`**: Shapiro, t pareado ou Wilcoxon, d de Cohen.

---

## Desenho experimental

**Por que 30 simulações?**
Com 30 repetições a média amostral já se comporta de forma aproximadamente normal (Teorema Central do Limite) e o intervalo de confiança fica estreito. A figura `convergencia_f1` mostra isso: a faixa do IC se estreita à medida que somamos simulações. Uma única execução não diz nada, porque o resultado pode ter sido sorte do sorteio.

**O que muda de uma simulação para outra?**
A seed. Ela controla quais imagens reais são sorteadas, a inicialização dos pesos dos geradores, a ordem dos lotes e a aleatoriedade das árvores do ensemble. O conjunto de teste é sempre o oficial do MedMNIST (624 imagens), para que todas as simulações sejam avaliadas na mesma régua.

**Por que só a técnica A muda entre os cenários?**
Para isolar a causa. Se mudássemos o gerador e também o classificador, não daria para saber qual mudança causou a diferença. É o mesmo princípio de um experimento controlado.

**Por que o cenário SEM_AUMENTO, se ele não entra no teste?**
É a linha de base. Responde "gerar dados ajuda?" antes de responder "qual gerador é melhor?". Se nenhum dos dois superar a referência, isso também é um resultado e vai para a discussão do paper.

**Por que criar escassez artificial em vez de usar o dataset inteiro?**
Com todas as imagens reais, a classe normal já tem mais de mil exemplos e o aumento sintético faria pouca diferença. Reduzindo para 300 criamos o cenário em que a geração de dados é de fato necessária, que é o caso de uso real (hospitais com poucos exames rotulados).

**Por que os dois geradores treinam o mesmo número de épocas?**
Justiça na comparação. Se a GAN treinasse mais, uma eventual vitória poderia ser só por ter recebido mais computação.

---

## `src/vae.py`

**Qual a diferença entre um autoencoder comum e um VAE?**
O autoencoder comum mapeia cada imagem para um ponto do espaço latente. Entre esses pontos pode haver "buracos" que não decodificam em nada com sentido. O VAE mapeia cada imagem para uma distribuição gaussiana, e o termo KL força todas essas gaussianas a ficarem perto de N(0, I). Assim o espaço latente fica contínuo e podemos sortear um z qualquer de N(0, I) para gerar uma imagem nova.

**Por que a rede prevê `logvar` e não a variância direto?**
A variância tem que ser positiva. Uma camada linear pode produzir qualquer número, inclusive negativo. Prevendo o log, qualquer valor real é válido e recuperamos a variância com `exp`, que é sempre positivo.

**O que é o truque da reparametrização e por que ele existe?**
Não dá para calcular gradiente através de um sorteio aleatório. Reescrevemos z = mu + sigma × epsilon, com epsilon sorteado de N(0, 1). O sorteio fica isolado em epsilon, que não depende dos pesos, e o gradiente passa normalmente por mu e sigma.

**Por que Sigmoid na última camada do decodificador?**
Os pixels estão em [0, 1] e a Sigmoid produz exatamente esse intervalo. Além disso, a perda BCE só faz sentido com valores entre 0 e 1.

**Por que ReLU nas camadas internas?**
É simples, barata e não satura para valores positivos, então o gradiente não some em redes com várias camadas. Para um VAE pequeno como este não há motivo para algo mais elaborado.

**O que cada termo da perda faz?**
A BCE mede quão fiel é a reconstrução. A KL mede quão longe a distribuição de cada imagem está de N(0, I). Sem a KL o modelo vira um autoencoder comum e perde a capacidade de gerar. Sem a BCE ele ignoraria a imagem e só produziria ruído.

**Por que `reduction="sum"` na BCE?**
A KL é uma soma sobre as 16 dimensões latentes. Se a BCE fosse uma média sobre os 784 pixels, ela ficaria cerca de 784 vezes menor e a KL dominaria, gerando imagens borradas demais. Somar mantém os dois termos na escala do artigo original.

**Por que as imagens do VAE saem borradas?**
A perda de reconstrução pixel a pixel penaliza erros grandes. Quando a rede está em dúvida entre duas posições de uma borda, a resposta que minimiza o erro médio é a média das duas, que é uma borda borrada.

**Para que serve `@torch.no_grad()`?**
Na geração não vamos treinar nada, então não precisamos guardar o grafo de operações para calcular gradientes. Economiza memória e tempo.

---

## `src/gan.py`

**Como a GAN aprende sem uma perda de reconstrução?**
O Discriminador aprende a separar reais de falsas. O Gerador recebe o gradiente do Discriminador e se ajusta para que suas imagens sejam classificadas como reais. É um jogo de soma zero: cada rede melhora porque a outra melhorou.

**Por que Tanh na saída do Gerador e não Sigmoid?**
Recomendação do DCGAN. A Tanh é centrada em zero (saída em [-1, 1]), o que ajuda o treino. Por isso as imagens reais são convertidas de [0, 1] para [-1, 1] antes do treino (`* 2.0 - 1.0`) e as geradas são convertidas de volta.

**Por que LeakyReLU no Discriminador e ReLU no Gerador?**
O Gerador só aprende pelo gradiente que atravessa o Discriminador. Se o Discriminador tivesse ReLU, neurônios com entrada negativa devolveriam gradiente zero e cortariam esse sinal. A LeakyReLU deixa passar uma fração (0,2) mesmo para entradas negativas. No Gerador a ReLU funciona bem, como mostra o artigo do DCGAN.

**Por que BatchNorm? E por que não na primeira camada do Discriminador?**
O BatchNorm mantém as ativações com média e variância controladas, o que estabiliza o jogo entre as duas redes. Na primeira camada do Discriminador o artigo do DCGAN observou que normalizar a imagem de entrada crua causava oscilação no treino.

**Para que serve o `.detach()` no passo do Discriminador?**
Quando treinamos o Discriminador, só os pesos dele devem mudar. O `.detach()` corta o grafo, então o gradiente da perda do Discriminador não volta até os pesos do Gerador.

**Por que o Discriminador termina em `Linear` sem Sigmoid?**
Usamos `BCEWithLogitsLoss`, que aplica a Sigmoid internamente junto com o log. Fazer as duas operações juntas evita erros numéricos quando a probabilidade fica muito perto de 0 ou de 1.

**O que é o label smoothing de 0,9?**
O alvo para imagens reais vira 0,9 em vez de 1,0. Isso impede o Discriminador de ficar confiante demais. Um Discriminador confiante demais satura e manda gradiente quase nulo para o Gerador.

**Por que o Gerador usa alvo 1 (e não minimiza log(1 menos D(G(z))))?**
É a formulação "non saturating" proposta pelo próprio Goodfellow. No começo do treino o Discriminador rejeita facilmente as falsas e a formulação original dá gradiente quase zero. Maximizar log D(G(z)) dá gradientes fortes justamente nessa fase.

**Por que inicializar os pesos com N(0, 0,02)?**
Recomendação do DCGAN. Pesos pequenos evitam que uma das redes domine nas primeiras iterações.

**Por que `betas=(0.5, 0.999)` no Adam?**
O momento padrão (0,9) acumula demais a direção dos gradientes passados. Num jogo entre duas redes, onde o alvo muda o tempo todo, isso causa oscilação. O DCGAN encontrou que 0,5 estabiliza o treino.

**O que é mode collapse?**
Quando o Gerador descobre algumas imagens que enganam o Discriminador e passa a produzir só variações delas, perdendo diversidade. É um dos motivos pelos quais esperamos maior variância da GAN entre as 30 simulações.

**Por que as curvas de perda da GAN não descem como as do VAE?**
Porque as duas perdas competem. Quando o Gerador melhora, a perda do Discriminador sobe e vice versa. Uma GAN saudável oscila em torno de um equilíbrio. Perda do Discriminador indo a zero indica que ele venceu e o Gerador parou de aprender.

---

## `src/extrator.py`

**Por que usar uma rede treinada no ImageNet para radiografias?**
As primeiras camadas de uma CNN aprendem bordas, texturas e formas genéricas, úteis para qualquer imagem. Essa é a ideia central do Transfer Learning: reaproveitar conhecimento de uma tarefa com milhões de exemplos numa tarefa com poucos.

**Por que congelar em vez de fazer fine tuning?**
Três motivos: poucas imagens reais (risco de overfitting com 11 milhões de pesos), custo (seriam 90 fine tunings) e controle experimental (com a ResNet fixa, só o gerador varia entre cenários).

**Por que `model.fc = nn.Identity()`?**
A camada `fc` classifica as 1000 classes do ImageNet, que não nos interessam. Trocando por uma identidade, a saída da rede passa a ser o vetor de 512 características anterior a ela.

**Por que ampliar de 28 para 112 pixels?**
A ResNet reduz a resolução por 32 até o fim. Com 28 pixels a imagem chegaria à última camada convolucional reduzida a um único pixel, e os filtros foram treinados para estruturas de imagens 224x224. Com 112 a rede trabalha numa escala parecida com a do treino, pela metade do custo de 224.

**Por que repetir o canal três vezes?**
A ResNet espera RGB. A radiografia é cinza; repetir o canal mantém a informação sem inventar cor.

**Por que normalizar com a média e o desvio do ImageNet?**
A rede foi treinada com entradas normalizadas por esses valores. Sem isso os pixels chegam numa escala que ela nunca viu e as ativações saem distorcidas.

**Por que `model.eval()` se não estamos treinando?**
O BatchNorm se comporta diferente em treino e avaliação. Em `eval()` ele usa as estatísticas fixas aprendidas no ImageNet em vez das do lote atual, o que deixa o embedding de cada imagem determinístico.

---

## `src/ensemble.py`

**Por que esses três modelos?**
São de famílias diferentes: um linear, um de bagging e um de boosting. Erram de formas diferentes, e é a diversidade dos erros que faz o ensemble funcionar.

**Qual a diferença entre bagging e boosting?**
Bagging (Random Forest) treina árvores em paralelo em amostras diferentes e tira a média, reduzindo a variância. Boosting treina árvores em sequência, cada uma corrigindo os erros da anterior, reduzindo o viés.

**Votação suave vs votação dura?**
A dura conta votos. A suave tira a média das probabilidades, então um modelo muito confiante pesa mais. A suave também fornece probabilidades, necessárias para a AUC.

**Por que `StandardScaler` só na Regressão Logística?**
A otimização da Regressão Logística depende da escala das variáveis. Árvores só comparam cada variável com um limiar, então a escala não muda nada para elas.

---

## `src/metricas.py`

**Por que F1 da classe normal e não acurácia?**
O teste tem 62,5% de pneumonia. Um modelo que diz "pneumonia" para tudo tem 62,5% de acurácia sem aprender nada. O F1 da classe normal só fica alto se o modelo reconhecer bem justamente a classe rara.

**O que a AUC mede de diferente?**
A AUC não depende do limiar de 0,5. Ela mede a probabilidade de o modelo dar uma nota maior a um caso de pneumonia do que a um normal sorteados ao acaso. É uma medida de ordenação.

**O que é a distância de Fréchet (FD ResNet)?**
A mesma fórmula do FID: modela os embeddings reais e os sintéticos como duas gaussianas e mede a distância entre elas usando média e covariância. Quanto menor, mais parecidas com as reais são as sintéticas. Usamos a ResNet18 em vez da InceptionV3 do FID original para reaproveitar o extrator do projeto, por isso o nome FD ResNet e não FID.

**Por que pegar só a parte real da raiz da matriz?**
Por erro numérico de ponto flutuante, `sqrtm` pode devolver uma parte imaginária minúscula. Ela não tem significado e é descartada.

---

## `src/estatistica.py`

**Por que teste pareado?**
Na simulação i os dois cenários usam a mesma seed, as mesmas imagens reais e o mesmo teste. Analisar a diferença dentro de cada par remove a variação causada pelo sorteio dos dados, o que deixa o teste bem mais sensível.

**Por que testar a normalidade das diferenças e não das amostras?**
O teste t pareado é, na prática, um teste t de uma amostra sobre as diferenças. A suposição de normalidade é sobre elas.

**Quando usamos Wilcoxon?**
Quando o Shapiro Wilk rejeita a normalidade das diferenças (p < 0,05). O Wilcoxon usa os postos das diferenças, não os valores, então não exige normalidade.

**Por que bilateral?**
Não sabíamos de antemão qual gerador venceria. Escolher a direção depois de ver os dados inflaria artificialmente a chance de "achar" significância.

**O que o p valor significa exatamente?**
A probabilidade de observar uma diferença pelo menos tão grande quanto a observada SE H0 fosse verdadeira. Não é a probabilidade de H0 ser verdadeira.

**Por que reportar o d de Cohen além do p valor?**
Com amostras grandes, diferenças minúsculas e sem importância prática podem ser significativas. O d de Cohen mostra o tamanho da diferença em desvios padrão: 0,2 pequeno, 0,5 médio, 0,8 grande.

**E se não rejeitarmos H0?**
É um resultado válido: dizemos que não há evidência de diferença com 30 simulações, olhando o IC de 95% para mostrar o tamanho máximo de diferença compatível com os dados. Não dizemos que "os dois são iguais", porque não rejeitar H0 não prova H0.

**Por que as métricas secundárias são "exploratórias"?**
Cada teste a 5% tem 5% de chance de falso positivo. Fazendo cinco testes, a chance de pelo menos um falso positivo cresce. A decisão do estudo se apoia só na métrica principal, definida antes de rodar.

---

## Resultados oficiais (perguntas difíceis)

As perguntas mais prováveis da arguição. Todo mundo precisa saber responder
estas três, porque elas vão direto ao resultado.

**Qual foi a conclusão: rejeitaram H0?**
Sim. O cenário com DCGAN teve F1 da classe normal superior ao do VAE: 0,717
contra 0,685. Diferença média de −0,032 a favor da GAN, IC95% [−0,037; −0,027],
p < 10⁻⁶, dz = −2,40. O Shapiro-Wilk não rejeitou normalidade das diferenças
(p = 0,69), então aplicamos o t pareado.

**O dz de 2,40 é enorme. A GAN é muito melhor então?**
Não, e essa é a parte interessante. O efeito é grande em desvios padrão porque
o desenho pareado remove a variância do sorteio dos dados: os dois cenários
veem exatamente as mesmas imagens. Isso torna detectável uma diferença
sistemática que, em termos absolutos, é pequena — 0,032 de F1. Mais: a GAN
supera a linha de base *sem aumento nenhum* por apenas 0,007, e o VAE a piora
em 0,025. Significância estatística e relevância prática divergem aqui.

**Então o aumento sintético funcionou ou não?**
Praticamente não. A GAN empata com não fazer nada e o VAE atrapalha. O
diagnóstico explica: uma regressão logística separa características sintéticas
de reais com acurácia 0,997 (VAE) e 0,986 (GAN). Até uma fronteira linear as
distingue. As imagens geradas não entram como exemplos extras da classe normal;
formam um agrupamento próprio que as árvores isolam. O balanceamento 1:1 é
nominal, não efetivo.

**Se os dois são igualmente separáveis, por que a GAN vai melhor?**
Pela norma da característica. A GAN reproduz a magnitude de ativação das
imagens reais (100,5% da norma real), o VAE fica em 79,4%. A perda de
reconstrução pixel a pixel do VAE produz imagens suavizadas, e a ResNet
responde com ativações mais fracas a textura pobre. A GAN é separável, mas
menos destoante.

**Por que a AUC favorece a linha de base?**
Porque o aumento sintético mexe mais no limiar de decisão do que na capacidade
de ordenar casos. A AUC não depende de limiar; o F1 depende. Isso reforça que o
ganho no F1 da GAN não vem de o modelo "entender melhor" a classe normal.

**Vocês chegaram a ter uma conclusão diferente antes?**
Sim, e isso está documentado no ADR-0003. Durante a validação do código rodamos
treinos curtos (3 a 20 épocas) e eles sugeriam o oposto: VAE mais fiel, GAN com
textura pobre, nuvens sintéticas sempre separadas. A 200 épocas três dessas
afirmações se inverteram. A causa é que uma GAN interrompida cedo ainda não
alcançou o discriminador, enquanto o VAE converge mais rápido — o orçamento
curto criava uma vantagem artificial para o VAE. A regra que adotamos: execução
de verificação responde "o código roda?", não "qual gerador é melhor?".

**Por que o `class_weight="balanced"` no ensemble?**
Porque o limiar é fixo em 0,5 e o treino é desbalanceado em 1500 contra 300.
Sem o peso, a revocação da classe normal travava em 0,53 com precisão 0,97, e o
F1 alcançável por varredura de limiar era 0,82 contra 0,68 medidos. Isso
comprimia os dois cenários contra o mesmo teto. O peso é idêntico nos três
cenários: corrige o instrumento de medida, não o tratamento. Ver ADR-0002.

**Por que a referência da distância de Fréchet usa só as 300 sorteadas?**
Porque o gerador só viu essas 300. Usar todas as 1214 normais do split mediria
generalização, não fidelidade, e tornaria a referência idêntica nas 30
simulações, quebrando o pareamento.

**Por que a coluna `epocas` no CSV?**
Para impedir contaminação silenciosa. Um teste rápido com 5 épocas deixaria
linhas que a execução oficial de 200 pularia na retomada, misturando dois
orçamentos na mesma análise. O script detecta e se recusa a continuar.

---

## `src/utilidades.py` e `simular.py`

**Por que fixar quatro seeds diferentes?**
Python, NumPy, PyTorch CPU e PyTorch GPU têm geradores aleatórios independentes. Se faltar um, a execução deixa de ser reproduzível.

**Por que chamar `fixar_sementes(seed)` antes de CADA gerador?**
Para que VAE e GAN comecem do mesmo estado aleatório. Sem isso, o segundo cenário herdaria um estado diferente só por ter rodado depois do primeiro.

**Por que `np.random.default_rng(seed)` no sorteio dos dados?**
Ele cria um gerador próprio que depende só da seed passada, não de quantos números aleatórios o resto do código já consumiu. Garante que os dois cenários recebam exatamente as mesmas imagens.

**Por que os embeddings das imagens reais ficam em cache?**
A ResNet está congelada, então o embedding de uma imagem real nunca muda. Calcular uma vez e reaproveitar economiza a maior parte do tempo de cada simulação.

**Por que o CSV é gravado a cada simulação?**
Se o computador desligar no meio, as simulações já feitas não se perdem. Rodar de novo continua de onde parou.

**O que é MPS?**
Metal Performance Shaders, a forma do PyTorch usar a GPU dos Macs com chip Apple. Algumas operações no MPS não são 100% determinísticas, então a mesma seed pode variar na terceira ou quarta casa decimal. Vale citar isso como limitação no paper.
