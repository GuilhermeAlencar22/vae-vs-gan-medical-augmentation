"""
Configuração central do experimento.

Todos os hiperparâmetros ficam aqui, num lugar só. Assim:
  1. ninguém precisa caçar números "mágicos" espalhados pelo código;
  2. os dois cenários (VAE e GAN) usam exatamente os mesmos valores
     onde isso faz sentido, o que torna a comparação justa;
  3. o paper pode citar esta tabela diretamente na seção de Metodologia.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent
PASTA_DADOS = RAIZ / "dados"              # onde o PneumoniaMNIST é baixado
PASTA_RESULTADOS = RAIZ / "resultados"    # CSV das simulações e relatório
PASTA_FIGURAS = PASTA_RESULTADOS / "figuras"
ARQUIVO_CSV = PASTA_RESULTADOS / "simulacoes.csv"

# ---------------------------------------------------------------------------
# Desenho experimental
# ---------------------------------------------------------------------------
N_SIMULACOES = 30   # exigência da atividade: 30 repetições independentes
SEED_BASE = 42      # simulação i usa a seed SEED_BASE + i

# Classes do PneumoniaMNIST: 0 = normal, 1 = pneumonia.
# No conjunto original a classe "normal" já é minoritária (cerca de 1:3).
CLASSE_MINORIA = 0
CLASSE_MAIORIA = 1

# Escassez simulada: em cada simulação sorteamos poucas imagens normais
# reais. Isso cria o problema que a geração sintética precisa resolver.
N_MINORIA_TREINO = 300    # imagens NORMAIS reais disponíveis no treino
N_MAIORIA_TREINO = 1500   # imagens com PNEUMONIA reais disponíveis no treino

# Quantas imagens sintéticas gerar: o suficiente para balancear 1:1.
N_SINTETICAS = N_MAIORIA_TREINO - N_MINORIA_TREINO   # 1200

# ---------------------------------------------------------------------------
# Técnica A: geradores (VAE no cenário H0, DCGAN no cenário H1)
# ---------------------------------------------------------------------------
# Os dois geradores recebem o MESMO orçamento de treino (épocas e lote),
# para que a diferença medida venha da técnica e não de um treino mais longo.
EPOCAS_GERADOR = 200
TAMANHO_LOTE = 64

# VAE
DIM_LATENTE_VAE = 16
LR_VAE = 1e-3
BETA_KL = 1.0   # peso do termo KL; 1.0 é o VAE "clássico" de Kingma e Welling

# DCGAN
DIM_LATENTE_GAN = 64
LR_GAN = 2e-4          # valor recomendado no artigo original do DCGAN
BETAS_ADAM_GAN = (0.5, 0.999)
SUAVIZACAO_ROTULO = 0.9  # one sided label smoothing: rótulo "real" vira 0.9

# ---------------------------------------------------------------------------
# Técnica B: Transfer Learning (ResNet18 pré treinada no ImageNet)
# ---------------------------------------------------------------------------
# A ResNet foi treinada com imagens 224x224. Ampliamos de 28 para 112 para
# que os filtros das primeiras camadas encontrem estruturas num tamanho
# parecido com o que aprenderam, sem pagar o custo total de 224.
TAMANHO_ENTRADA_RESNET = 112
LOTE_EXTRACAO = 256

# ---------------------------------------------------------------------------
# Técnica C: Ensemble Learning (votação suave)
# ---------------------------------------------------------------------------
N_ARVORES_RF = 200
MAX_ITER_LR = 2000

# Pesos de classe inversamente proporcionais à frequência ("balanced").
# Sem isso, o ensemble treina com 1500 pneumonias contra 300 normais e o
# limiar fixo de 0,5 fica enviesado: a revocação da classe normal trava em
# torno de 0,53 com precisão 0,97, ou seja, o modelo quase só acerta os
# normais mais óbvios. O peso recoloca as duas classes no mesmo patamar da
# função de perda e libera a métrica do viés do limiar (F1 0,683 -> 0,727
# na seed 42, sem aumento sintético).
# Aplicado DE FORMA IDÊNTICA nos três cenários, então não favorece VAE nem
# GAN: é uma correção do instrumento de medida, não do tratamento.
PESO_CLASSE = "balanced"


# ---------------------------------------------------------------------------
# Estatística
# ---------------------------------------------------------------------------
ALFA = 0.05                    # nível de significância
METRICA_PRINCIPAL = "f1_minoria"   # métrica usada no teste de hipótese
