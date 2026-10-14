"""
Técnica C: Ensemble Learning por votação suave (soft voting).

Três classificadores de famílias diferentes olham os mesmos embeddings:
  1. Regressão Logística: modelo linear, enxerga fronteiras simples;
  2. Random Forest: bagging de árvores, reduz variância;
  3. Histogram Gradient Boosting: boosting de árvores, reduz viés.

Cada um erra de um jeito diferente. Na votação suave, a probabilidade final
é a MÉDIA das probabilidades dos três. Quando os erros não são correlacionados,
a média cancela parte deles (é o mesmo motivo pelo qual a média de várias
medições é mais precisa que uma medição só).

Votação suave em vez de dura: a dura conta votos (2 contra 1); a suave usa a
confiança de cada modelo, então um modelo muito seguro pesa mais que dois em
dúvida. Ela também fornece probabilidade, que precisamos para calcular a AUC.

Os três modelos recebem `class_weight` (ver config.PESO_CLASSE) porque o
treino é desbalanceado e a decisão sai de um limiar fixo de 0,5. Ver a nota
em config.py para os números que motivaram a escolha.
"""

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config


def criar_ensemble(seed: int) -> VotingClassifier:
    # StandardScaler só na Regressão Logística: ela é sensível à escala das
    # variáveis (o gradiente da otimização depende disso). Árvores não são,
    # porque só comparam cada variável com um limiar.
    logistica = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=config.MAX_ITER_LR,
            random_state=seed,
            class_weight=config.PESO_CLASSE,
        ),
    )
    floresta = RandomForestClassifier(
        n_estimators=config.N_ARVORES_RF,
        n_jobs=-1,                # usa todos os núcleos da CPU
        random_state=seed,
        class_weight=config.PESO_CLASSE,
    )
    boosting = HistGradientBoostingClassifier(
        random_state=seed,
        class_weight=config.PESO_CLASSE,
    )

    return VotingClassifier(
        estimators=[("logistica", logistica), ("floresta", floresta), ("boosting", boosting)],
        voting="soft",
    )
