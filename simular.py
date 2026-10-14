"""
Executa as 30 simulações do estudo comparativo.

Cenário H0:  VAE  + Transfer Learning (ResNet18) + Ensemble
Cenário H1:  GAN  + Transfer Learning (ResNet18) + Ensemble   <- contrafactual
Referência:  sem dados sintéticos     + ResNet18 + Ensemble   (não entra no teste;
             serve para mostrar se gerar dados ajuda de fato)

Uso:
    python simular.py                       # 30 simulações completas
    python simular.py --simulacoes 2 --epocas 5   # teste rápido

Cada simulação terminada é gravada no CSV na mesma hora. Se o programa for
interrompido, basta rodar de novo: as seeds já concluídas são puladas.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

import config
from src.dados import carregar_split, sortear_indices_escassez
from src.ensemble import criar_ensemble
from src.extrator import ExtratorResNet
from src.gan import gerar_gan, treinar_gan
from src.metricas import avaliar, distancia_frechet
from src.utilidades import escolher_dispositivo, fixar_sementes
from src.vae import gerar_vae, treinar_vae

# Os dois geradores expõem a mesma interface (treinar, gerar). Por isso o
# laço principal trata os dois cenários com o mesmo código: a única linha que
# muda entre H0 e H1 é qual par de funções é chamado.
GERADORES = {
    "VAE": (treinar_vae, gerar_vae),
    "GAN": (treinar_gan, gerar_gan),
}


def preparar_cache(extrator: ExtratorResNet):
    """
    Carrega os dados e calcula os embeddings das imagens REAIS uma única vez.

    A ResNet está congelada, então o embedding de uma imagem real é sempre o
    mesmo. Recalcular em cada simulação seria desperdício.
    """
    imagens_treino, rotulos_treino = carregar_split("train")
    imagens_teste, rotulos_teste = carregar_split("test")

    arquivo_cache = config.PASTA_DADOS / f"embeddings_resnet18_{config.TAMANHO_ENTRADA_RESNET}.npz"
    if arquivo_cache.exists():
        cache = np.load(arquivo_cache)
        emb_treino, emb_teste = cache["treino"], cache["teste"]
        print("Embeddings reais carregados do cache.")
    else:
        print("Calculando embeddings reais com a ResNet18 (só na primeira execução)...")
        emb_treino = extrator.extrair(imagens_treino)
        emb_teste = extrator.extrair(imagens_teste)
        np.savez_compressed(arquivo_cache, treino=emb_treino, teste=emb_teste)

    return {
        "imagens_treino": imagens_treino, "rotulos_treino": rotulos_treino,
        "emb_treino": emb_treino, "emb_teste": emb_teste, "rotulos_teste": rotulos_teste,
    }


def treinar_e_avaliar_ensemble(x_treino, y_treino, cache, seed):
    modelo = criar_ensemble(seed)
    modelo.fit(x_treino, y_treino)
    return avaliar(modelo, cache["emb_teste"], cache["rotulos_teste"])


def executar_simulacao(indice: int, seed: int, cache: dict, extrator, dispositivo, epocas: int):
    linhas = []
    idx_min, idx_maj = sortear_indices_escassez(cache["rotulos_treino"], seed)
    imagens_minoria = cache["imagens_treino"][idx_min]

    # Embeddings reais desta simulação (vêm do cache)
    emb_reais = np.concatenate([cache["emb_treino"][idx_min], cache["emb_treino"][idx_maj]])
    y_reais = np.concatenate([np.full(len(idx_min), config.CLASSE_MINORIA),
                              np.full(len(idx_maj), config.CLASSE_MAIORIA)])

    # Referência: só dados reais, desbalanceados
    fixar_sementes(seed)
    metricas = treinar_e_avaliar_ensemble(emb_reais, y_reais, cache, seed)
    linhas.append({"simulacao": indice, "seed": seed, "cenario": "SEM_AUMENTO",
                   "epocas": epocas, "fd_resnet": np.nan, "tempo_gerador_s": 0.0, **metricas})

    amostras = {"reais": imagens_minoria[:16]}
    historicos = {}
    for nome, (treinar, gerar) in GERADORES.items():
        # Mesma seed antes de CADA gerador: os dois começam do mesmo estado
        # aleatório, o que mantém o pareamento justo.
        fixar_sementes(seed)
        inicio = time.perf_counter()
        modelo, historico = treinar(imagens_minoria, dispositivo, epocas=epocas)
        sinteticas = gerar(modelo, config.N_SINTETICAS, dispositivo)
        tempo = time.perf_counter() - inicio

        emb_sinteticas = extrator.extrair(sinteticas)
        # Referência da distância de Fréchet: SÓ as normais reais que este
        # gerador viu nesta simulação. Usar todas as 1214 normais do split de
        # treino compararia o gerador com imagens que ele nunca recebeu, o que
        # mede generalização e não fidelidade, além de tornar a referência
        # idêntica em todas as 30 simulações e quebrar o pareamento.
        fd = distancia_frechet(cache["emb_treino"][idx_min], emb_sinteticas)

        # Treino balanceado: reais + sintéticas (todas rotuladas como normal)
        x_treino = np.concatenate([emb_reais, emb_sinteticas])
        y_treino = np.concatenate([y_reais, np.full(len(emb_sinteticas), config.CLASSE_MINORIA)])
        metricas = treinar_e_avaliar_ensemble(x_treino, y_treino, cache, seed)

        linhas.append({"simulacao": indice, "seed": seed, "cenario": nome,
                       "epocas": epocas, "fd_resnet": fd, "tempo_gerador_s": tempo, **metricas})
        amostras[nome.lower()] = sinteticas[:16]
        historicos[nome] = historico

    return linhas, amostras, historicos


def main():
    parser = argparse.ArgumentParser(description="Simulações VAE vs GAN")
    parser.add_argument("--simulacoes", type=int, default=config.N_SIMULACOES)
    parser.add_argument("--epocas", type=int, default=config.EPOCAS_GERADOR)
    parser.add_argument("--reiniciar", action="store_true", help="apaga resultados anteriores")
    args = parser.parse_args()

    config.PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    if args.reiniciar and config.ARQUIVO_CSV.exists():
        config.ARQUIVO_CSV.unlink()

    dispositivo = escolher_dispositivo()
    print(f"Dispositivo: {dispositivo}")
    extrator = ExtratorResNet(dispositivo)
    cache = preparar_cache(extrator)

    # Retomada: só contam como concluídas as seeds já gravadas COM O MESMO
    # número de épocas. Sem essa checagem, um teste rápido (--epocas 5) deixaria
    # linhas no CSV que a execução oficial (200 épocas) pularia, misturando dois
    # orçamentos de treino na mesma análise sem ninguém perceber.
    concluidas = set()
    if config.ARQUIVO_CSV.exists():
        anteriores = pd.read_csv(config.ARQUIVO_CSV)
        if "epocas" not in anteriores.columns:
            raise SystemExit(
                f"{config.ARQUIVO_CSV} foi gerado por uma versão antiga, sem a coluna "
                "'epocas'. Rode com --reiniciar para refazer as simulações."
            )
        mesmo_orcamento = anteriores[anteriores["epocas"] == args.epocas]
        concluidas = set(mesmo_orcamento["seed"])
        outras = set(anteriores["seed"]) - concluidas
        if outras:
            raise SystemExit(
                f"{config.ARQUIVO_CSV} já tem {len(outras)} seed(s) treinadas com outro "
                f"número de épocas. Misturar orçamentos invalida a comparação.\n"
                "Rode com --reiniciar para começar do zero."
            )

    for indice in range(args.simulacoes):
        seed = config.SEED_BASE + indice
        if seed in concluidas:
            continue
        inicio = time.perf_counter()
        linhas, amostras, historicos = executar_simulacao(indice, seed, cache, extrator, dispositivo, args.epocas)

        # Grava esta simulação imediatamente (modo append)
        pd.DataFrame(linhas).to_csv(config.ARQUIVO_CSV, mode="a", index=False,
                                    header=not config.ARQUIVO_CSV.exists())

        # A primeira simulação guarda amostras e curvas de perda para as figuras
        if indice == 0:
            np.savez_compressed(config.PASTA_RESULTADOS / "amostras_simulacao_0.npz", **amostras)
            with open(config.PASTA_RESULTADOS / "historico_simulacao_0.json", "w") as arquivo:
                json.dump(historicos, arquivo)

        resumo = "  ".join(f"{l['cenario']}: F1={l['f1_minoria']:.3f}" for l in linhas)
        print(f"[{indice + 1:02d}/{args.simulacoes}] seed {seed}  {resumo}  ({time.perf_counter() - inicio:.0f}s)")

    print(f"\nResultados em {config.ARQUIVO_CSV}")
    print("Próximo passo: python analisar.py")


if __name__ == "__main__":
    main()
