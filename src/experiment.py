"""Orquestração ponta a ponta dos sete experimentos."""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_collection import obter_dados, salvar_dados_originais
from src.evaluation import avaliar_classificacao
from src.labeling import anexar_rotulos
from src.model import (
    configurar_reprodutibilidade,
    criar_modelo,
    limpar_sessao,
    treinar_modelo,
)
from src.noise import criar_cenarios
from src.plots import (
    plotar_comparacao_f1,
    plotar_comparacao_fpr,
    plotar_comparacao_serie,
    plotar_matriz_confusao,
    plotar_original_vs_ruidos,
)
from src.preprocessing import criar_features, normalizar_sem_vazamento, preparar_dados
from src.sequences import (
    calcular_limites_temporais,
    criar_sequencias,
    separar_sequencias_temporais,
)


COLUNAS_COMPARACAO = [
    "cenario",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "FPR_compra",
    "FPR_venda",
    "FDR_compra",
    "FDR_venda",
    "quantidade_sinais_compra",
    "quantidade_sinais_venda",
]


def _slug(texto: str) -> str:
    """Cria nomes de arquivos portáveis, sem acentos ou espaços."""
    normalizado = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return "_".join(normalizado.lower().replace(".", "_").split())


def _serializavel(valor):
    if isinstance(valor, dict):
        return {str(chave): _serializavel(item) for chave, item in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_serializavel(item) for item in valor]
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if isinstance(valor, (np.integer, np.floating)):
        return valor.item()
    return valor


def _preparar_pastas(config) -> dict[str, Path]:
    pastas = {
        "resultados": config.pasta_resultados,
        "dados": config.pasta_dados_processados,
        "graficos": config.pasta_resultados / "graficos",
        "modelos": config.pasta_resultados / "modelos",
        "metricas": config.pasta_resultados / "metricas",
        "predicoes": config.pasta_resultados / "predicoes",
        "historicos": config.pasta_resultados / "historicos_treinamento",
    }
    for pasta in pastas.values():
        pasta.mkdir(parents=True, exist_ok=True)
    return pastas


def _criar_dataset_cenario(nome, dados_cenario, close_referencia, config):
    """Calcula features e usa a mesma verdade original em todos os cenários."""
    dados = criar_features(dados_cenario, config)
    dados = anexar_rotulos(
        dados,
        close_referencia=close_referencia,
        horizonte=config.horizonte_futuro,
        limiar=config.limiar_retorno,
    )
    colunas_obrigatorias = list(config.features) + ["rotulo", "retorno_futuro"]
    dados = dados.dropna(subset=colunas_obrigatorias).reset_index(drop=True)
    dados["rotulo"] = dados["rotulo"].astype(int)
    minimo = config.tamanho_janela + 10
    if len(dados) < minimo:
        raise ValueError(
            f"O cenário {nome!r} tem só {len(dados)} linhas válidas; "
            f"são necessárias pelo menos {minimo}."
        )
    return dados


def executar_experimentos(config) -> pd.DataFrame:
    """Executa coleta, sete treinamentos, avaliação, CSVs e gráficos."""
    config.validar()
    pastas = _preparar_pastas(config)

    logging.info("Obtendo dados históricos de %s (%s)...", config.simbolo, config.timeframe)
    dados_originais = preparar_dados(obter_dados(config))
    caminho_original = salvar_dados_originais(dados_originais, pastas["dados"])
    logging.info("Dados originais salvos em %s", caminho_original)

    logging.info("Criando cenários experimentais...")
    cenarios = criar_cenarios(dados_originais, config)
    plotar_original_vs_ruidos(
        dados_originais, cenarios, pastas["graficos"] / "original_vs_ruidos.png"
    )
    plotar_comparacao_serie(
        dados_originais,
        cenarios["EMA"],
        "EMA",
        pastas["graficos"] / "original_vs_ema.png",
    )
    plotar_comparacao_serie(
        dados_originais,
        cenarios["Wavelet"],
        "Wavelet",
        pastas["graficos"] / "original_vs_wavelet.png",
    )

    resultados = []
    for numero, (nome, dados_cenario) in enumerate(cenarios.items(), start=1):
        logging.info("[%d/%d] Processando cenário: %s", numero, len(cenarios), nome)
        slug = _slug(nome)
        dados = _criar_dataset_cenario(
            nome, dados_cenario, dados_originais["close"], config
        )
        limites = calcular_limites_temporais(
            len(dados), config.proporcao_treino, config.proporcao_validacao
        )

        # O scaler vê somente as linhas do treino. Validação e teste apenas usam transform().
        normalizados, scaler = normalizar_sem_vazamento(
            dados,
            config.features,
            limites.fim_treino,
            config.tipo_scaler,
        )
        joblib.dump(scaler, pastas["modelos"] / f"scaler_{slug}.joblib")

        # O CSV processado preserva valores originais e acrescenta versões normalizadas.
        para_salvar = dados.copy()
        for feature in config.features:
            para_salvar[f"{feature}_normalizado"] = normalizados[feature]
        para_salvar["particao"] = np.select(
            [
                para_salvar.index < limites.fim_treino,
                para_salvar.index < limites.fim_validacao,
            ],
            ["treino", "validacao"],
            default="teste",
        )
        para_salvar.to_csv(pastas["dados"] / f"{slug}.csv", index=False)

        x, y, indices = criar_sequencias(
            normalizados,
            config.features,
            tamanho_janela=config.tamanho_janela,
        )
        conjuntos = separar_sequencias_temporais(
            x,
            y,
            indices,
            limites,
            horizonte_rotulo=config.horizonte_futuro,
        )
        x_treino, y_treino, _ = conjuntos["treino"]
        x_validacao, y_validacao, _ = conjuntos["validacao"]
        x_teste, y_teste, indices_teste = conjuntos["teste"]

        limpar_sessao()
        configurar_reprodutibilidade(config.seed)
        modelo = criar_modelo(x_treino.shape[1:], config)
        historico = treinar_modelo(
            modelo,
            x_treino,
            y_treino,
            x_validacao,
            y_validacao,
            config,
        )
        pd.DataFrame(historico.history).to_csv(
            pastas["historicos"] / f"historico_{slug}.csv", index=False
        )
        modelo.save(pastas["modelos"] / f"modelo_{slug}.keras")

        # Inferência direta evita recompilar uma função predict a cada novo
        # modelo do laço de cenários e mantém exatamente as mesmas probabilidades.
        probabilidades = modelo(x_teste, training=False).numpy()
        y_predito = np.argmax(probabilidades, axis=1)
        metricas, matriz = avaliar_classificacao(y_teste, y_predito)
        metricas_completas = {
            "cenario": nome,
            **metricas,
            "matriz_confusao": matriz,
            "quantidade_amostras_treino": len(x_treino),
            "quantidade_amostras_validacao": len(x_validacao),
            "quantidade_amostras_teste": len(x_teste),
        }
        with (pastas["metricas"] / f"metricas_{slug}.json").open(
            "w", encoding="utf-8"
        ) as arquivo:
            json.dump(_serializavel(metricas_completas), arquivo, indent=2, ensure_ascii=False)

        predicoes = pd.DataFrame(
            {
                "time": dados.iloc[indices_teste]["time"].to_numpy(),
                "classe_real": y_teste,
                "classe_predita": y_predito,
                "prob_manter": probabilidades[:, 0],
                "prob_compra": probabilidades[:, 1],
                "prob_venda": probabilidades[:, 2],
            }
        )
        predicoes.to_csv(pastas["predicoes"] / f"predicoes_{slug}.csv", index=False)
        plotar_matriz_confusao(
            matriz,
            nome,
            pastas["graficos"] / f"matriz_confusao_{slug}.png",
        )
        resultados.append({"cenario": nome, **metricas})

    tabela = pd.DataFrame(resultados)
    tabela.loc[:, COLUNAS_COMPARACAO].to_csv(
        pastas["resultados"] / "comparacao_cenarios.csv", index=False
    )
    # Uma versão detalhada inclui também o FPR geral macro.
    tabela.drop(columns=["metricas_por_classe"]).to_csv(
        pastas["resultados"] / "comparacao_cenarios_detalhada.csv", index=False
    )
    plotar_comparacao_fpr(tabela, pastas["graficos"] / "comparacao_fpr.png")
    plotar_comparacao_f1(tabela, pastas["graficos"] / "comparacao_f1.png")
    logging.info("Experimentos concluídos. Resultados em %s", pastas["resultados"])
    return tabela

