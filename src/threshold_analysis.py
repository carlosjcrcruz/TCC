"""Diagnóstico do limiar theta usado para gerar os rótulos direcionais."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def analisar_limiar_retorno(
    close: pd.Series,
    serie_ruidosa: pd.Series,
    horizonte: int,
    proporcao_treino: float,
    theta_adotado: float | None = None,
    candidatos: tuple[float, ...] = (
        0.002,
        0.003,
        0.004,
        0.005,
        0.006,
        0.0075,
        0.008,
        0.010,
    ),
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compara candidatos usando somente o trecho temporal de treinamento.

    Para ruído multiplicativo independente nos preços inicial e final, o erro
    do retorno tem desvio aproximado de ``sqrt(2) * sigma``. O limite de 95%
    usa 1,96 desvios. A volatilidade robusta é estimada por 1,4826 * MAD.
    """
    fim_treino = int(len(close) * proporcao_treino)
    original_treino = pd.to_numeric(close.iloc[:fim_treino], errors="coerce")
    ruidosa_treino = pd.to_numeric(serie_ruidosa.iloc[:fim_treino], errors="coerce")

    ruido_relativo = ruidosa_treino.to_numpy() / original_treino.to_numpy() - 1.0
    sigma_ruido = float(np.std(ruido_relativo, ddof=1))
    limite_ruido_95 = float(1.96 * math.sqrt(2.0) * sigma_ruido)

    retornos = (original_treino.shift(-horizonte) / original_treino - 1.0).dropna()
    mediana = float(retornos.median())
    mad = float(np.median(np.abs(retornos.to_numpy() - mediana)))
    volatilidade_robusta = 1.4826 * mad
    volatilidade_padrao = float(retornos.std(ddof=1))

    linhas = []
    for theta in candidatos:
        compra = int((retornos >= theta).sum())
        venda = int((retornos <= -theta).sum())
        manter = int(len(retornos) - compra - venda)
        linhas.append(
            {
                "theta": theta,
                "theta_percentual": 100.0 * theta,
                "quantidade_manter": manter,
                "quantidade_compra": compra,
                "quantidade_venda": venda,
                "proporcao_manter": manter / len(retornos),
                "proporcao_compra": compra / len(retornos),
                "proporcao_venda": venda / len(retornos),
                "proporcao_direcional": (compra + venda) / len(retornos),
                "supera_limite_ruido_95": bool(theta >= limite_ruido_95),
                "supera_volatilidade_robusta": bool(theta >= volatilidade_robusta),
                "supera_volatilidade_padrao": bool(theta >= volatilidade_padrao),
            }
        )

    theta_minimo = max(limite_ruido_95, volatilidade_padrao)
    if theta_adotado is None:
        theta_adotado = math.ceil(theta_minimo * 1000.0) / 1000.0
    resumo = {
        "horizonte_candles": int(horizonte),
        "quantidade_treino": int(len(retornos)),
        "sigma_ruido_relativo_medido": sigma_ruido,
        "sigma_erro_retorno_aproximado": math.sqrt(2.0) * sigma_ruido,
        "limite_ruido_95": limite_ruido_95,
        "volatilidade_robusta_retorno_horizonte": volatilidade_robusta,
        "volatilidade_padrao_retorno_horizonte": volatilidade_padrao,
        "theta_minimo_conservador": theta_minimo,
        "theta_adotado": float(theta_adotado),
    }
    return pd.DataFrame(linhas), resumo
