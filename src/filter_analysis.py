"""Diagnóstico quantitativo e seleção reproduzível dos filtros de ruído."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from src.noise import adicionar_ruido_gaussiano, suavizar_ema, wavelet_denoising


@dataclass
class ResultadoSelecaoFiltros:
    """Artefatos necessários para documentar e reutilizar a seleção."""

    serie_ruidosa: pd.Series
    series_filtradas: dict[str, pd.Series]
    series_originais_filtradas: dict[str, pd.Series]
    melhores_parametros: dict[str, dict]
    tabela_busca: pd.DataFrame


def _rmse(valores: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(valores))))


def _correlacao(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def calcular_metricas_filtragem(
    original: pd.Series,
    ruidosa: pd.Series,
    filtrada: pd.Series,
    original_filtrada: pd.Series,
    inicio: int = 0,
    fim: int | None = None,
) -> dict[str, float | bool]:
    """Mede remoção do ruído conhecido e deformação do sinal original.

    ``ruido_injetado`` é observável porque o experimento constrói a série
    ruidosa. ``ruido_removido`` é a diferença entre a entrada ruidosa e a saída
    do filtro. Filtrar também a série original permite estimar separadamente a
    deformação que o método causaria mesmo sem o ruído artificial.
    """
    fatia = slice(inicio, fim)
    limpo = original.to_numpy(dtype=float)[fatia]
    com_ruido = ruidosa.to_numpy(dtype=float)[fatia]
    tratado = filtrada.to_numpy(dtype=float)[fatia]
    limpo_tratado = original_filtrada.to_numpy(dtype=float)[fatia]

    ruido_injetado = com_ruido - limpo
    ruido_removido = com_ruido - tratado
    erro_total = tratado - limpo
    distorcao_sinal = limpo_tratado - limpo
    ruido_pos_filtro = tratado - limpo_tratado

    rmse_ruido = _rmse(ruido_injetado)
    rmse_filtrada = _rmse(erro_total)
    rmse_distorcao = _rmse(distorcao_sinal)
    rmse_ruido_pos = _rmse(ruido_pos_filtro)
    eps = np.finfo(float).eps

    return {
        "rmse_serie_ruidosa": rmse_ruido,
        "rmse_filtrada_vs_original": rmse_filtrada,
        "reducao_rmse_percentual": 100.0 * (1.0 - rmse_filtrada / max(rmse_ruido, eps)),
        "correlacao_ruido_removido_injetado": _correlacao(
            ruido_removido, ruido_injetado
        ),
        "rmse_distorcao_sinal_original": rmse_distorcao,
        "rmse_ruido_apos_filtro": rmse_ruido_pos,
        "atenuacao_ruido_db": 20.0
        * np.log10(max(rmse_ruido, eps) / max(rmse_ruido_pos, eps)),
        # Menor é melhor. Valor < 1 significa que o filtro aproximou a série
        # da referência original; valor > 1 significa que piorou o sinal.
        "razao_rmse_objetivo": rmse_filtrada / max(rmse_ruido, eps),
        "melhorou_sobre_serie_ruidosa": bool(rmse_filtrada < rmse_ruido),
    }


def calcular_estatisticas_ruido(
    original: pd.Series, ruidosa: pd.Series, intensidade: float
) -> dict[str, float]:
    """Caracteriza o ruído multiplicativo relativo usado no experimento."""
    relativo = ruidosa.to_numpy(dtype=float) / original.to_numpy(dtype=float) - 1.0
    serie = pd.Series(relativo)
    n = len(serie)
    assimetria = float(serie.skew())
    curtose_excesso = float(serie.kurt())
    jb = n * (assimetria**2 / 6.0 + curtose_excesso**2 / 24.0)
    # Para 2 graus de liberdade, a sobrevivência qui-quadrado é exp(-x/2).
    return {
        "intensidade_configurada": float(intensidade),
        "quantidade": int(n),
        "media_relativa": float(serie.mean()),
        "desvio_padrao_relativo": float(serie.std(ddof=1)),
        "assimetria": assimetria,
        "curtose_excesso": curtose_excesso,
        "autocorrelacao_lag1": float(serie.autocorr(lag=1)),
        "jarque_bera": float(jb),
        "jarque_bera_p_valor_aproximado": float(np.exp(-jb / 2.0)),
    }


def _candidatos_ema(config) -> list[dict]:
    spans = config.ema_spans_busca if config.executar_busca_filtros else (config.ema_span_cenario,)
    return [{"span": int(span)} for span in spans]


def _candidatos_wavelet(config) -> list[dict]:
    if not config.executar_busca_filtros:
        return [
            {
                "wavelet": config.wavelet,
                "nivel": config.wavelet_nivel,
                "janela_causal": config.wavelet_janela_causal,
                "threshold_scale": config.wavelet_threshold_scale,
                "threshold_mode": config.wavelet_threshold_mode,
            }
        ]
    return [
        {
            "wavelet": wavelet,
            "nivel": int(nivel),
            "janela_causal": int(janela),
            "threshold_scale": float(escala),
            "threshold_mode": modo,
        }
        for wavelet, nivel, janela, escala, modo in product(
            config.wavelets_busca,
            config.wavelet_niveis_busca,
            config.wavelet_janelas_busca,
            config.wavelet_threshold_scales_busca,
            config.wavelet_threshold_modes_busca,
        )
    ]


def selecionar_filtros(original: pd.Series, config) -> ResultadoSelecaoFiltros:
    """Seleciona EMA e Wavelet no trecho de treino e devolve séries completas."""
    ruidosa = adicionar_ruido_gaussiano(
        original, config.intensidade_ruido_filtros, config.seed
    )
    fim_treino = max(2, int(len(original) * config.proporcao_treino))
    registros: list[dict] = []
    candidatos_calculados: dict[str, list[tuple[dict, pd.Series, pd.Series, dict]]] = {
        "EMA": [],
        "Wavelet": [],
    }

    for parametros in _candidatos_ema(config):
        filtrada = suavizar_ema(ruidosa, **parametros)
        original_filtrada = suavizar_ema(original, **parametros)
        metricas = calcular_metricas_filtragem(
            original, ruidosa, filtrada, original_filtrada, fim=fim_treino
        )
        candidatos_calculados["EMA"].append(
            (parametros, filtrada, original_filtrada, metricas)
        )
        registros.append({"metodo": "EMA", "particao_selecao": "treino", **parametros, **metricas})

    for parametros in _candidatos_wavelet(config):
        filtrada = wavelet_denoising(ruidosa, **parametros)
        original_filtrada = wavelet_denoising(original, **parametros)
        metricas = calcular_metricas_filtragem(
            original, ruidosa, filtrada, original_filtrada, fim=fim_treino
        )
        candidatos_calculados["Wavelet"].append(
            (parametros, filtrada, original_filtrada, metricas)
        )
        registros.append(
            {"metodo": "Wavelet", "particao_selecao": "treino", **parametros, **metricas}
        )

    series_filtradas: dict[str, pd.Series] = {}
    originais_filtradas: dict[str, pd.Series] = {}
    melhores_parametros: dict[str, dict] = {}
    for metodo, candidatos in candidatos_calculados.items():
        melhor = min(candidatos, key=lambda item: item[3]["razao_rmse_objetivo"])
        parametros, filtrada, original_filtrada, _ = melhor
        melhores_parametros[metodo] = parametros
        series_filtradas[metodo] = filtrada
        originais_filtradas[metodo] = original_filtrada

    tabela = pd.DataFrame(registros).sort_values(
        ["metodo", "razao_rmse_objetivo"], ignore_index=True
    )
    tabela["selecionado"] = False
    for metodo in melhores_parametros:
        indice = tabela.index[tabela["metodo"] == metodo][0]
        tabela.loc[indice, "selecionado"] = True

    return ResultadoSelecaoFiltros(
        serie_ruidosa=ruidosa,
        series_filtradas=series_filtradas,
        series_originais_filtradas=originais_filtradas,
        melhores_parametros=melhores_parametros,
        tabela_busca=tabela,
    )


def diagnosticar_filtros_selecionados(
    original: pd.Series, resultado: ResultadoSelecaoFiltros, config
) -> pd.DataFrame:
    """Avalia os filtros escolhidos separadamente em treino, validação e teste."""
    n = len(original)
    fim_treino = int(n * config.proporcao_treino)
    fim_validacao = int(n * (config.proporcao_treino + config.proporcao_validacao))
    particoes = {
        "treino": (0, fim_treino),
        "validacao": (fim_treino, fim_validacao),
        "teste": (fim_validacao, n),
        "completo": (0, n),
    }
    linhas = []
    for metodo, filtrada in resultado.series_filtradas.items():
        for nome_particao, (inicio, fim) in particoes.items():
            metricas = calcular_metricas_filtragem(
                original,
                resultado.serie_ruidosa,
                filtrada,
                resultado.series_originais_filtradas[metodo],
                inicio,
                fim,
            )
            linhas.append(
                {
                    "metodo": metodo,
                    "particao": nome_particao,
                    **resultado.melhores_parametros[metodo],
                    **metricas,
                }
            )
    return pd.DataFrame(linhas)
