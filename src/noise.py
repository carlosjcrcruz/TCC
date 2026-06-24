"""Construção dos cenários original, ruidosos, EMA e Wavelet."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd


def adicionar_ruido_gaussiano(
    close: pd.Series, intensidade: float, seed: int = 42
) -> pd.Series:
    """Aplica close_ruidoso = close * (1 + ruído) de modo reprodutível."""
    if intensidade < 0:
        raise ValueError("A intensidade do ruído não pode ser negativa.")
    gerador = np.random.default_rng(seed)
    ruido = gerador.normal(loc=0.0, scale=intensidade, size=len(close))
    return pd.Series(close.to_numpy() * (1.0 + ruido), index=close.index)


def suavizar_ema(close: pd.Series, span: int) -> pd.Series:
    """Suaviza a série por média móvel exponencial causal."""
    if span <= 0:
        raise ValueError("O span da EMA deve ser positivo.")
    return close.ewm(span=span, adjust=False).mean()


def wavelet_denoising(
    close: pd.Series,
    wavelet: str = "db4",
    nivel: int = 3,
    janela_causal: int = 256,
    threshold_scale: float = 1.0,
    threshold_mode: str = "soft",
) -> pd.Series:
    """Remove ruído em janelas causais, usando apenas o presente e o passado.

    Para cada candle, a transformação é refeita na janela que termina naquele
    candle. Tomar apenas o último valor reconstruído evita que preços futuros
    vazem para os atributos avaliados no teste.
    """
    try:
        import pywt
    except ImportError as exc:
        raise ImportError("Instale PyWavelets para executar o cenário Wavelet.") from exc

    if janela_causal <= 1:
        raise ValueError("A janela causal da Wavelet deve ser maior que 1.")
    valores = close.to_numpy(dtype=float)
    objeto_wavelet = pywt.Wavelet(wavelet)
    saida = valores.copy()
    for fim in range(len(valores)):
        inicio = max(0, fim - janela_causal + 1)
        bloco = valores[inicio : fim + 1]
        nivel_maximo = pywt.dwt_max_level(len(bloco), objeto_wavelet.dec_len)
        nivel_usado = min(nivel, nivel_maximo)
        if nivel_usado < 1:
            continue

        coeficientes = pywt.wavedec(bloco, objeto_wavelet, level=nivel_usado)
        detalhe_fino = coeficientes[-1]
        sigma = (
            np.median(np.abs(detalhe_fino)) / 0.6745 if len(detalhe_fino) else 0.0
        )
        limiar = threshold_scale * sigma * np.sqrt(2.0 * np.log(len(bloco)))
        tratados = [coeficientes[0]] + [
            pywt.threshold(coef, limiar, mode=threshold_mode)
            for coef in coeficientes[1:]
        ]
        reconstruida = pywt.waverec(tratados, objeto_wavelet)[: len(bloco)]
        saida[fim] = reconstruida[-1]
    return pd.Series(saida, index=close.index)


def criar_cenarios(
    dados: pd.DataFrame,
    config,
    series_filtradas: dict[str, pd.Series] | None = None,
):
    """Devolve os sete cenários em uma ordem estável para comparação."""
    cenarios = OrderedDict()

    original = dados.copy()
    original["close_modelo"] = original["close"]
    cenarios["Original"] = original

    # A mesma sequência aleatória em todas as intensidades isola o efeito da escala.
    for intensidade in config.intensidades_ruido:
        cenario = dados.copy()
        cenario["close_modelo"] = adicionar_ruido_gaussiano(
            dados["close"], intensidade, config.seed
        )
        nome = f"Ruído {intensidade:g}"
        cenarios[nome] = cenario

    nome_ruido_filtro = f"Ruído {config.intensidade_ruido_filtros:g}"
    entrada_filtros = cenarios[nome_ruido_filtro]["close_modelo"]

    ema = dados.copy()
    ema["close_modelo"] = (
        series_filtradas["EMA"]
        if series_filtradas is not None
        else suavizar_ema(entrada_filtros, config.ema_span_cenario)
    )
    cenarios["EMA"] = ema

    wavelet = dados.copy()
    wavelet["close_modelo"] = (
        series_filtradas["Wavelet"]
        if series_filtradas is not None
        else wavelet_denoising(
            entrada_filtros,
            wavelet=config.wavelet,
            nivel=config.wavelet_nivel,
            janela_causal=config.wavelet_janela_causal,
            threshold_scale=config.wavelet_threshold_scale,
            threshold_mode=config.wavelet_threshold_mode,
        )
    )
    cenarios["Wavelet"] = wavelet
    return cenarios
