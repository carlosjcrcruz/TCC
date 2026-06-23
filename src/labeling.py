"""Geração dos rótulos manter, compra e venda."""

from __future__ import annotations

import numpy as np
import pandas as pd


NOMES_CLASSES = {0: "manter", 1: "compra", 2: "venda"}


def criar_rotulos(
    close_referencia: pd.Series,
    horizonte: int = 10,
    limiar: float = 0.002,
) -> pd.DataFrame:
    """Calcula retorno futuro e classe usando a série original como referência."""
    if horizonte <= 0:
        raise ValueError("O horizonte futuro deve ser positivo.")
    if limiar < 0:
        raise ValueError("O limiar de retorno não pode ser negativo.")

    close = pd.to_numeric(close_referencia, errors="coerce")
    futuro = close.shift(-horizonte)
    retorno_futuro = (futuro - close) / close
    rotulo = np.select(
        [retorno_futuro >= limiar, retorno_futuro <= -limiar],
        [1, 2],
        default=0,
    ).astype(float)
    rotulo[futuro.isna().to_numpy()] = np.nan
    return pd.DataFrame(
        {"retorno_futuro": retorno_futuro, "rotulo": rotulo},
        index=close.index,
    )


def anexar_rotulos(
    dados: pd.DataFrame,
    close_referencia: pd.Series,
    horizonte: int,
    limiar: float,
) -> pd.DataFrame:
    """Anexa rótulos comparáveis, sempre calculados no close original."""
    saida = dados.copy()
    rotulos = criar_rotulos(close_referencia, horizonte, limiar)
    saida[["retorno_futuro", "rotulo"]] = rotulos
    return saida

