"""Limpeza, indicadores técnicos e normalização sem vazamento de dados."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.data_collection import COLUNAS_OBRIGATORIAS, validar_dados


def preparar_dados(dados: pd.DataFrame) -> pd.DataFrame:
    """Ordena, converte tipos, remove duplicatas e valores inválidos."""
    dados = validar_dados(dados)
    dados["time"] = pd.to_datetime(dados["time"], utc=True, errors="coerce")
    for coluna in COLUNAS_OBRIGATORIAS[1:]:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")

    dados = dados.replace([np.inf, -np.inf], np.nan)
    dados = dados.dropna(subset=COLUNAS_OBRIGATORIAS)
    dados = dados.drop_duplicates(subset="time", keep="last")
    dados = dados.sort_values("time").reset_index(drop=True)
    if dados.empty:
        raise ValueError("Não restaram linhas válidas após o pré-processamento.")
    return dados


def calcular_rsi(close: pd.Series, periodo: int = 14) -> pd.Series:
    """Calcula RSI com médias exponenciais no estilo de Wilder."""
    delta = close.diff()
    ganhos = delta.clip(lower=0.0)
    perdas = -delta.clip(upper=0.0)
    media_ganhos = ganhos.ewm(alpha=1 / periodo, min_periods=periodo, adjust=False).mean()
    media_perdas = perdas.ewm(alpha=1 / periodo, min_periods=periodo, adjust=False).mean()
    rs = media_ganhos / media_perdas.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.mask((media_perdas == 0) & (media_ganhos > 0), 100.0)
    rsi = rsi.mask((media_perdas == 0) & (media_ganhos == 0), 50.0)
    return rsi


def criar_features(dados_cenario: pd.DataFrame, config) -> pd.DataFrame:
    """Cria atributos exclusivamente a partir do presente e do passado."""
    dados = dados_cenario.copy()
    if "close_modelo" not in dados:
        dados["close_modelo"] = dados["close"]

    close = pd.to_numeric(dados["close_modelo"], errors="coerce")
    dados["retorno_percentual"] = close.pct_change(fill_method=None)
    dados["sma"] = close.rolling(config.periodo_sma).mean()
    dados["ema"] = close.ewm(span=config.periodo_ema_feature, adjust=False).mean()
    dados["volatilidade"] = dados["retorno_percentual"].rolling(
        config.periodo_volatilidade
    ).std()
    dados["rsi"] = calcular_rsi(close, config.periodo_rsi)
    return dados.replace([np.inf, -np.inf], np.nan)


def criar_scaler(tipo: str):
    """Instancia o normalizador definido na configuração."""
    if tipo.lower() == "standard":
        return StandardScaler()
    if tipo.lower() == "minmax":
        return MinMaxScaler()
    raise ValueError("Scaler desconhecido. Use 'standard' ou 'minmax'.")


def normalizar_sem_vazamento(
    dados: pd.DataFrame,
    colunas_features: Iterable[str],
    fim_treino: int,
    tipo_scaler: str,
) -> Tuple[pd.DataFrame, object]:
    """Ajusta o scaler só no treino e apenas transforma validação/teste."""
    colunas = list(colunas_features)
    if fim_treino <= 0 or fim_treino > len(dados):
        raise ValueError("Limite temporal de treino inválido.")
    scaler = criar_scaler(tipo_scaler)
    scaler.fit(dados.iloc[:fim_treino][colunas])

    normalizados = dados.copy()
    # O tick_volume costuma chegar como inteiro. Construir explicitamente um
    # DataFrame float evita atribuição incompatível e avisos nas versões novas
    # do Pandas.
    valores_normalizados = pd.DataFrame(
        scaler.transform(dados[colunas]),
        columns=colunas,
        index=dados.index,
        dtype=float,
    )
    normalizados[colunas] = valores_normalizados
    return normalizados, scaler
