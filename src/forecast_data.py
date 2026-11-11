"""Atributos causais, alvos contínuos e classes com cortes por sessão/data."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_collection import COLUNAS_OBRIGATORIAS
from src.preprocessing import calcular_rsi


FEATURES = (
    "retorno_1", "retorno_3", "retorno_5", "retorno_10", "retorno_20", "retorno_60",
    "corpo_relativo", "amplitude_relativa", "pavio_superior", "pavio_inferior",
    "distancia_sma20", "distancia_ema20", "volatilidade_20", "rsi14",
    "volume_relativo", "hora_sin", "hora_cos", "gap_log_minutos",
)


def limpar_candles(dados):
    ausentes = set(COLUNAS_OBRIGATORIAS) - set(dados.columns)
    if ausentes:
        raise ValueError(f"Colunas ausentes: {sorted(ausentes)}")
    d = dados.loc[:, COLUNAS_OBRIGATORIAS].copy()
    n = len(d)
    if pd.api.types.is_numeric_dtype(d.time):
        d["time"] = pd.to_datetime(d.time, unit="s", errors="coerce", utc=True)
    else:
        d["time"] = pd.to_datetime(d.time, errors="coerce", utc=True)
    for coluna in COLUNAS_OBRIGATORIAS[1:]:
        d[coluna] = pd.to_numeric(d[coluna], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan)
    invalido = d.isna().any(axis=1)
    invalido |= (d[["open", "high", "low", "close"]] <= 0).any(axis=1)
    invalido |= d.tick_volume < 0
    invalido |= d.high < d[["open", "close", "low"]].max(axis=1)
    invalido |= d.low > d[["open", "close", "high"]].min(axis=1)
    removidos = int(invalido.sum())
    d = d.loc[~invalido].drop_duplicates().sort_values("time")
    if d.time.duplicated().any():
        raise ValueError("Há candles diferentes com o mesmo timestamp; resolva a origem antes de treinar.")
    if d.empty:
        raise ValueError("Nenhum candle válido.")
    qualidade = {"linhas_recebidas": n, "invalidas_removidas": removidos,
                 "duplicatas_identicas_removidas": n - removidos - len(d), "linhas_validas": len(d)}
    return d.reset_index(drop=True), qualidade


def criar_atributos(dados, config):
    """Não recebe rótulos; a última linha pode ser usada em inferência."""
    d = dados.copy()
    close = d.close
    for lag in [1, 3, 5, 10, 20, 60]:
        d[f"retorno_{lag}"] = close.pct_change(lag, fill_method=None)
    d["corpo_relativo"] = (close - d.open) / d.open
    d["amplitude_relativa"] = (d.high - d.low) / d.open
    d["pavio_superior"] = (d.high - d[["open", "close"]].max(axis=1)) / d.open
    d["pavio_inferior"] = (d[["open", "close"]].min(axis=1) - d.low) / d.open
    d["distancia_sma20"] = close / close.rolling(20).mean() - 1
    d["distancia_ema20"] = close / close.ewm(span=20, adjust=False).mean() - 1
    d["volatilidade_20"] = d.retorno_1.rolling(20).std()
    d["rsi14"] = calcular_rsi(close, 14) / 100
    # +1 mantém válido o caso de volume zero e média zero.
    d["volume_relativo"] = (d.tick_volume + 1) / (d.tick_volume.rolling(20).mean() + 1) - 1
    local = d.time.dt.tz_convert(config.timezone)
    minutos = local.dt.hour * 60 + local.dt.minute
    d["hora_sin"] = np.sin(2 * np.pi * minutos / 1440)
    d["hora_cos"] = np.cos(2 * np.pi * minutos / 1440)
    gap = d.time.diff().dt.total_seconds().div(60).clip(lower=0)
    d["gap_log_minutos"] = np.log1p(gap)
    d["sessao"] = local.dt.strftime("%Y-%m-%d")
    return d.replace([np.inf, -np.inf], np.nan)


def classes_do_retorno(retorno, limiar):
    r = np.asarray(retorno)
    return np.select([r >= limiar, r <= -limiar], [1, 2], default=0).astype(np.int64)


def criar_alvos(dados, config):
    d = dados.copy()
    h = config.horizonte
    passo = pd.Timedelta(minutes=config.minutos_candle)
    # A soma em rolling é alinhada em t+h: todos os h intervalos precisam existir.
    intervalos = d.time.diff().eq(passo).astype(int)
    continuo = intervalos.rolling(h, min_periods=h).sum().shift(-h).eq(h)
    futuro_time = d.time.shift(-h)
    mesma_sessao = d.time.dt.tz_convert(config.timezone).dt.date.eq(
        futuro_time.dt.tz_convert(config.timezone).dt.date
    )
    d["time_alvo"] = futuro_time
    d["retorno_futuro"] = d.close.shift(-h) / d.close - 1
    d["alvo_elegivel"] = continuo & mesma_sessao & futuro_time.notna()
    d["rotulo"] = classes_do_retorno(d.retorno_futuro, config.limiar).astype(float)
    d.loc[~d.alvo_elegivel, ["rotulo", "retorno_futuro"]] = np.nan
    return d


@dataclass
class Amostras:
    features_linhas: np.ndarray
    finais_janelas: np.ndarray
    metadados: pd.DataFrame


def montar_amostras(dados, config):
    matriz = dados.loc[:, list(FEATURES)].to_numpy(dtype=np.float32)
    linhas_validas = np.isfinite(matriz).all(axis=1)
    janelas_validas = pd.Series(linhas_validas).rolling(config.janela).sum().eq(config.janela).to_numpy()
    mascara = janelas_validas & dados.alvo_elegivel.to_numpy()
    indices = np.flatnonzero(mascara)
    meta = dados.loc[mascara, ["time", "time_alvo", "sessao", "close", "retorno_futuro", "rotulo"]].copy()
    meta["rotulo"] = meta.rotulo.astype(int)
    meta["posicao_bruta"] = indices
    return Amostras(matriz, indices, meta.reset_index(drop=True))


@dataclass(frozen=True)
class CorteTemporal:
    nome: str
    inicio_validacao: pd.Timestamp
    inicio_teste: pd.Timestamp
    fim_teste: pd.Timestamp


def planejar_cortes(dados, config):
    sessoes = dados.groupby("sessao", sort=True).time.min().reset_index(drop=True)
    n = len(sessoes)
    n_final = max(1, int(np.ceil(n * config.fracao_teste_final)))
    n_dev = n - n_final
    bloco = max(1, int(n_dev * config.fracao_bloco_desenvolvimento))
    inicial = n_dev - (config.folds + 1) * bloco
    if inicial < 5 or n_final < 2:
        raise ValueError("Poucas sessões para os cortes: aumente o histórico ou reduza os folds.")
    cortes = []
    for i in range(config.folds):
        val = inicial + i * bloco
        teste = val + bloco
        cortes.append(CorteTemporal(f"fold_{i+1}", sessoes.iloc[val], sessoes.iloc[teste], sessoes.iloc[teste+bloco]))
    cortes.append(CorteTemporal("teste_final", sessoes.iloc[n_dev-bloco], sessoes.iloc[n_dev], dados.time.iloc[-1] + pd.Timedelta(microseconds=1)))
    return cortes


def indices_do_corte(meta, corte):
    # Os timestamps de fim do alvo controlam o expurgo, mesmo após remover linhas.
    treino = (meta.time < corte.inicio_validacao) & (meta.time_alvo < corte.inicio_validacao)
    validacao = (meta.time >= corte.inicio_validacao) & (meta.time < corte.inicio_teste) & (meta.time_alvo < corte.inicio_teste)
    teste = (meta.time >= corte.inicio_teste) & (meta.time < corte.fim_teste) & (meta.time_alvo < corte.fim_teste)
    partes = {"treino": np.flatnonzero(treino), "validacao": np.flatnonzero(validacao), "teste": np.flatnonzero(teste)}
    if any(len(v) == 0 for v in partes.values()):
        raise ValueError(f"Partição vazia em {corte.nome}; aumente o histórico.")
    return partes


def extrair_entradas(amostras, indices, scaler, config, sequencial=False):
    finais = amostras.finais_janelas[indices]
    if not sequencial:
        return scaler.transform(amostras.features_linhas[finais]).astype(np.float32)
    janelas = finais[:, None] - np.arange(config.janela - 1, -1, -1)[None, :]
    valores = amostras.features_linhas[janelas]
    forma = valores.shape
    return scaler.transform(valores.reshape(-1, forma[-1])).reshape(forma).astype(np.float32)
