"""Coleta histórica pelo MetaTrader 5 e alternativa de leitura por CSV."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd


COLUNAS_OBRIGATORIAS = ["time", "open", "high", "low", "close", "tick_volume"]


class ErroMetaTrader(RuntimeError):
    """Erro de conexão ou de coleta no terminal MetaTrader 5."""


def _importar_mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise ErroMetaTrader(
            "O pacote MetaTrader5 não está instalado. Instale requirements.txt "
            "ou informe um CSV de entrada."
        ) from exc
    return mt5


def _resolver_timeframe(mt5, timeframe: str) -> int:
    """Converte, por exemplo, 'M5' no código de timeframe da API."""
    nome = f"TIMEFRAME_{timeframe.strip().upper()}"
    valor = getattr(mt5, nome, None)
    if valor is None:
        raise ValueError(f"Timeframe inválido: {timeframe!r}.")
    return valor


def conectar_mt5(caminho_terminal: Optional[str] = None):
    """Inicializa o terminal e devolve o módulo conectado da API."""
    mt5 = _importar_mt5()
    inicializado = (
        mt5.initialize(path=caminho_terminal)
        if caminho_terminal
        else mt5.initialize()
    )
    if not inicializado:
        erro = mt5.last_error()
        mt5.shutdown()
        raise ErroMetaTrader(f"Não foi possível conectar ao MetaTrader 5: {erro}")
    return mt5


def validar_dados(dados: pd.DataFrame) -> pd.DataFrame:
    """Valida e devolve somente as colunas necessárias, em ordem conhecida."""
    ausentes = [coluna for coluna in COLUNAS_OBRIGATORIAS if coluna not in dados]
    if ausentes:
        raise ValueError(f"Colunas ausentes no conjunto de dados: {ausentes}")
    if dados.empty:
        raise ValueError("O conjunto de dados está vazio.")
    return dados.loc[:, COLUNAS_OBRIGATORIAS].copy()


def coletar_candles_mt5(
    simbolo: str,
    timeframe: str,
    quantidade: int,
    caminho_terminal: Optional[str] = None,
    ignorar_candle_em_formacao: bool = True,
) -> pd.DataFrame:
    """Coleta candles históricos, sem enviar qualquer ordem ao mercado."""
    mt5 = conectar_mt5(caminho_terminal)
    try:
        if not mt5.symbol_select(simbolo, True):
            raise ErroMetaTrader(
                f"O ativo {simbolo!r} não está disponível no terminal: {mt5.last_error()}"
            )
        codigo_timeframe = _resolver_timeframe(mt5, timeframe)
        posicao_inicial = 1 if ignorar_candle_em_formacao else 0
        rates = mt5.copy_rates_from_pos(
            simbolo, codigo_timeframe, posicao_inicial, quantidade
        )
        if rates is None or len(rates) == 0:
            raise ErroMetaTrader(
                f"Nenhum candle foi retornado para {simbolo}: {mt5.last_error()}"
            )
        dados = pd.DataFrame(rates)
        dados["time"] = pd.to_datetime(dados["time"], unit="s", utc=True)
        return validar_dados(dados)
    finally:
        mt5.shutdown()


def carregar_csv(caminho: Path | str) -> pd.DataFrame:
    """Carrega um CSV que possua as mesmas seis colunas da coleta."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"CSV de entrada não encontrado: {caminho}")
    dados = pd.read_csv(caminho)
    dados.columns = [str(coluna).strip().lower() for coluna in dados.columns]
    dados = validar_dados(dados)

    # Aceita tanto datas textuais quanto epoch em segundos.
    if pd.api.types.is_numeric_dtype(dados["time"]):
        dados["time"] = pd.to_datetime(dados["time"], unit="s", utc=True)
    else:
        dados["time"] = pd.to_datetime(dados["time"], utc=True, errors="coerce")
    return dados


def obter_dados(config) -> pd.DataFrame:
    """Usa MT5 e, se configurado, faz fallback controlado para um CSV."""
    if config.usar_metatrader:
        try:
            return coletar_candles_mt5(
                simbolo=config.simbolo,
                timeframe=config.timeframe,
                quantidade=config.quantidade_candles,
                caminho_terminal=config.caminho_terminal_mt5,
                ignorar_candle_em_formacao=config.ignorar_candle_em_formacao,
            )
        except (ErroMetaTrader, ValueError) as exc:
            if config.csv_entrada:
                logging.warning("Falha no MetaTrader 5 (%s). Usando o CSV informado.", exc)
                return carregar_csv(config.csv_entrada)
            raise

    if not config.csv_entrada:
        raise ValueError("Informe csv_entrada quando usar_metatrader=False.")
    return carregar_csv(config.csv_entrada)


def salvar_dados_originais(dados: pd.DataFrame, pasta: Path | str) -> Path:
    """Salva uma cópia imutável da entrada usada no experimento."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / "dados_originais.csv"
    validar_dados(dados).to_csv(destino, index=False)
    return destino

