"""Ponto de entrada do projeto de TCC.

Este programa analisa dados históricos. Ele não contém chamadas de envio de
ordens, posições ou operações reais no MetaTrader 5.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import ConfigProjeto
from src.experiment import COLUNAS_COMPARACAO, executar_experimentos


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Avalia a influência do ruído em falsos sinais financeiros."
    )
    parser.add_argument("--simbolo", help="Ativo no MetaTrader 5, por exemplo PETR4.")
    parser.add_argument("--timeframe", help="Timeframe da API, por exemplo M5.")
    parser.add_argument("--candles", type=int, help="Quantidade de candles históricos.")
    parser.add_argument("--csv", type=Path, help="CSV alternativo ou de fallback.")
    parser.add_argument(
        "--sem-mt5",
        action="store_true",
        help="Não tenta conectar ao MT5; exige --csv.",
    )
    parser.add_argument("--saida", type=Path, help="Pasta raiz para todos os artefatos.")
    parser.add_argument("--modelo", choices=["LSTM", "GRU"], help="Camada recorrente.")
    return parser


def aplicar_argumentos(config: ConfigProjeto, argumentos) -> ConfigProjeto:
    if argumentos.simbolo:
        config.simbolo = argumentos.simbolo
    if argumentos.timeframe:
        config.timeframe = argumentos.timeframe
    if argumentos.candles:
        config.quantidade_candles = argumentos.candles
    if argumentos.csv:
        config.csv_entrada = argumentos.csv
    if argumentos.sem_mt5:
        config.usar_metatrader = False
    if argumentos.saida:
        config.pasta_saida = argumentos.saida
    if argumentos.modelo:
        config.tipo_modelo = argumentos.modelo
    return config


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    config = aplicar_argumentos(ConfigProjeto(), criar_parser().parse_args())
    try:
        tabela = executar_experimentos(config)
    except Exception:
        logging.exception("O experimento foi interrompido por um erro.")
        return 1

    print("\nResumo dos resultados:")
    print(tabela.loc[:, COLUNAS_COMPARACAO].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
