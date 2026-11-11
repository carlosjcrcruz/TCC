"""Ponto de entrada do projeto de TCC.

Este programa analisa dados históricos. Ele não contém chamadas de envio de
ordens, posições ou operações reais no MetaTrader 5.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import ConfigProjeto


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Desenvolve e avalia previsão temporal; ruído é um experimento separado."
    )
    parser.add_argument("--modo", choices=["lucro", "previsao", "prever", "ruido"], default="lucro",
                        help="Padrão: prever operações lucrativas. 'previsao' mantém o rótulo de 0,8%; 'prever' carrega uma execução; 'ruido' é legado.")
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
    parser.add_argument("--modelo", choices=["LSTM", "GRU"], help="Camada recorrente do experimento legado (--modo ruido).")
    parser.add_argument("--modelos", help="Modelos da previsão separados por vírgula (majoritaria é obrigatória).")
    parser.add_argument("--epocas", type=int, default=25, help="Máximo de épocas das redes de previsão.")
    parser.add_argument("--sementes", default="42,123,2026", help="Sementes das redes, separadas por vírgula.")
    parser.add_argument("--folds", type=int, default=3, help="Blocos de teste de desenvolvimento temporal.")
    parser.add_argument("--horizonte", type=int, default=10, help="Candles futuros consecutivos na mesma sessão.")
    parser.add_argument("--minutos-candle", type=int, default=5, help="Duração de cada candle na nova previsão.")
    parser.add_argument("--limiar", type=float, default=0.008, help="Barreira de retorno das classes, fixada antes da avaliação.")
    parser.add_argument("--sem-rede", action="store_true", help="Executa apenas referências e modelos tabulares.")
    parser.add_argument("--execucao", type=Path, help="Pasta da execução concluída para --modo prever.")
    parser.add_argument("--prever-modelo", help="Modelo salvo a inspecionar em --modo prever; padrão: selecionado no desenvolvimento.")
    parser.add_argument("--custo-por-lado", type=float, default=.0002, help="Taxa proporcional por lado no rótulo de lucro; padrão: 2 bps.")
    parser.add_argument("--minimo-operacoes", type=int, default=20, help="Mínimo de compras e de vendas para avaliar a meta de lucro.")
    parser.add_argument("--rotulo", default="lucro_liquido", help="Rotulagem de treino do modo lucro; veja ROTULAGENS_E_LUCRO.md.")
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
    argumentos = criar_parser().parse_args()
    try:
        if argumentos.modo == "ruido":
            from src.experiment import COLUNAS_COMPARACAO, executar_experimentos
            config = aplicar_argumentos(ConfigProjeto(), argumentos)
            tabela = executar_experimentos(config)
            print(tabela.loc[:, COLUNAS_COMPARACAO].to_string(index=False))
            return 0
        from src.forecast_config import ConfigPrevisao
        from src.forecast_experiment import executar_previsao
        from src.prediction import carregar_e_prever
        import json
        import pandas as pd

        raiz = Path(__file__).resolve().parent
        coletas = sorted((raiz / "dados_adicionais").glob("PETR4_M5_mt5_*.csv"))
        csv = argumentos.csv or (coletas[-1] if coletas else raiz / "dados_processados" / "dados_originais.csv")
        if argumentos.modo == "prever":
            if argumentos.execucao is None:
                raise ValueError("Informe --execucao com a pasta de um treinamento concluído.")
            d = pd.read_csv(csv)
            d.columns = [str(c).strip().lower() for c in d.columns]
            previsao = carregar_e_prever(d, argumentos.execucao, argumentos.prever_modelo)
            print(json.dumps(previsao, ensure_ascii=False, indent=2))
            return 0
        if argumentos.modelo is not None or argumentos.timeframe is not None or argumentos.simbolo is not None or argumentos.candles is not None:
            raise ValueError("--modelo, --timeframe, --simbolo e --candles são opções de coleta do modo legado. Na previsão use --csv e --minutos-candle.")
        config = ConfigPrevisao(epocas=argumentos.epocas, sementes=tuple(int(x) for x in argumentos.sementes.split(",")),
                               folds=argumentos.folds, horizonte=argumentos.horizonte, limiar=argumentos.limiar,
                               minutos_candle=argumentos.minutos_candle)
        if argumentos.modo == "lucro":
            from src.profit_experiment import MODELOS, executar_lucro
            config.minimo_sinais = argumentos.minimo_operacoes
            modelos = tuple(x.strip() for x in argumentos.modelos.split(",")) if argumentos.modelos else MODELOS
            if argumentos.sem_rede:
                modelos = tuple(x for x in modelos if x not in {"mlp_lucro", "gru_lucro"})
            saida = argumentos.saida or raiz / "previsao_lucro"
            pasta, tabela = executar_lucro(csv, saida, config, argumentos.custo_por_lado, modelos, rotulagem=argumentos.rotulo)
            print(tabela[["modelo", "operacoes_compra", "taxa_lucro_compra", "operacoes_venda", "taxa_lucro_venda", "meta_70_lucro_ambas"]].to_string(index=False))
            print(f"\nResultados do lucro líquido: {pasta}")
            return 0
        if argumentos.modelos:
            config.modelos = tuple(x.strip() for x in argumentos.modelos.split(","))
        if argumentos.sem_rede:
            config.modelos = tuple(x for x in config.modelos if not x.startswith("lstm"))
        saida = argumentos.saida or Path(__file__).resolve().parent / "previsao_sem_ruido"
        pasta, tabela = executar_previsao(csv, saida, config)
    except Exception:
        logging.exception("O experimento foi interrompido por um erro.")
        return 1

    print("\nAvaliação retrospectiva sem ruído:")
    print(tabela[["modelo", "accuracy", "f1_macro", "recall_macro", "meta_70_diagonal_todas_classes"]].to_string(index=False))
    print(f"\nResultados e modelos: {pasta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
