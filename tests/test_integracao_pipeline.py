"""Teste de integração rápido do pipeline completo com dados sintéticos."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from config import ConfigProjeto
from src.experiment import executar_experimentos


class TestIntegracaoPipeline(unittest.TestCase):
    def test_sete_cenarios_geram_artefatos(self):
        gerador = np.random.default_rng(123)
        total = 420
        indice = np.arange(total)
        close = (
            25.0
            + 0.12 * np.sin(indice / 4.0)
            + 0.05 * np.sin(indice / 17.0)
            + gerador.normal(0.0, 0.015, total)
        )
        dados = pd.DataFrame(
            {
                "time": pd.date_range("2025-01-01", periods=total, freq="5min", tz="UTC"),
                "open": close + gerador.normal(0.0, 0.005, total),
                "high": close + 0.03,
                "low": close - 0.03,
                "close": close,
                "tick_volume": gerador.integers(100, 1000, total),
            }
        )

        with tempfile.TemporaryDirectory() as temporario:
            raiz = Path(temporario)
            csv = raiz / "candles_sinteticos.csv"
            dados.to_csv(csv, index=False)

            config = ConfigProjeto(
                usar_metatrader=False,
                csv_entrada=csv,
                pasta_saida=raiz / "saida",
                periodo_sma=8,
                periodo_ema_feature=8,
                periodo_volatilidade=8,
                periodo_rsi=6,
                ema_span_cenario=8,
                wavelet="db2",
                wavelet_nivel=2,
                wavelet_janela_causal=32,
                executar_busca_filtros=False,
                horizonte_futuro=3,
                limiar_retorno=0.0015,
                tamanho_janela=12,
                unidades_recorrentes=8,
                unidades_dense=4,
                dropout=0.1,
                epocas=1,
                batch_size=16,
                paciencia_early_stopping=1,
                verbose_treinamento=0,
            )
            tabela = executar_experimentos(config)

            self.assertEqual(len(tabela), 7)
            self.assertEqual(
                tabela["cenario"].tolist(),
                [
                    "Original",
                    "Ruído 0.0005",
                    "Ruído 0.001",
                    "Ruído 0.002",
                    "Ruído 0.005",
                    "EMA",
                    "Wavelet",
                ],
            )
            self.assertTrue((config.pasta_resultados / "comparacao_cenarios.csv").exists())
            self.assertEqual(len(list((config.pasta_resultados / "modelos").glob("*.keras"))), 7)
            self.assertEqual(
                len(list((config.pasta_resultados / "graficos").glob("matriz_confusao_*.png"))),
                7,
            )
            self.assertEqual(
                len(list(config.pasta_dados_processados.glob("*.csv"))),
                8,  # dados originais + sete datasets processados
            )


if __name__ == "__main__":
    unittest.main()
