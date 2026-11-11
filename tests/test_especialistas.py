"""Execução conjunta, horizontes distintos e confirmação protegida."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.forecast_data import limpar_candles
from src.prediction import carregar_e_prever
from src.specialist_study import avaliar_par, executar_especialistas, selecionar_limiar_lado
from src.trade_policy import avaliar_operacoes
from tests.test_binario_eventos import plano_teste
from tests.test_previsao import candles


class TestEspecialistas(unittest.TestCase):
    def test_horizonte_da_venda_controla_sobreposicao(self):
        m = pd.DataFrame({"posicao_bruta": np.arange(7),
                          "retorno_liquido_compra": .01, "retorno_liquido_venda": .02})
        r, usados = avaliar_operacoes(m, [2, 1, 1, 1, 1, 1, 1], 1, minimo=1, horizonte_venda=4)
        np.testing.assert_array_equal(usados, [0, 4, 5, 6])
        self.assertEqual(r["operacoes_venda"], 1)
        self.assertEqual(r["operacoes_compra"], 3)

    def test_limiar_independente_nao_exige_operacoes_da_outra_direcao(self):
        m = pd.DataFrame({"posicao_bruta": np.arange(20),
                          "retorno_liquido_compra": .01, "retorno_liquido_venda": -.01})
        p = np.tile([0., .8, .9], (20, 1))
        corte = selecionar_limiar_lado(m, p, np.ones(20, dtype=bool), 1, 1, minimo=20)
        self.assertAlmostEqual(corte, .8)

    def test_uniao_nao_negocia_casos_indisponiveis_e_respeita_horizontes(self):
        tempo = pd.date_range("2025-01-02", periods=5, freq="5min", tz="UTC")
        c = pd.DataFrame({"time": tempo[[0, 2, 4]], "posicao_bruta": [0, 2, 4],
            "retorno_liquido_compra": [.01, -.01, .01], "score": [.8, .8, .8], "permitido": True})
        v = pd.DataFrame({"time": tempo[[1, 3]], "posicao_bruta": [1, 3],
            "retorno_liquido_venda": [.02, -.02], "score": [.9, .9], "permitido": True})
        r, pred = avaliar_par({"selecao": c, "limiar": .7, "horizonte": 1},
                             {"selecao": v, "limiar": .7, "horizonte": 3}, "selecao", minimo=1)
        np.testing.assert_array_equal(pred.sinal, [1, 2, 1, 2, 1])
        self.assertEqual(r["operacoes"], 3)  # Compra 0, venda 1, compra 4.
        self.assertEqual(r["lucrativas_compra"], 2)
        self.assertEqual(r["lucrativas_venda"], 1)

    def test_falha_na_validacao_nao_abre_confirmacao(self):
        d, _ = limpar_candles(candles(sessoes=70))
        plano = plano_teste(d)
        def constantes(familia, modelos, x):
            return np.tile([0., .6, .4], (len(x), 1))
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            d.to_csv(raiz / "dados.csv", index=False)
            with patch("src.specialist_study.prever_scores", side_effect=constantes) as score:
                pasta, _, status = executar_especialistas(raiz / "dados.csv", raiz / "saida", plano)
            self.assertFalse(status["confirmacao_avaliada"])
            self.assertFalse((pasta / "resultado_confirmacao.json").exists())
            self.assertEqual(score.call_count, 2)

    def test_fluxo_completo_em_fixture_sintetica_previsivel(self):
        d, _ = limpar_candles(candles(sessoes=70))
        close = np.where(np.arange(len(d)) % 2 == 0, 100., 101.)
        abertura = np.r_[close[0], close[:-1]]
        d["open"], d["close"] = abertura, close
        d["high"], d["low"] = np.maximum(abertura, close)+.02, np.minimum(abertura, close)-.02
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            d.to_csv(raiz / "dados.csv", index=False)
            pasta, _, status = executar_especialistas(raiz / "dados.csv", raiz / "saida", plano_teste(d))
            self.assertTrue(status["meta_confirmada"])
            pred = carregar_e_prever(d, pasta / "artefato")
            self.assertIn(pred["sinal"], ["compra", "venda"])
            self.assertEqual(pred["tipo_modelo"], "especialistas_eventos")


if __name__ == "__main__":
    unittest.main()
