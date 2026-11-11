"""Causalidade, rótulos de clareza e proteção da confirmação reservada."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.binary_events import (FEATURES_EVENTOS, PlanoBinario, ajustar_modelo,
    carregar_prever_binario, criar_contexto, mascarar_eventos,
    perda_binaria_mascarada, prever_scores, rotulos_binarios)
from src.binary_study import executar_busca_binaria
from src.forecast_config import ConfigPrevisao
from src.forecast_data import limpar_candles
from src.forecast_models import tensorflow
from src.trade_policy import avaliar_operacoes, selecionar_politica
from tests.test_previsao import candles


def plano_teste(dados):
    dias = pd.DatetimeIndex(dados.time.dt.normalize().unique())
    return PlanoBinario(fim_treino=str(dias[25]), fim_early=str(dias[35]), fim_ajuste=str(dias[45]),
        fim_selecao=str(dias[55]), fim_teste=str(dias[-1] + pd.Timedelta(days=1)),
        horizontes=(1,), bandas=(0.,), familias=("hist_binario",), eventos=("todos",),
        minimo=2, epocas=1, sementes_finais=(42,))


class TestBinarioEventos(unittest.TestCase):
    def test_contexto_e_eventos_nao_enxergam_o_futuro(self):
        d, _ = limpar_candles(candles(sessoes=3))
        total = criar_contexto(d, ConfigPrevisao())
        prefixo = criar_contexto(d.iloc[:150], ConfigPrevisao())
        pd.testing.assert_frame_equal(total.loc[:149, list(FEATURES_EVENTOS)], prefixo[list(FEATURES_EVENTOS)])

    def test_rotulo_claro_remove_ambiguidade_so_do_treinamento(self):
        m = pd.DataFrame({"posicao_bruta": [0, 1, 2], "retorno_liquido_compra": [.003, .0001, -.003],
                          "retorno_liquido_venda": [-.003, -.0005, .003]})
        y = rotulos_binarios(m, np.full(3, .002), 1, 1.)
        np.testing.assert_array_equal(y[0], [1., 0.])
        self.assertTrue(np.isnan(y[1]).all())
        np.testing.assert_array_equal(y[2], [0., 1.])
        resultado, _ = avaliar_operacoes(m, [1, 1, 1], 1, minimo=1)
        self.assertEqual(resultado["operacoes"], 3)
        self.assertEqual(resultado["lucrativas_compra"], 2)

    def test_retorno_indisponivel_nao_vira_perda_conhecida(self):
        m = pd.DataFrame({"retorno_liquido_compra": [np.nan], "retorno_liquido_venda": [np.nan]})
        self.assertTrue(np.isnan(rotulos_binarios(m, [0.], 1, 0.)).all())

    def test_direcao_proibida_nao_impede_sinal_da_direcao_permitida(self):
        p = np.array([[0., .7, .9], [0., .9, .7]])
        pol = {"limiar_compra": .6, "limiar_venda": .6}
        sinais = mascarar_eventos(p, pol, (np.array([True, False]), np.array([False, True])))
        np.testing.assert_array_equal(sinais, [1, 2])

    def test_ajuste_de_politica_respeita_os_eventos(self):
        m = pd.DataFrame({"posicao_bruta": np.arange(40), "retorno_liquido_compra": .01,
                          "retorno_liquido_venda": -.01})
        p = np.column_stack([np.zeros(40), np.linspace(.5, .9, 40), np.linspace(.9, .5, 40)])
        permissao = np.arange(40) < 5
        _, resultado, _ = selecionar_politica(m, p, 1, minimo=1,
            permissao_compra=permissao, permissao_venda=np.zeros(40, dtype=bool))
        self.assertLessEqual(resultado["operacoes_compra"], 5)
        self.assertEqual(resultado["operacoes_venda"], 0)

    def test_perda_mascarada_e_gradientes_sao_finitos(self):
        tf = tensorflow()
        p = tf.Variable([[.7, .2], [.3, .9]], dtype=tf.float32)
        y = tf.constant([[1., np.nan], [0., 1.]], dtype=tf.float32)
        with tf.GradientTape() as tape:
            perda = tf.reduce_mean(perda_binaria_mascarada(y, p))
        grad = tape.gradient(perda, p).numpy()
        self.assertTrue(np.isfinite(perda.numpy()))
        self.assertTrue(np.isfinite(grad).all())
        self.assertEqual(grad[0, 1], 0)

    def test_rede_binaria_com_rotulos_parciais_pode_ser_recarregada(self):
        rng = np.random.default_rng(31)
        x = rng.normal(size=(40, 12, len(FEATURES_EVENTOS))).astype(np.float32)
        y = (rng.uniform(size=(40, 2)) > .5).astype(np.float32)
        y[:5, 1] = np.nan
        plano = PlanoBinario(epocas=1)
        modelos, _ = ajustar_modelo("mlp_binario", x[:30], y[:30], x[30:], y[30:], plano)
        antes = prever_scores("mlp_binario", modelos, x[30:])
        with tempfile.TemporaryDirectory() as tmp:
            caminho = Path(tmp) / "modelo.keras"
            modelos[0].save(caminho)
            restaurado = tensorflow().keras.models.load_model(caminho, compile=False)
            depois = prever_scores("mlp_binario", [restaurado], x[30:])
            np.testing.assert_allclose(antes, depois, atol=1e-6)

    def test_plano_sem_modelos_nao_inicia_busca_vazia(self):
        with self.assertRaises(ValueError):
            PlanoBinario(familias=()).validar()

    def test_confirmacao_nao_e_avaliada_se_validacao_falhar(self):
        d, _ = limpar_candles(candles(sessoes=70))
        plano = plano_teste(d)
        def scores_constantes(familia, modelos, x):
            return np.tile([0., .6, .4], (len(x), 1))
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            d.to_csv(raiz / "dados.csv", index=False)
            with patch("src.binary_study.prever_scores", side_effect=scores_constantes) as score:
                pasta, _, status = executar_busca_binaria(raiz / "dados.csv", raiz / "saida", plano)
            self.assertFalse(status["confirmacao_avaliada"])
            self.assertFalse((pasta / "resultado_confirmacao.json").exists())
            self.assertEqual(score.call_count, 2)  # Ajuste e seleção; nenhuma previsão de teste.

    def test_confirmacao_e_artefato_funcionam_em_sinal_sintetico_previsivel(self):
        d, _ = limpar_candles(candles(sessoes=70))
        close = np.where(np.arange(len(d)) % 2 == 0, 100., 101.)
        abertura = np.r_[close[0], close[:-1]]
        d["open"], d["close"] = abertura, close
        d["high"], d["low"] = np.maximum(abertura, close)+.02, np.minimum(abertura, close)-.02
        plano = plano_teste(d)
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            d.to_csv(raiz / "dados.csv", index=False)
            pasta, _, status = executar_busca_binaria(raiz / "dados.csv", raiz / "saida", plano)
            self.assertTrue(status["confirmacao_avaliada"])
            self.assertTrue(status["meta_confirmada"])
            previsao = carregar_prever_binario(d, pasta / "artefato")
            self.assertIn(previsao["sinal"], ["compra", "venda"])
            self.assertTrue(previsao["meta_confirmada"])


if __name__ == "__main__":
    unittest.main()
