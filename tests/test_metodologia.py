"""Testes que não dependem de MetaTrader 5 nem de TensorFlow."""

import unittest

import numpy as np
import pandas as pd

from src.evaluation import avaliar_classificacao
from src.labeling import criar_rotulos
from src.noise import adicionar_ruido_gaussiano
from src.preprocessing import normalizar_sem_vazamento
from src.sequences import (
    LimitesTemporais,
    criar_sequencias,
    separar_sequencias_temporais,
)


class TestMetodologia(unittest.TestCase):
    def test_rotulos_compra_venda_e_ultimo_invalido(self):
        close = pd.Series([100.0, 102.0, 98.0, 100.0])
        resultado = criar_rotulos(close, horizonte=1, limiar=0.01)
        self.assertEqual(resultado["rotulo"].iloc[:3].tolist(), [1.0, 2.0, 1.0])
        self.assertTrue(np.isnan(resultado["rotulo"].iloc[-1]))

    def test_scaler_e_ajustado_somente_no_treino(self):
        dados = pd.DataFrame({"feature": np.arange(10.0)})
        normalizados, scaler = normalizar_sem_vazamento(
            dados, ["feature"], fim_treino=5, tipo_scaler="standard"
        )
        self.assertAlmostEqual(float(scaler.mean_[0]), 2.0)
        self.assertAlmostEqual(float(normalizados.iloc[:5]["feature"].mean()), 0.0)
        self.assertGreater(float(normalizados.iloc[5]["feature"]), 0.0)

    def test_sequencias_respeitam_ordem_e_expurgo(self):
        dados = pd.DataFrame(
            {"feature": np.arange(10.0), "rotulo": np.arange(10) % 3}
        )
        x, y, indices = criar_sequencias(dados, ["feature"], tamanho_janela=3)
        self.assertEqual(indices.tolist(), list(range(2, 10)))
        self.assertEqual(x[0, :, 0].tolist(), [0.0, 1.0, 2.0])

        conjuntos = separar_sequencias_temporais(
            x,
            y,
            indices,
            LimitesTemporais(fim_treino=6, fim_validacao=8, total=10),
            horizonte_rotulo=1,
        )
        self.assertEqual(conjuntos["treino"][2].tolist(), [2, 3, 4])
        self.assertEqual(conjuntos["validacao"][2].tolist(), [6])
        self.assertEqual(conjuntos["teste"][2].tolist(), [8, 9])

    def test_ruido_e_reprodutivel(self):
        close = pd.Series([100.0, 101.0, 102.0])
        primeiro = adicionar_ruido_gaussiano(close, 0.001, seed=7)
        segundo = adicionar_ruido_gaussiano(close, 0.001, seed=7)
        np.testing.assert_allclose(primeiro, segundo)

    def test_fpr_e_fdr_por_classe(self):
        real = np.array([0, 0, 1, 1, 2, 2])
        predito = np.array([0, 1, 1, 2, 2, 0])
        metricas, matriz = avaliar_classificacao(real, predito)
        self.assertEqual(matriz.shape, (3, 3))
        self.assertAlmostEqual(metricas["FPR_compra"], 1 / 4)
        self.assertAlmostEqual(metricas["FDR_compra"], 1 / 2)


if __name__ == "__main__":
    unittest.main()
