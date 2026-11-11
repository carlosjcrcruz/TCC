"""Testes dos rótulos auxiliares e da separação da avaliação operacional."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.forecast_config import ConfigPrevisao
from src.forecast_data import criar_atributos, limpar_candles
from src.label_variants import ROTULAGENS, rotulos_alternativos
from src.profit_experiment import executar_lucro
from src.profit_labels import criar_alvos_lucro
from tests.test_previsao import candles


def serie_plana():
    return pd.DataFrame({"time": pd.date_range("2026-01-02 13:00", periods=80, freq="5min", tz="UTC"),
                         "open": 100., "close": 100., "high": 100.01, "low": 99.99, "tick_volume": 100})


class TestRotulosAlternativos(unittest.TestCase):
    def test_comparacao_congela_escolha_antes_do_resultado_posterior(self):
        from scripts.comparar_rotulos import comparar
        c = ConfigPrevisao(horizonte=3, janela=12, folds=1, minimo_sinais=2, sementes=(42,))
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            candles(sessoes=60, por_sessao=40).to_csv(raiz / "dados.csv", index=False)
            estudo = comparar(raiz / "dados.csv", raiz / "saida", c,
                              rotulagens=("direcao_bruta", "barreiras_002"), modelos=("hist_gradient",))
            escolha = json.loads((estudo / "escolha_antes_do_teste.json").read_text(encoding="utf-8"))
            self.assertIn(escolha["rotulagem"], ["direcao_bruta", "barreiras_002"])
            final = pd.read_csv(estudo / "resultado_posterior.csv")
            self.assertEqual(len(final), 3)  # escolhido e as duas referências operacionais.
            for busca in (estudo / "buscas").glob("*/*"):
                self.assertFalse((busca / "metricas_teste.csv").exists())

    def test_direcao_bruta_diferencia_custo_de_movimento(self):
        c = ConfigPrevisao(horizonte=3)
        d = serie_plana()
        d.loc[23, "close"] = 100.01
        d = criar_alvos_lucro(d, c)
        y = rotulos_alternativos(d, c, "direcao_bruta")
        self.assertEqual(y.iloc[20], 1)
        self.assertEqual(d.rotulo.iloc[20], 0)

    def test_barreira_ambigua_nao_apaga_caso_da_avaliacao_economica(self):
        c = ConfigPrevisao(horizonte=3)
        d = serie_plana()
        d.loc[21, ["high", "low"]] = [100.3, 99.7]
        d.loc[23, "close"] = 100.1
        d = criar_alvos_lucro(d, c)
        y = rotulos_alternativos(d, c, "barreiras_002")
        self.assertTrue(np.isnan(y.iloc[20]))
        self.assertTrue(d.alvo_elegivel.iloc[20])
        self.assertEqual(d.rotulo.iloc[20], 1)

    def test_primeira_barreira_nao_e_sinonimo_de_lucro_no_fechamento(self):
        c = ConfigPrevisao(horizonte=3)
        d = serie_plana()
        d.loc[21, ["high", "low", "close"]] = [100.25, 99.95, 100.1]
        d.loc[22, ["open", "high", "low", "close"]] = [100.1, 100.15, 99.7, 99.75]
        d.loc[23, ["open", "high", "low", "close"]] = [99.75, 99.8, 99.6, 99.7]
        d = criar_alvos_lucro(d, c)
        y = rotulos_alternativos(d, c, "barreiras_002")
        self.assertEqual(y.iloc[20], 1)
        self.assertEqual(d.rotulo.iloc[20], 2)

    def test_trajetoria_media_pode_diferir_do_ultimo_preco(self):
        c = ConfigPrevisao(horizonte=3)
        d = serie_plana()
        d.loc[21:23, "close"] = [101., 101., 99.7]
        d = criar_alvos_lucro(d, c)
        y = rotulos_alternativos(d, c, "trajetoria_media")
        self.assertEqual(y.iloc[20], 1)
        self.assertEqual(d.rotulo.iloc[20], 2)

    def test_zero_volatilidade_e_taxa_zero_nao_criam_compra_no_empate(self):
        c = ConfigPrevisao(horizonte=3)
        d = criar_alvos_lucro(serie_plana(), c, 0)
        y = rotulos_alternativos(d, c, "volatilidade_10", 0)
        self.assertEqual(y.iloc[30], 0)

    def test_futuro_apos_horizonte_nao_muda_rotulo(self):
        c = ConfigPrevisao(horizonte=3)
        bruto, _ = limpar_candles(candles(sessoes=3))
        antes = criar_alvos_lucro(criar_atributos(bruto, c), c)
        modificado = bruto.copy()
        modificado.loc[104:, ["open", "high", "low", "close"]] *= 2
        depois = criar_alvos_lucro(criar_atributos(modificado, c), c)
        for nome in ROTULAGENS:
            y1, y2 = rotulos_alternativos(antes, c, nome), rotulos_alternativos(depois, c, nome)
            np.testing.assert_allclose(y1.iloc[100], y2.iloc[100], equal_nan=True)

    def test_etapa_de_busca_nao_avalia_teste_e_mantem_mesmas_datas(self):
        c = ConfigPrevisao(horizonte=3, janela=12, folds=1, minimo_sinais=2)
        d = candles(sessoes=60, por_sessao=40)
        contagens = []
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            d.to_csv(raiz / "dados.csv", index=False)
            for nome in ["direcao_bruta", "barreiras_002"]:
                pasta, _ = executar_lucro(raiz / "dados.csv", raiz / nome, c,
                    modelos=("hist_gradient",), rotulagem=nome, apenas_validacao=True)
                self.assertTrue((pasta / "CONCLUIDO_VALIDACAO.json").exists())
                self.assertFalse((pasta / "metricas_teste.csv").exists())
                self.assertEqual(len(list((pasta / "predicoes").glob("teste*"))), 0)
                limites = json.loads((pasta / "particoes.json").read_text(encoding="utf-8"))
                contagens.append(limites["teste"]["amostras"])
            self.assertEqual(contagens[0], contagens[1])


if __name__ == "__main__":
    unittest.main()
