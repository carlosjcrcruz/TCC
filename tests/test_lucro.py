"""Contrato do rótulo econômico e da seleção de operações lucrativas."""

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

from src.forecast_config import ConfigPrevisao
from src.forecast_data import FEATURES, criar_atributos, limpar_candles
from src.forecast_experiment import salvar_json
from src.forecast_models import ModeloPrevisao
from src.prediction import carregar_e_prever, prever_ultima_janela, salvar_modelo
from src.profit_labels import criar_alvos_lucro
from src.trade_policy import aplicar_politica, avaliar_operacoes, selecionar_politica
from tests.test_previsao import candles


class TestRotuloLucro(unittest.TestCase):
    def test_usa_proxima_abertura_e_custos_nas_duas_pontas(self):
        c = ConfigPrevisao(horizonte=2)
        d = pd.DataFrame({"time": pd.date_range("2026-01-02 13:00", periods=7, freq="5min", tz="UTC"),
                          "open": [100, 100, 100, 100, 100, 100, 100],
                          "close": [90, 100, 100.1, 99.9, 100, 100, 100]})
        r = criar_alvos_lucro(d, c, .0002)
        self.assertAlmostEqual(r.retorno_liquido_compra.iloc[0], 1.001-1-.0002*(1+1.001))
        self.assertEqual(r.rotulo.iloc[0], 1)
        self.assertEqual(r.rotulo.iloc[1], 2)
        self.assertEqual(r.rotulo.iloc[2], 0)
        self.assertTrue(r.rotulo.iloc[-2:].isna().all())

    def test_lucro_pequeno_ja_pode_ser_correto_sem_exigir_08(self):
        c = ConfigPrevisao(horizonte=1)
        d = pd.DataFrame({"time": pd.date_range("2026-01-02 13:00", periods=3, freq="5min", tz="UTC"),
                          "open": [100., 100., 100.], "close": [100., 100.1, 100.]})
        r = criar_alvos_lucro(d, c, .0002)
        self.assertEqual(r.rotulo.iloc[0], 1)
        self.assertLess(d.close.iloc[1] / d.close.iloc[0] - 1, .008)

    def test_empate_nao_e_lucro_e_nao_ha_duas_direcoes_vencedoras(self):
        c = ConfigPrevisao(horizonte=1)
        d = pd.DataFrame({"time": pd.date_range("2026-01-02 13:00", periods=3, freq="5min", tz="UTC"),
                          "open": [100., 100., 100.], "close": [100., 100., 100.]})
        r = criar_alvos_lucro(d, c, 0)
        self.assertEqual(r.rotulo.iloc[0], 0)
        self.assertFalse(((r.retorno_liquido_compra > 0) & (r.retorno_liquido_venda > 0)).any())

    def test_gap_futuro_e_fronteira_de_sessao_continuam_inelegiveis(self):
        c = ConfigPrevisao(horizonte=3)
        d, _ = limpar_candles(candles(sessoes=2))
        d = d.drop(index=40).reset_index(drop=True)
        r = criar_alvos_lucro(d, c)
        self.assertTrue(r.rotulo.iloc[37:40].isna().all())
        self.assertTrue(r.rotulo.iloc[76:79].isna().all())


class TestPoliticaLucro(unittest.TestCase):
    def test_pipeline_lucro_integra_rotulo_decisao_e_backtest(self):
        from src.profit_experiment import executar_lucro
        d = candles(sessoes=60, por_sessao=40)
        c = ConfigPrevisao(horizonte=3, janela=12, folds=1, minimo_sinais=2)
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            d.to_csv(raiz / "dados.csv", index=False)
            pasta, tabela = executar_lucro(raiz / "dados.csv", raiz / "resultado", c,
                                           modelos=("logistica", "hist_gradient"))
            self.assertTrue((pasta / "CONCLUIDO.json").exists())
            self.assertIn("meta_70_lucro_ambas", tabela)
            pred = carregar_e_prever(d, pasta)
            self.assertEqual(pred["tipo_alvo"], "lucro_liquido")
            self.assertEqual(pred["custo_por_lado"], .0002)

    def test_abstencao_quando_nao_ha_confianca(self):
        p = np.array([[.2, .5, .3], [.05, .8, .15], [.05, .15, .8]])
        y = aplicar_politica(p, {"limiar_compra": .7, "limiar_venda": .7})
        np.testing.assert_array_equal(y, [0, 1, 2])

    def test_duas_direcoes_aprovadas_tem_desempate_deterministico(self):
        y = aplicar_politica([[.1, .45, .45]], {"limiar_compra": .4, "limiar_venda": .4})
        self.assertEqual(y[0], 1)

    def test_sinais_sobrepostos_nao_inflam_quantidade_de_operacoes(self):
        d = pd.DataFrame({"posicao_bruta": np.arange(10), "retorno_liquido_compra": .01,
                          "retorno_liquido_venda": -.01})
        m, indices = avaliar_operacoes(d, np.ones(10), horizonte=3, minimo=2)
        np.testing.assert_array_equal(indices, [0, 3, 6, 9])
        self.assertEqual(m["operacoes_compra"], 4)
        self.assertIsNone(m["taxa_lucro_venda"])
        self.assertFalse(m["meta_70_lucro_ambas"])

    def test_poucas_operacoes_nao_satisfazem_meta_de_suporte(self):
        d = pd.DataFrame({"posicao_bruta": [0, 10], "retorno_liquido_compra": [.01, -.01],
                          "retorno_liquido_venda": [-.01, .01]})
        m, _ = avaliar_operacoes(d, [1, 2], horizonte=10, minimo=20)
        self.assertEqual(m["taxa_lucro_compra"], 1.)
        self.assertEqual(m["taxa_lucro_venda"], 1.)
        self.assertFalse(m["meta_70_lucro_ambas"])

    def test_selecao_recebe_apenas_validacao_e_reporta_cobertura(self):
        n = 200
        d = pd.DataFrame({"posicao_bruta": np.arange(n)*10,
                          "retorno_liquido_compra": np.where(np.arange(n)%2 == 0, .01, -.01),
                          "retorno_liquido_venda": np.where(np.arange(n)%2 == 1, .01, -.01)})
        p = np.array([[.1, .8, .1] if i%2 == 0 else [.1, .1, .8] for i in range(n)])
        pol, m, busca = selecionar_politica(d, p, 10)
        self.assertTrue(m["meta_70_lucro_ambas"])
        self.assertGreaterEqual(len(busca), 1)
        self.assertTrue(0 < m["cobertura_sinais"] <= 1)
        self.assertEqual(len(aplicar_politica(p, pol)), n)

    def test_modelo_recarrega_politica_e_tipo_do_alvo(self):
        c = ConfigPrevisao(janela=12)
        d, _ = limpar_candles(candles())
        x = criar_atributos(d, c)[list(FEATURES)].dropna().to_numpy()
        s = StandardScaler().fit(x[:300])
        estimador = DummyClassifier(strategy="prior").fit(x[:3], [0, 1, 2])
        m = ModeloPrevisao("majoritaria", c, estimadores=[estimador], tipo_alvo="lucro_liquido", custo_por_lado=.0002,
                           politica_sinais={"limiar_compra": .8, "limiar_venda": .8})
        original = prever_ultima_janela(d, m, s, c)
        self.assertEqual(original["classe_predita"], 0)
        self.assertNotIn("limiar_retorno", original)
        with tempfile.TemporaryDirectory() as tmp:
            pasta = Path(tmp)
            salvar_modelo(m, s, pasta / "modelos/majoritaria")
            salvar_json(pasta / "manifesto.json", {"config": asdict(c)})
            salvar_json(pasta / "selecao_modelo.json", {"modelo_selecionado": "majoritaria"})
            self.assertEqual(original, carregar_e_prever(d, pasta))
        m.rotulagem_treinamento = "direcao_bruta"
        auxiliar = prever_ultima_janela(d, m, s, c)
        self.assertNotIn("prob_nao_lucrativa_ambas", auxiliar)
        self.assertIn("score_classe_neutra_auxiliar", auxiliar)


if __name__ == "__main__":
    unittest.main()
