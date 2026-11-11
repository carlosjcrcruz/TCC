"""Verifica temporalidade, inferência sem alvos e interpretação dos 70%."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

from src.forecast_backtest import simular
from src.forecast_config import ConfigPrevisao
from src.forecast_data import (FEATURES, criar_alvos, criar_atributos, extrair_entradas,
    indices_do_corte, limpar_candles, montar_amostras, planejar_cortes)
from src.forecast_evaluation import avaliar, selecionar_modelo
from src.forecast_models import ModeloPrevisao, treinar
from src.prediction import prever_ultima_janela


def candles(sessoes=24, por_sessao=80):
    rng = np.random.default_rng(482)
    tempos = pd.DatetimeIndex(np.concatenate([
        pd.date_range(d + pd.Timedelta(hours=13), periods=por_sessao, freq="5min", tz="UTC").to_numpy()
        for d in pd.bdate_range("2025-01-02", periods=sessoes)
    ]))
    retornos = rng.normal(0, .0015, len(tempos))
    close = 40 * np.exp(np.cumsum(retornos))
    abertura = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"time": tempos, "open": abertura, "high": np.maximum(close, abertura) + .02,
                         "low": np.minimum(close, abertura) - .02, "close": close,
                         "tick_volume": rng.integers(0, 500, len(tempos))})


class TestDadosPrevisao(unittest.TestCase):
    def setUp(self):
        self.config = ConfigPrevisao(horizonte=3, janela=12, limiar=.002, folds=2, bootstrap_sessoes=0)
        self.dados, _ = limpar_candles(candles())

    def test_atributos_nao_mudam_quando_acrescenta_futuro(self):
        completo = criar_atributos(self.dados, self.config)
        prefixo = criar_atributos(self.dados.iloc[:200].copy(), self.config)
        pd.testing.assert_frame_equal(completo.loc[:199, list(FEATURES)], prefixo[list(FEATURES)])

    def test_alvo_exige_intervalos_continuos_e_mesma_sessao(self):
        d = self.dados.drop(index=100).reset_index(drop=True)
        alvos = criar_alvos(d, self.config)
        for p in np.flatnonzero(alvos.alvo_elegivel):
            bloco = d.iloc[p:p+self.config.horizonte+1]
            self.assertTrue(bloco.time.diff().iloc[1:].eq(pd.Timedelta(minutes=5)).all())
            self.assertEqual(bloco.time.iloc[0].date(), bloco.time.iloc[-1].date())
            self.assertAlmostEqual(alvos.retorno_futuro.iloc[p], bloco.close.iloc[-1] / bloco.close.iloc[0] - 1)
        self.assertFalse(alvos.alvo_elegivel.iloc[77:80].any())
        self.assertFalse(alvos.alvo_elegivel.iloc[97:100].any())
        self.assertTrue(alvos.rotulo.iloc[-3:].isna().all())

    def test_janelas_preservam_indices_antes_de_remover_alvos(self):
        d = criar_alvos(criar_atributos(self.dados, self.config), self.config)
        amostras = montar_amostras(d, self.config)
        scaler = StandardScaler().fit(amostras.features_linhas[np.isfinite(amostras.features_linhas).all(axis=1)])
        x = extrair_entradas(amostras, np.array([20]), scaler, self.config, True)
        fim = amostras.finais_janelas[20]
        esperado = scaler.transform(amostras.features_linhas[fim-11:fim+1])
        np.testing.assert_allclose(x[0], esperado, atol=1e-6)

    def test_cortes_expurgam_por_time_alvo_e_testes_nao_sobrepoem(self):
        d = criar_alvos(criar_atributos(self.dados, self.config), self.config)
        a = montar_amostras(d, self.config)
        anteriores = set()
        ultimo_treino = 0
        for corte in planejar_cortes(d, self.config):
            p = indices_do_corte(a.metadados, corte)
            self.assertTrue((a.metadados.iloc[p["treino"]].time_alvo < corte.inicio_validacao).all())
            self.assertTrue((a.metadados.iloc[p["validacao"]].time_alvo < corte.inicio_teste).all())
            self.assertGreater(len(p["treino"]), ultimo_treino)
            ultimo_treino = len(p["treino"])
            atual = set(p["teste"])
            self.assertFalse(anteriores & atual)
            anteriores |= atual

    def test_rejeita_duplicata_conflitante(self):
        duplicata = self.dados.iloc[:1].copy()
        duplicata["tick_volume"] += 1
        with self.assertRaisesRegex(ValueError, "mesmo timestamp"):
            limpar_candles(pd.concat([self.dados, duplicata]))

    def test_inferencia_usa_ultimo_candle_sem_resposta_futura(self):
        atributos = criar_atributos(self.dados, self.config)
        matriz = atributos[list(FEATURES)].dropna().to_numpy()
        scaler = StandardScaler().fit(matriz[:300])
        media_antes = scaler.mean_.copy()
        estimador = DummyClassifier(strategy="constant", constant=1).fit(scaler.transform(matriz[:300]), np.ones(300))
        modelo = ModeloPrevisao("majoritaria", self.config, estimadores=[estimador])
        previsao = prever_ultima_janela(self.dados, modelo, scaler, self.config)
        self.assertEqual(pd.Timestamp(previsao["time_ultimo_candle"]), self.dados.time.iloc[-1])
        self.assertEqual(previsao["classe_predita"], 1)
        self.assertEqual(previsao["prob_compra"], 1)
        np.testing.assert_array_equal(media_antes, scaler.mean_)


class TestAvaliacaoPrevisao(unittest.TestCase):
    def test_acuracia_alta_sem_acertar_compra_venda_nao_atende_diagonal(self):
        real = np.array([0] * 90 + [1] * 5 + [2] * 5)
        m, _ = avaliar(real, np.zeros(100), ConfigPrevisao())
        self.assertTrue(m["meta_70_acuracia"])
        self.assertFalse(m["meta_70_diagonal_todas_classes"])
        self.assertIsNone(m["FDR_compra"])
        self.assertIsNone(m["precision_venda"])

    def test_teste_final_nao_pode_selecionar_modelo(self):
        tabela = pd.DataFrame({"fold": ["teste_final"], "modelo": ["x"], "f1_macro": [1.], "recall_macro": [1.]})
        with self.assertRaises(ValueError):
            selecionar_modelo(tabela)

    def test_regressao_compara_com_retorno_zero(self):
        retorno = np.array([.01, -.01, 0.])
        m, _ = avaliar([1, 2, 0], [0, 0, 0], ConfigPrevisao(), retorno_real=retorno, retorno_previsto=np.zeros(3))
        self.assertEqual(m["ganho_mse_vs_retorno_zero"], 0.)

    def test_backtest_entra_na_abertura_seguinte_e_nao_sobrepoe(self):
        c = ConfigPrevisao(horizonte=3)
        d, _ = limpar_candles(candles(sessoes=1))
        d = criar_alvos(d, c)
        pred = pd.DataFrame({"time": d.time.iloc[:10], "posicao_bruta": np.arange(10), "classe_predita": 1})
        trades, m = simular(d, pred, c)
        self.assertEqual(trades.preco_entrada.iloc[0], d.open.iloc[1])
        self.assertEqual(trades.preco_saida.iloc[0], d.close.iloc[3])
        self.assertTrue((trades.time_entrada.iloc[1:].to_numpy() >= trades.time_saida.iloc[:-1].to_numpy()).all())
        self.assertTrue(m.retorno_acumulado.is_monotonic_decreasing)

    def test_backtest_sem_operacoes_nao_declara_acerto_perfeito(self):
        c = ConfigPrevisao()
        d, _ = limpar_candles(candles(sessoes=1))
        d = criar_alvos(d, c)
        p = pd.DataFrame({"time": d.time.iloc[:10], "posicao_bruta": np.arange(10), "classe_predita": 0})
        trades, m = simular(d, p, c)
        self.assertTrue(trades.empty)
        self.assertTrue((m.retorno_acumulado == 0).all())
        self.assertTrue(m.taxa_operacoes_positivas.isna().all())


class TestIntegracaoPrevisao(unittest.TestCase):
    def test_arquiteturas_da_busca_treinam_e_focal_recarrega(self):
        from src.neural_search import EspecificacaoRede, treinar_candidato
        from src.forecast_models import tensorflow
        rng = np.random.default_rng(81)
        x = rng.normal(size=(30, 12, len(FEATURES))).astype(np.float32)
        y = np.arange(30) % 3
        c = ConfigPrevisao(epocas=1, sementes=(42,), unidades=4, batch_size=16, janela=12)
        for arquitetura, focal in [("gru", False), ("cnn", False), ("cnn_gru", False), ("lstm", True), ("mlp", False), ("cnn_dilatada", False)]:
            spec = EspecificacaoRede("teste", arquitetura, 12, unidades=4, focal=focal,
                                     criterio="diagonal" if arquitetura == "gru" else "f1_macro",
                                     ajustar_decisao=arquitetura == "gru")
            modelo = treinar_candidato(spec, x[:24], y[:24], x[24:], y[24:], c)
            p = modelo.prever(x[24:], np.zeros(6))["probabilidades"]
            self.assertEqual(p.shape, (6, 3))
            np.testing.assert_allclose(p.sum(axis=1), 1)
            with tempfile.TemporaryDirectory() as temporario:
                caminho = Path(temporario) / "rede.keras"
                modelo.estimadores[0].save(caminho)
                restaurada = tensorflow().keras.models.load_model(caminho, compile=False)
                np.testing.assert_allclose(restaurada(x[24:], training=False).numpy(), modelo.estimadores[0](x[24:], training=False).numpy(), atol=1e-6)

    def test_ajuste_decisao_persistido_na_inferencia(self):
        from dataclasses import asdict
        from src.forecast_experiment import salvar_json
        from src.prediction import carregar_e_prever, salvar_modelo
        d, _ = limpar_candles(candles())
        c = ConfigPrevisao(janela=12)
        matriz = criar_atributos(d, c)[list(FEATURES)].dropna().to_numpy()
        scaler = StandardScaler().fit(matriz[:300])
        dummy = DummyClassifier(strategy="prior").fit(scaler.transform(matriz[:4]), [0, 0, 1, 2])
        modelo = ModeloPrevisao("majoritaria", c, estimadores=[dummy], ajuste_decisao=[1., 4., .5])
        antes = prever_ultima_janela(d, modelo, scaler, c)
        self.assertEqual(antes["classe_predita"], 1)
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            salvar_modelo(modelo, scaler, raiz / "modelos/majoritaria")
            salvar_json(raiz / "manifesto.json", {"config": asdict(c)})
            salvar_json(raiz / "selecao_modelo.json", {"modelo_selecionado": "majoritaria"})
            depois = carregar_e_prever(d, raiz)
            self.assertEqual(antes, depois)

    def test_pipeline_exporta_e_recupera_inferencia(self):
        from src.forecast_experiment import executar_previsao
        from src.prediction import carregar_e_prever
        d = candles()
        config = ConfigPrevisao(horizonte=3, janela=12, limiar=.002, folds=2,
                               modelos=("majoritaria", "momentum", "logistica", "ridge_retorno"), bootstrap_sessoes=5)
        with tempfile.TemporaryDirectory() as temporario:
            raiz = Path(temporario)
            d.to_csv(raiz / "dados.csv", index=False)
            pasta, tabela = executar_previsao(raiz / "dados.csv", raiz / "saida", config)
            self.assertEqual(len(tabela), 4)
            self.assertTrue((pasta / "CONCLUIDO.json").exists())
            prevista = carregar_e_prever(d, pasta)
            self.assertEqual(pd.Timestamp(prevista["time_ultimo_candle"]), d.time.iloc[-1])
            numerica = carregar_e_prever(d, pasta, "ridge_retorno")
            self.assertIn("retorno_previsto", numerica)
            with self.assertRaises(ValueError):
                carregar_e_prever(d, pasta, "modelo_inexistente")
            self.assertEqual(len(list((pasta / "graficos").glob("*.png"))), 4)

    def test_redes_classificacao_e_regressao_treinam_e_salvam(self):
        from src.prediction import salvar_modelo
        from src.forecast_models import tensorflow
        rng = np.random.default_rng(97)
        x = rng.normal(size=(40, 6, len(FEATURES))).astype(np.float32)
        y = np.arange(40) % 3
        r = rng.normal(0, .005, 40)
        c = ConfigPrevisao(epocas=1, sementes=(42,), unidades=4, batch_size=16, janela=6)
        for nome in ["lstm", "lstm_retorno"]:
            modelo = treinar(nome, x[:30], y[:30], r[:30], x[30:], y[30:], r[30:], c)
            saida = modelo.prever(x[30:], np.zeros(10))
            self.assertEqual(saida["classes"].shape, (10,))
            with tempfile.TemporaryDirectory() as temporario:
                scaler = StandardScaler().fit(x[:30].reshape(-1, len(FEATURES)))
                salvar_modelo(modelo, scaler, temporario)
                recarregada = tensorflow().keras.models.load_model(Path(temporario) / "estimador_0.keras", compile=False)
                np.testing.assert_allclose(modelo.estimadores[0](x[30:], training=False).numpy(), recarregada(x[30:], training=False).numpy(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
