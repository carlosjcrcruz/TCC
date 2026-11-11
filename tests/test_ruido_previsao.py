"""Experimento congelado: OHLC, causalidade e respostas financeiras comuns."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

from src.binary_events import FEATURES_EVENTOS, PlanoBinario, criar_contexto, salvar_binario
from src.bracket_strategy import SaidaBarreiras, criar_operacoes_barreiras
from src.forecast_config import ConfigPrevisao
from src.forecast_data import limpar_candles
from src.forecast_noise import (PlanoRuidoPrevisao, _hashes, executar_ruido_previsao,
    perturbar_ohlc, selecionar_filtros, tratar_ohlc)
from tests.test_previsao import candles


class TestRuidoPrevisao(unittest.TestCase):
    def test_ruido_preserva_geometria_e_volume(self):
        d,_=limpar_candles(candles(sessoes=3))
        r=perturbar_ohlc(d,.005,42)
        for coluna in ["open","high","low"]:
            np.testing.assert_allclose(r[coluna]/d[coluna],r.close/d.close)
        self.assertTrue((r.low<=r[["open","close"]].min(axis=1)).all())
        self.assertTrue((r.high>=r[["open","close"]].max(axis=1)).all())
        pd.testing.assert_series_equal(r.tick_volume,d.tick_volume)
        pd.testing.assert_series_equal(r.time,d.time)

    def test_intensidade_zero_reproduz_exatamente_ohlc(self):
        d,_=limpar_candles(candles(sessoes=2))
        pd.testing.assert_frame_equal(d,perturbar_ohlc(d,0.,42))

    def test_semente_e_prefixo_reproduziveis_e_escalas_pareadas(self):
        d,_=limpar_candles(candles(sessoes=3))
        a=perturbar_ohlc(d,.001,42)
        b=perturbar_ohlc(d,.002,42)
        pd.testing.assert_frame_equal(a.iloc[:123],perturbar_ohlc(d.iloc[:123],.001,42))
        np.testing.assert_allclose(b.close/d.close-1,2*(a.close/d.close-1),atol=1e-15)
        self.assertFalse(a.close.equals(perturbar_ohlc(d,.001,123).close))

    def test_filtros_causais_e_ohlc_coerente(self):
        d,_=limpar_candles(candles(sessoes=3))
        r=perturbar_ohlc(d,.002,42)
        for nome,param in [("EMA",{"span":2}),("Wavelet",{"wavelet":"db2","nivel":2,"janela_causal":64,"threshold_scale":.5,"threshold_mode":"soft"})]:
            completo=tratar_ohlc(r,nome,param)
            prefixo=tratar_ohlc(r.iloc[:123],nome,param)
            pd.testing.assert_frame_equal(completo.iloc[:123],prefixo)
            np.testing.assert_allclose(completo.high/completo.close,r.high/r.close)

    def test_selecao_filtros_nao_usa_futuro(self):
        d,_=limpar_candles(candles(sessoes=8))
        limite=d.time.iloc[400]
        plano=PlanoRuidoPrevisao(ema_spans=(2,3),wavelet_opcoes=(("db2",2,64,.5),))
        a,ra=selecionar_filtros(d,limite,plano)
        alterado=d.copy()
        alterado.loc[400:,["open","high","low","close"]]*=3
        b,rb=selecionar_filtros(alterado,limite,plano)
        self.assertEqual(a,b)
        self.assertEqual(ra,rb)

    def test_plano_rejeita_sementes_repetidas_ou_contaminadas(self):
        for plano in [PlanoRuidoPrevisao(sementes=(42,42)),PlanoRuidoPrevisao(sementes=(314159,)),
                      PlanoRuidoPrevisao(intensidades=()),PlanoRuidoPrevisao(escopos=("invalido",))]:
            with self.assertRaises(ValueError):
                plano.validar()

    def test_ensaio_nao_retreina_e_mantem_resultados_originais(self):
        d,_=limpar_candles(candles(sessoes=28))
        dias=pd.DatetimeIndex(d.time.dt.normalize().unique())
        contexto=criar_contexto(d,ConfigPrevisao())
        features=contexto[list(FEATURES_EVENTOS)].to_numpy(dtype=np.float32)
        finitas=np.isfinite(features).all(axis=1)
        scaler=StandardScaler().fit(features[finitas & (d.time<dias[12]).to_numpy()])
        x=scaler.transform(features[finitas])
        modelos=[DummyClassifier(strategy="constant",constant=1).fit(x,np.ones(len(x))) for _ in range(2)]
        plano_modelo=PlanoBinario(fim_treino=str(dias[12]))
        plano=PlanoRuidoPrevisao(inicio=str(dias[20]),fim=str(dias[-1]+pd.Timedelta(days=1)),
            sementes=(42,),intensidades=(.002,),ema_spans=(2,),wavelet_opcoes=(("db2",2,64,.5),),bootstrap=100,minimo=1)
        regra=SaidaBarreiras(3,"fixa",.002,1.)
        with tempfile.TemporaryDirectory() as tmp:
            raiz=Path(tmp)
            artefato=raiz/"modelo"
            d.to_csv(raiz/"dados.csv",index=False)
            salvar_binario(artefato,modelos,scaler,{"familia":"hist_binario","janela":1,"horizonte":3,
                "evento":"todos","politica":{"limiar_compra":.5,"limiar_venda":.5}},plano_modelo)
            (artefato/"modelo_estrategia.json").write_text(json.dumps({"regra":{"horizonte":3,"tipo":"fixa","alvo":.002,"razao_stop":1.},"evento":"reversao_media"}),encoding="utf-8")
            antes=_hashes(artefato)
            with patch.object(StandardScaler,"fit",side_effect=AssertionError("Não pode reajustar o scaler.")),patch.object(DummyClassifier,"fit",side_effect=AssertionError("Não pode retreinar.")):
                pasta,tabela,resumo=executar_ruido_previsao(raiz/"dados.csv",artefato,raiz/"out",plano)
            self.assertEqual(antes,_hashes(artefato))
            self.assertEqual(len(tabela),12)  # Seis condições e dois escopos.
            ref=pd.read_csv(pasta/"referencia_operacoes.csv")
            for caminho in (pasta/"predicoes").glob("*.csv.gz"):
                pred=pd.read_csv(caminho)
                for coluna in ["posicao_bruta","retorno_liquido_compra","retorno_liquido_venda","preco_entrada"]:
                    np.testing.assert_allclose(pred[coluna],ref[coluna],atol=1e-12)
            a=json.loads((pasta/"auditoria.json").read_text(encoding="utf-8"))
            self.assertTrue(a["precos_execucao_e_alvos_comuns"])
            self.assertFalse(a["meta_confirmada"])


if __name__=="__main__":
    unittest.main()
