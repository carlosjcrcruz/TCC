"""Replicação do procedimento do TCC com o novo alvo financeiro."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.binary_events import criar_contexto
from src.forecast_config import ConfigPrevisao
from src.forecast_data import limpar_candles
from src.noise import adicionar_ruido_gaussiano, suavizar_ema, wavelet_denoising
from src.prediction import carregar_e_prever
from src.tcc_noise_training import (CENARIOS,PlanoTCCRuido,contexto_cenario,cortes_tcc,
    criar_series_tcc,executar_tcc_ruido,metricas_binarias,selecionar_filtros_tcc,wavelet_causal_lotes)
from tests.test_previsao import candles


def plano_pequeno():
    return PlanoTCCRuido(epocas=1,paciencia=1,sementes_redes=(42,),minimo=2,
        ema_spans=(2,),wavelets=("db2",),niveis=(2,),janelas_wavelet=(64,),escalas_wavelet=(.5,))


class TestTCCRuidoTreino(unittest.TestCase):
    def test_wavelet_em_lotes_equivale_a_referencia_causal(self):
        rng=np.random.default_rng(171)
        serie=pd.Series(40+np.cumsum(rng.normal(0,.03,420)))
        for w in ["db2","db4","sym4"]:
            for n in [1,2,3]:
                for j in [64,65]:
                    p={"wavelet":w,"nivel":n,"janela_causal":j,"threshold_scale":.5,"threshold_mode":"soft"}
                    np.testing.assert_allclose(wavelet_causal_lotes(serie,**p),wavelet_denoising(serie,**p),atol=1e-12,rtol=1e-12)

    def test_wavelet_lotes_nao_antecipa_futuro(self):
        rng=np.random.default_rng(4)
        serie=pd.Series(40+np.cumsum(rng.normal(0,.1,400)))
        total=wavelet_causal_lotes(serie)
        pd.testing.assert_series_equal(total.iloc[:173],wavelet_causal_lotes(serie.iloc[:173]))

    def test_sete_cenarios_e_filtros_sobre_mesmo_ruido_002(self):
        s=pd.Series(np.linspace(40,42,200))
        f={"EMA":{"span":2},"Wavelet":{"wavelet":"db2","nivel":2,"janela_causal":64,"threshold_scale":.5,"threshold_mode":"soft"}}
        c=criar_series_tcc(s,f,42)
        self.assertEqual(tuple(c),CENARIOS)
        pd.testing.assert_series_equal(c["Ruido_0_002"],adicionar_ruido_gaussiano(s,.002,42))
        pd.testing.assert_series_equal(c["EMA"],suavizar_ema(c["Ruido_0_002"],2))
        self.assertFalse(np.allclose(c["EMA"],suavizar_ema(s,2)))

    def test_tcc_preserva_ohl_volume_e_altera_so_fechamento_de_entrada(self):
        d,_=limpar_candles(candles(sessoes=3))
        novo=adicionar_ruido_gaussiano(d.close,.005,42)
        ctx=contexto_cenario(d,novo)
        for c in ["open","high","low","tick_volume","time"]:
            pd.testing.assert_series_equal(ctx[c],d[c])
        np.testing.assert_allclose(ctx.close,novo)
        self.assertTrue(((ctx.close<ctx.low)|(ctx.close>ctx.high)).any())

    def test_filtros_so_usam_treino(self):
        s=pd.Series(40+np.cumsum(np.random.default_rng(3).normal(0,.1,500)))
        plano=plano_pequeno()
        a,ra=selecionar_filtros_tcc(s,300,plano)
        s.iloc[300:]*=3
        b,rb=selecionar_filtros_tcc(s,300,plano)
        self.assertEqual(a,b)
        self.assertEqual(ra,rb)

    def test_cortes_70_15_15_expurgam_alvos_que_atravessam_limites(self):
        t=pd.date_range("2024-01-01",periods=1000,freq="5min",tz="UTC")
        meta=pd.DataFrame({"time":t,"time_alvo":t+pd.Timedelta(minutes=30)})
        idx,l=cortes_tcc(meta,plano_pequeno())
        self.assertEqual(l["inicio_validacao"],t[700])
        self.assertEqual(l["inicio_teste"],t[850])
        self.assertLess(meta.iloc[idx["treino"]].time_alvo.max(),l["inicio_validacao"])
        self.assertLess(meta.iloc[idx["early"]].time_alvo.max(),l["inicio_ajuste"])
        self.assertLess(meta.iloc[idx["ajuste"]].time_alvo.max(),l["inicio_teste"])

    def test_matrizes_binarias_nao_forcam_rotulo_multiclasse(self):
        m=pd.DataFrame({"retorno_liquido_compra":[.01,-.01,.01,-.01],"retorno_liquido_venda":[.01,.01,-.01,-.01]})
        a=metricas_binarias(m,[1,1,0,2])
        self.assertEqual(a["compra"]["TP"],1)
        self.assertEqual(a["compra"]["FP"],1)
        self.assertEqual(a["compra"]["FDR"],.5)
        self.assertEqual(a["compra"]["FPR"],.5)
        self.assertIsNone(metricas_binarias(m,[0,0,0,0])["venda"]["FDR"])

    def test_plano_rejeita_sementes_vazias(self):
        with self.assertRaises(ValueError):
            PlanoTCCRuido(sementes_redes=()).validar()

    def test_sete_redes_treinadas_com_alvos_comuns_e_artefatos_proprios(self):
        from src.bracket_strategy import SaidaBarreiras
        from src.tcc_noise_training import ajustar_modelo as ajustar_original
        d,_=limpar_candles(candles(sessoes=50,por_sessao=40))
        with tempfile.TemporaryDirectory() as tmp:
            raiz=Path(tmp)
            d.to_csv(raiz/"dados.csv",index=False)
            with patch("src.tcc_noise_training.ajustar_modelo",wraps=ajustar_original) as ajustar:
                pasta,tabela,resumo=executar_tcc_ruido(raiz/"dados.csv",raiz/"out",plano_pequeno(),SaidaBarreiras(3))
            self.assertEqual(ajustar.call_count,7)
            self.assertEqual(len(tabela),7)
            self.assertEqual(len(resumo),7)
            ref=pd.read_csv(pasta/"referencia_alvos.csv")
            partes=pd.read_csv(pasta/"particoes.csv")
            teste=ref[partes.particao.eq("teste")].reset_index(drop=True)
            for nome in CENARIOS:
                pred=pd.read_csv(pasta/"predicoes"/f"{nome}_semente_42.csv.gz")
                for col in ["posicao_bruta","retorno_liquido_compra","retorno_liquido_venda","preco_entrada"]:
                    np.testing.assert_allclose(pred[col],teste[col])
                self.assertTrue((pasta/"modelos"/nome/"semente_42/scaler.joblib").exists())
            doc=json.loads((pasta/"CONCLUIDO.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["modelos_independentes"],7)
            previsao=carregar_e_prever(d,pasta/"modelos/EMA/semente_42")
            self.assertEqual(previsao["cenario"],"EMA")
            self.assertFalse(previsao["meta_confirmada"])


if __name__=="__main__":
    unittest.main()
