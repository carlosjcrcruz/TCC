"""Causalidade multiescala, expurgo temporal e emissão do segundo estágio."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.binary_events import criar_contexto
from src.bracket_strategy import SaidaBarreiras
from src.forecast_config import ConfigPrevisao
from src.forecast_data import limpar_candles
from src.meta_filter import (PlanoMeta, atributos_multiescala, executar_metafiltro,
    scores_meta, separar_janelas, treinar_variantes)
from src.prediction import carregar_e_prever
from tests.test_previsao import candles


class TestMetafiltro(unittest.TestCase):
    def test_candles_maiores_e_direcao_nao_antecipam_o_futuro(self):
        d,_=limpar_candles(candles(sessoes=20))
        ctx=criar_contexto(d,ConfigPrevisao())
        completo,lado=atributos_multiescala(ctx)
        for fim in [123,189,701]:
            parcial,lp=atributos_multiescala(criar_contexto(d.iloc[:fim],ConfigPrevisao()))
            pd.testing.assert_frame_equal(completo.iloc[:fim],parcial)
            np.testing.assert_array_equal(lado[:fim],lp)

    def test_h1_do_candle_atual_so_aparece_ao_fechar_a_hora(self):
        d,_=limpar_candles(candles(sessoes=20))
        ctx=criar_contexto(d,ConfigPrevisao())
        x,_=atributos_multiescala(ctx)
        # Todas as emissões antes de 14:00 ainda não têm o H1 iniciado às 13:00.
        self.assertTrue(x.loc[:10,"m60_amplitude"].isna().all())
        self.assertTrue(np.isfinite(x.loc[11,"m60_amplitude"]))

    def test_indices_expurgam_alvos_nas_fronteiras(self):
        t=pd.date_range("2023-01-01",periods=900,freq="D",tz="UTC")
        m=pd.DataFrame({"time":t,"time_alvo":t+pd.Timedelta(days=2)})
        idx,f=separar_janelas(m,"2024-07-01T00:00:00Z",12,3,3)
        self.assertLess(m.iloc[idx["treino"]].time_alvo.max(),pd.Timestamp(f["inicio_calibracao"]))
        self.assertLess(m.iloc[idx["calibracao"]].time_alvo.max(),pd.Timestamp(f["inicio_avaliacao"]))
        self.assertLess(m.iloc[idx["avaliacao"]].time_alvo.max(),pd.Timestamp(f["fim_avaliacao"]))
        self.assertFalse(set(idx["treino"])&set(idx["avaliacao"]))

    def test_score_so_e_atribuido_a_direcao_da_regra(self):
        x=np.arange(12).reshape(4,3)
        a=DummyClassifier(strategy="constant",constant=1).fit(x,[0,1,0,1])
        p=scores_meta([a],x,np.array([1,-1,0,1]))
        np.testing.assert_array_equal(p,[[0,1,0],[0,0,1],[0,0,0],[0,1,0]])

    def test_ensemble_e_media_dos_modelos(self):
        x=np.arange(12).reshape(4,3)
        a=DummyClassifier(strategy="constant",constant=1).fit(x,[0,1,0,1])
        b=DummyClassifier(strategy="constant",constant=0).fit(x,[0,1,0,1])
        p=scores_meta([a,b],x,np.ones(4))
        np.testing.assert_allclose(p[:,1],.5)

    def test_uma_classe_nao_inverte_probabilidade_de_lucro(self):
        rng=np.random.default_rng(3)
        x=rng.normal(size=(60,5))
        modelos=treinar_variantes(x,np.zeros(60,dtype=int))
        p=scores_meta(modelos["ensemble"],x,np.ones(60))
        np.testing.assert_array_equal(p,np.zeros((60,3)))

    def test_plano_invalido_nao_inicia_uma_busca_vazia(self):
        for plano in [PlanoMeta(variantes=()),PlanoMeta(meses_treino=0),PlanoMeta(evento="todos")]:
            with self.assertRaises(ValueError):
                plano.validar()

    def test_fluxo_walkforward_e_artefato_em_dados_sinteticos(self):
        # Fixture aleatória: testa execução do software, sem exigir taxa alta.
        d,_=limpar_candles(candles(sessoes=400,por_sessao=20))
        plano=PlanoMeta(inicio_avaliacao="2026-01-01T00:00:00Z",minimo=2,
            variantes=("logistica",),evento="rsi_extremo",meses_treino=12)
        with tempfile.TemporaryDirectory() as tmp:
            raiz=Path(tmp)
            d.to_csv(raiz/"dados.csv",index=False)
            pasta,tabela,resumo=executar_metafiltro(raiz/"dados.csv",raiz/"out",plano,SaidaBarreiras(1))
            self.assertGreater(len(tabela),0)
            self.assertTrue(resumo["exploratorio"])
            self.assertFalse(resumo["meta_confirmada"])
            # A mesma fixture não contém o recorte fixo de janeiro–junho/2024.
            self.assertEqual(resumo["janeiro_junho_2024"]["operacoes"],0)
            previsao=carregar_e_prever(d,pasta/"modelo_experimental")
            self.assertEqual(previsao["tipo_modelo"],"metafiltro_temporal")
            self.assertFalse(previsao["meta_confirmada"])
            pred=pd.read_csv(pasta/"predicoes_walkforward.csv")
            self.assertEqual(int(pred.operacao_executada.sum()),resumo["walkforward"]["operacoes"])


if __name__=="__main__":
    unittest.main()
