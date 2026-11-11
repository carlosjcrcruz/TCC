"""Saídas executáveis conservadoras e proteção contra alta taxa com prejuízo."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.binary_events import criar_contexto
from src.bracket_strategy import (SaidaBarreiras, amplitude_passada, avaliar_barreiras,
    ajustar_valor, criar_operacoes_barreiras, executar_busca_estrategias, reservar_confirmacao)
from src.forecast_config import ConfigPrevisao
from src.forecast_data import limpar_candles
from src.prediction import carregar_e_prever
from tests.test_binario_eventos import plano_teste
from tests.test_previsao import candles


def serie_pequena():
    d, _ = limpar_candles(candles(sessoes=1, por_sessao=8))
    d[["open", "close"]] = 100.
    d["high"], d["low"] = 100.01, 99.99
    return criar_contexto(d, ConfigPrevisao())


class TestEstrategias(unittest.TestCase):
    def test_alvo_compra_paga_taxas_sobre_entrada_e_saida(self):
        d = serie_pequena()
        d.loc[1,"high"] = 100.3
        m = criar_operacoes_barreiras(d, SaidaBarreiras(2,"fixa",.002,1))
        self.assertEqual(m.loc[0,"motivo_compra"], "alvo")
        self.assertEqual(m.loc[0,"duracao_compra"], 1)
        self.assertAlmostEqual(m.loc[0,"retorno_liquido_compra"], .002-.0002*2.002)

    def test_alvo_venda_tem_taxa_na_saida_mais_baixa(self):
        d = serie_pequena()
        d.loc[1,"low"] = 99.7
        m = criar_operacoes_barreiras(d, SaidaBarreiras(2,"fixa",.002,1))
        self.assertEqual(m.loc[0,"motivo_venda"], "alvo")
        self.assertAlmostEqual(m.loc[0,"retorno_liquido_venda"], .002-.0002*1.998)

    def test_candle_ambiguo_e_perda_em_ambas_direcoes(self):
        d = serie_pequena()
        d.loc[1,["high","low"]] = [100.3,99.7]
        m = criar_operacoes_barreiras(d, SaidaBarreiras(2,"fixa",.002,1))
        for lado in ["compra","venda"]:
            self.assertTrue(m.loc[0,f"ambiguo_{lado}"])
            self.assertLess(m.loc[0,f"retorno_liquido_{lado}"], 0)

    def test_gap_stop_execucao_no_open_pior_que_o_limite(self):
        d = serie_pequena()
        d.loc[2,["open","high","low","close"]] = [98.,98.1,97.9,98.]
        m = criar_operacoes_barreiras(d, SaidaBarreiras(3,"fixa",.002,1))
        self.assertAlmostEqual(m.loc[0,"retorno_liquido_compra"], -.02-.0002*1.98)
        self.assertEqual(m.loc[0,"duracao_compra"], 2)

    def test_gap_alvo_no_open_antecede_toque_do_stop(self):
        d = serie_pequena()
        d.loc[2,["open","high","low","close"]] = [101.,101.2,99.,100.]
        m = criar_operacoes_barreiras(d, SaidaBarreiras(3,"fixa",.002,1))
        self.assertEqual(m.loc[0,"motivo_compra"], "alvo")
        self.assertFalse(m.loc[0,"ambiguo_compra"])
        self.assertAlmostEqual(m.loc[0,"retorno_liquido_compra"], .002-.0002*2.002)

    def test_prazo_e_continuidade_obrigatorios(self):
        d = serie_pequena()
        d.loc[2,"close"] = 100.02
        m = criar_operacoes_barreiras(d, SaidaBarreiras(2,"fixa",.002,1))
        self.assertEqual(m.loc[0,"motivo_compra"], "prazo")
        self.assertAlmostEqual(m.loc[0,"retorno_liquido_compra"], .0002-.0002*2.0002)
        quebrado = d.drop(index=2).reset_index(drop=True)
        b = criar_operacoes_barreiras(quebrado, SaidaBarreiras(2,"fixa",.002,1))
        self.assertNotIn(0, b.posicao_bruta.to_list())

    def test_atr_usa_apenas_passado(self):
        d, _ = limpar_candles(candles(sessoes=2))
        a = criar_contexto(d, ConfigPrevisao())
        np.testing.assert_allclose(amplitude_passada(a).iloc[:100], amplitude_passada(a.iloc[:100]), equal_nan=True)

    def test_escore_de_valor_preserva_ordem_e_intervalo(self):
        rng=np.random.default_rng(712)
        x=rng.normal(size=(400,3))
        y=np.column_stack([x[:,0]*.001,-x[:,0]*.002])
        heads=ajustar_valor(x,y)
        consulta=np.array([[-2.,0.,0.],[2.,0.,0.]])
        p=heads[0].predict_proba(consulta)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(((p>=0)&(p<=1)).all())
        self.assertLess(p[0,1],p[1,1])

    def test_80_porcento_com_prejuizo_nao_e_aprovacao(self):
        retornos = np.tile([.001]*8+[-.01]*2, 10)
        m = pd.DataFrame({"posicao_bruta": np.arange(100), "duracao_compra": 1,"duracao_venda": 1,
            "retorno_liquido_compra": retornos,"retorno_liquido_venda": retornos,
            "ambiguo_compra": False,"ambiguo_venda": False})
        sinais = np.r_[np.ones(50),np.full(50,2)]
        r,_ = avaliar_barreiras(m,sinais)
        self.assertTrue(r["meta_70_lucro_ambas"])
        self.assertLess(r["retorno_acumulado"],0)
        self.assertFalse(r["aprovado"])

    def test_sobreposicao_usa_saida_real_da_barreira(self):
        m = pd.DataFrame({"posicao_bruta": np.arange(6), "duracao_compra": 3,"duracao_venda": 2,
            "retorno_liquido_compra": .01,"retorno_liquido_venda": .02,
            "ambiguo_compra": False,"ambiguo_venda": False})
        _,usados = avaliar_barreiras(m,[1,2,2,2,1,1],minimo=1)
        np.testing.assert_array_equal(usados,[0,3,5])

    def test_confirmacao_bloqueada_quando_validacao_falha(self):
        d,_ = limpar_candles(candles(sessoes=70))
        with tempfile.TemporaryDirectory() as tmp:
            raiz=Path(tmp)
            d.to_csv(raiz/"dados.csv",index=False)
            with patch("src.bracket_strategy.prever_scores",side_effect=lambda f,m,x: np.tile([0,.6,.4],(len(x),1))) as prever:
                pasta,_,status=executar_busca_estrategias(raiz/"dados.csv",raiz/"out",plano_teste(d),[SaidaBarreiras(1)],("todos",))
            self.assertFalse(status["confirmacao_avaliada"])
            self.assertEqual(prever.call_count,2)

    def test_artefato_e_confirmacao_com_fixture_sintetica(self):
        d,_ = limpar_candles(candles(sessoes=70))
        close=np.where(np.arange(len(d))%2==0,100.,101.)
        abertura=np.r_[close[0],close[:-1]]
        d["open"],d["close"]=abertura,close
        d["high"],d["low"]=np.maximum(abertura,close)+.02,np.minimum(abertura,close)-.02
        with tempfile.TemporaryDirectory() as tmp:
            raiz=Path(tmp)
            d.to_csv(raiz/"dados.csv",index=False)
            pasta,_,status=executar_busca_estrategias(raiz/"dados.csv",raiz/"out",plano_teste(d),[SaidaBarreiras(1)],("todos",))
            self.assertTrue(status["aprovado_na_confirmacao"])
            pred=carregar_e_prever(d,pasta/"modelo_experimental")
            self.assertEqual(pred["tipo_modelo"],"lucro_com_barreiras")
            self.assertAlmostEqual(pred["alvo_fracao"],.002)
            self.assertAlmostEqual(pred["stop_fracao"],.004)

    def test_regressao_de_valor_pode_ser_salva_e_recarregada(self):
        d,_=limpar_candles(candles(sessoes=70))
        close=np.where(np.arange(len(d))%2==0,100.,101.)
        abertura=np.r_[close[0],close[:-1]]
        d["open"],d["close"]=abertura,close
        d["high"],d["low"]=np.maximum(abertura,close)+.02,np.minimum(abertura,close)-.02
        with tempfile.TemporaryDirectory() as tmp:
            raiz=Path(tmp)
            d.to_csv(raiz/"dados.csv",index=False)
            pasta,_,status=executar_busca_estrategias(raiz/"dados.csv",raiz/"out",plano_teste(d),[SaidaBarreiras(1)],("todos",),objetivo="valor")
            self.assertTrue(status["meta_confirmada"])
            pred=carregar_e_prever(d,pasta/"modelo_experimental")
            self.assertEqual(pred["objetivo_treino"],"valor")
            self.assertIn("não probabilidade",pred["natureza_scores"])

    def test_confirmacao_nao_pode_ser_reaberta_na_mesma_busca(self):
        d,_=limpar_candles(candles(sessoes=70))
        with tempfile.TemporaryDirectory() as tmp:
            raiz=Path(tmp)
            d.to_csv(raiz/"dados.csv",index=False)
            plano=plano_teste(d)
            self.assertTrue(reservar_confirmacao(raiz/"dados.csv",raiz/"out",plano,"primeira",{}))
            self.assertFalse(reservar_confirmacao(raiz/"dados.csv",raiz/"out",plano,"segunda",{}))


if __name__ == "__main__":
    unittest.main()
