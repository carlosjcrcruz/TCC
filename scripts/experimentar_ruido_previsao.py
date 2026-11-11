"""Executa ruído/EMA/Wavelet no novo previsor congelado de lucro."""
import argparse
import logging
from pathlib import Path
import sys

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from src.forecast_noise import PlanoRuidoPrevisao,executar_ruido_previsao
from scripts.resumir_ruido_previsao import gerar


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv",type=Path,default=RAIZ/"dados_adicionais/PETR4_M5_mt5_99998_20261107.csv")
    p.add_argument("--artefato",type=Path,default=RAIZ/"busca_estrategias/estrategias_20261107T173657_410746Z/modelo_experimental")
    p.add_argument("--saida",type=Path,default=RAIZ/"ruido_nova_previsao")
    p.add_argument("--sementes",default="42,123,2026,7,17,31,73,101,211,509")
    p.add_argument("--intensidades",default="0.0005,0.001,0.002,0.005")
    p.add_argument("--inicio",default="2024-01-01T00:00:00Z")
    p.add_argument("--fim",default="2024-07-01T00:00:00Z")
    a=p.parse_args()
    plano=PlanoRuidoPrevisao(inicio=a.inicio,fim=a.fim,sementes=tuple(int(x) for x in a.sementes.split(",")),
        intensidades=tuple(float(x) for x in a.intensidades.split(",")))
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(message)s",datefmt="%H:%M:%S")
    pasta,tabela,resumo=executar_ruido_previsao(a.csv,a.artefato,a.saida,plano)
    gerar(pasta,RAIZ/"RESULTADOS_RUIDO_NOVA_PREVISAO.md")
    print(resumo.to_string(index=False))
    print(pasta)
