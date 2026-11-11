"""Sete cenários do TCC com redes novas independentes de compra/venda lucrativa."""
import argparse
import logging
from pathlib import Path
import sys

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from src.tcc_noise_training import PlanoTCCRuido,executar_tcc_ruido
from scripts.resumir_tcc_nova_predicao import gerar


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv",type=Path,default=RAIZ/"dados_adicionais/PETR4_M5_mt5_99998_20261107.csv")
    p.add_argument("--saida",type=Path,default=RAIZ/"ruido_tcc_nova_predicao")
    p.add_argument("--sementes",default="42,123,2026")
    p.add_argument("--epocas",type=int,default=30)
    a=p.parse_args()
    plano=PlanoTCCRuido(sementes_redes=tuple(int(x) for x in a.sementes.split(",")),epocas=a.epocas)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(message)s",datefmt="%H:%M:%S")
    pasta,tabela,resumo=executar_tcc_ruido(a.csv,a.saida,plano)
    gerar(pasta,RAIZ/"RESULTADOS_TCC_NOVA_PREDICAO.md")
    print(resumo.to_string(index=False))
    print(pasta)
