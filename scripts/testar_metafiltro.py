"""Um novo método: meta-rótulos, contexto M15/H1 e walk-forward exploratório."""
import argparse
import logging
from pathlib import Path
import shutil
import sys

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from src.meta_filter import executar_metafiltro


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv",type=Path,default=RAIZ/"dados_adicionais/PETR4_M5_mt5_99998_20261107.csv")
    p.add_argument("--saida",type=Path,default=RAIZ/"busca_metafiltro")
    a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(message)s",datefmt="%H:%M:%S")
    pasta,tabela,resumo=executar_metafiltro(a.csv,a.saida)
    shutil.copy2(pasta/"LEIA_RESULTADOS.md",RAIZ/"RESULTADOS_METAFILTRO.md")
    print(resumo)
    print(pasta)
