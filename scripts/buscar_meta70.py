"""Busca com confirmação reservada: dois rótulos binários e eventos passados."""

import argparse
import logging
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.binary_events import PlanoBinario
from src.binary_study import executar_busca_binaria


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=RAIZ / "dados_adicionais/PETR4_M5_mt5_99998_20261107.csv")
    parser.add_argument("--saida", type=Path, default=RAIZ / "busca_meta70")
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--horizontes", default="1,3,5,10,20,40,60")
    parser.add_argument("--bandas", default="0,0.5,1")
    parser.add_argument("--familias", default="hist_binario,mlp_binario")
    a = parser.parse_args()
    plano = PlanoBinario(epocas=a.epocas, horizontes=tuple(int(x) for x in a.horizontes.split(",")),
        bandas=tuple(float(x) for x in a.bandas.split(",")), familias=tuple(a.familias.split(",")))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    pasta, tabela, status = executar_busca_binaria(a.csv, a.saida, plano)
    print(status)
    print(pasta)
