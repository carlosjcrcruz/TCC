import argparse
import logging
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.binary_events import PlanoBinario
from src.specialist_study import executar_especialistas


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=RAIZ / "dados_adicionais/PETR4_M5_mt5_99998_20261107.csv")
    p.add_argument("--saida", type=Path, default=RAIZ / "busca_especialistas")
    p.add_argument("--epocas", type=int, default=30)
    p.add_argument("--horizontes", default="1,3,4,5,6,8,10,12,15,20,40,60")
    p.add_argument("--bandas", default="0,0.5,1,1.5")
    a = p.parse_args()
    plano = PlanoBinario(epocas=a.epocas, horizontes=tuple(int(v) for v in a.horizontes.split(",")),
                         bandas=tuple(float(v) for v in a.bandas.split(",")))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    pasta, tabela, status = executar_especialistas(a.csv, a.saida, plano)
    print(status)
    print(pasta)
