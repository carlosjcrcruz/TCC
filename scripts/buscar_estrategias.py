"""Compara mudanças de saída e entrada preservando a confirmação reservada."""
import argparse
import logging
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.binary_events import PlanoBinario
from src.bracket_strategy import SaidaBarreiras, executar_busca_estrategias


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=RAIZ/"dados_adicionais/PETR4_M5_mt5_99998_20261107.csv")
    p.add_argument("--saida", type=Path, default=RAIZ/"busca_estrategias")
    p.add_argument("--horizontes", default="10,20,40,60")
    p.add_argument("--alvos", default="0.001,0.002,0.003")
    p.add_argument("--atr", default="0.5,1")
    p.add_argument("--razoes-stop", default="1,2,3")
    p.add_argument("--familias", default="hist_binario,mlp_binario")
    p.add_argument("--epocas", type=int, default=30)
    p.add_argument("--objetivo", choices=["taxa","valor"], default="taxa")
    p.add_argument("--somente-validacao", action="store_true", help="Não abrir a confirmação, mesmo se houver aprovação na seleção.")
    a = p.parse_args()
    hs = [int(x) for x in a.horizontes.split(",")]
    razoes = [float(x) for x in a.razoes_stop.split(",")]
    regras = [SaidaBarreiras(h,tipo,float(alvo),r) for h in hs
        for tipo,alvos in [("fixa",a.alvos), ("atr",a.atr)] if alvos
        for alvo in alvos.split(",") for r in razoes]
    plano = PlanoBinario(epocas=a.epocas, familias=tuple(a.familias.split(",")))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    pasta,tabela,status = executar_busca_estrategias(a.csv,a.saida,plano,regras,objetivo=a.objetivo,confirmar=not a.somente_validacao)
    print(status)
    print(pasta)
