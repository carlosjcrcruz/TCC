"""Compara rótulos na validação e só depois avalia o candidato selecionado."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pandas as pd

from src.forecast_config import ConfigPrevisao
from src.forecast_experiment import salvar_json
from src.label_variants import ROTULAGENS
from src.profit_experiment import executar_lucro
from src.trade_policy import pontuar


def comparar(csv, saida, config, rotulagens=ROTULAGENS,
             modelos=("hist_gradient", "mlp_lucro", "gru_lucro"), custo=.0002):
    if not rotulagens or not set(rotulagens) <= set(ROTULAGENS):
        raise ValueError("Rotulagens inválidas.")
    raiz = Path(saida) / datetime.now(timezone.utc).strftime("estudo_%Y%m%dT%H%M%S_%fZ")
    raiz.mkdir(parents=True, exist_ok=False)
    salvar_json(raiz / "plano_antes_dos_resultados.json", {"rotulagens": list(rotulagens), "modelos": list(modelos),
        "config": asdict(config), "custo_por_lado": custo, "semente_busca": 42,
        "avaliacao": "Operações líquidas, mesma entrada/saída e horizonte para todos os rótulos.",
        "selecao": "Maior menor taxa de operações lucrativas por direção na validação, com suporte mínimo.",
        "teste": "Apenas o candidato escolhido será avaliado no período posterior; período já examinado antes, portanto estudo exploratório."})
    linhas = []
    for rotulo in rotulagens:
        c = deepcopy(config)
        c.sementes = (42,)
        pasta, tabela = executar_lucro(csv, raiz / "buscas" / rotulo, c, custo,
                                      modelos=modelos, rotulagem=rotulo, apenas_validacao=True)
        for registro in tabela.to_dict("records"):
            linhas.append({"rotulagem": rotulo, "pasta_busca": str(pasta), **registro})
        pd.DataFrame(linhas).to_csv(raiz / "comparacao_validacao.csv", index=False)
        logging.info("Rótulo %s concluído na validação, sem avaliar o teste posterior.", rotulo)
    escolhido = max(linhas, key=pontuar)
    salvar_json(raiz / "escolha_antes_do_teste.json", escolhido)
    logging.info("Escolha congelada: %s / %s", escolhido["rotulagem"], escolhido["modelo"])
    pasta, final = executar_lucro(csv, raiz / "confirmacao", deepcopy(config), custo,
                                  modelos=(escolhido["modelo"],), rotulagem=escolhido["rotulagem"])
    final.to_csv(raiz / "resultado_posterior.csv", index=False)
    salvar_json(raiz / "CONCLUIDO.json", {"concluido": True, "execucao_modelo": str(pasta),
        "rotulagem": escolhido["rotulagem"], "modelo": escolhido["modelo"]})
    mostrar = pd.DataFrame(linhas)
    texto = ["# Comparação de rótulos para operações lucrativas", "",
        "A meta operacional é 70% das compras executadas e 70% das vendas executadas terminarem com lucro após taxas. A entrada, saída, horizonte e custos foram mantidos em todos os candidatos. Não houve spread ou slippage.", "",
        f"Horizonte: {config.horizonte} candles de {config.minutos_candle} minutos. Taxa de referência: {custo*10000:g} pontos-base por lado. Suporte mínimo: {config.minimo_sinais} operações de cada direção.", "",
        "Os rótulos auxiliares podem representar movimento bruto, intensidade relativa à volatilidade, trajetória ou primeira barreira atingida. Seu acerto não foi confundido com lucro: todos foram comparados pelo mesmo resultado líquido da operação de saída fixa.", "",
        "Os candidatos foram comparados somente na validação. A escolha foi registrada antes de avaliar o candidato final no período posterior. As operações não se sobrepõem e o suporte mínimo é exigido por direção.", "",
        "Rótulos de barreira ambíguos, por tocarem os dois limites dentro do mesmo candle, foram excluídos apenas do treinamento e do Early Stopping. Todos os casos elegíveis permaneceram na avaliação financeira; não se eliminaram casos difíceis do teste.", "",
        "## Resultados da validação", "",
        "| Rótulo | Modelo | Compras | % lucrativas compra | Vendas | % lucrativas venda |",
        "|---|---|---:|---:|---:|---:|"]
    pct = lambda v: "NA" if pd.isna(v) else f"{v:.2%}"
    for r in linhas:
        texto.append(f"| {r['rotulagem']} | {r['modelo']} | {r['operacoes_compra']} | {pct(r['taxa_lucro_compra'])} | {r['operacoes_venda']} | {pct(r['taxa_lucro_venda'])} |")
    texto += ["", f"Escolhido na validação: **{escolhido['rotulagem']} / {escolhido['modelo']}**.", "",
        "## Avaliação posterior do escolhido", "",
        "| Estratégia | Compras | Compras lucrativas | % compra | Vendas | Vendas lucrativas | % venda | Retorno acumulado |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in final.to_dict("records"):
        texto.append(f"| {r['modelo']} | {r['operacoes_compra']} | {r['lucrativas_compra']} | {pct(r['taxa_lucro_compra'])} | {r['operacoes_venda']} | {r['lucrativas_venda']} | {pct(r['taxa_lucro_venda'])} | {pct(r['retorno_acumulado'])} |")
    candidato = final.loc[final.modelo == escolhido["modelo"]].iloc[0]
    texto += ["", f"Meta observada de 70% nas duas direções, com suporte mínimo: **{'sim' if candidato.meta_70_lucro_ambas else 'não'}**.", "",
        "O período posterior já havia sido inspecionado em outros objetivos. Esta comparação é exploratória e precisa de confirmação prospectiva. Comparar muitos candidatos na validação pode produzir otimismo; não é garantia de repetição futura.", "",
        f"Modelo, política de emissão, backtest e rótulos detalhados: `{pasta}`.", "",
        "Os modelos da busca usaram a semente 42. Se uma rede é escolhida, a confirmação usa as sementes declaradas no plano e média das probabilidades. Para modelos tabulares, usa-se o ajuste e a semente fixados no plano. Os limiares são reajustados na validação, sem usar os resultados do teste."]
    (raiz / "LEIA_RESULTADOS.md").write_text("\n".join(texto) + "\n", encoding="utf-8")
    print(mostrar[["rotulagem", "modelo", "taxa_lucro_compra", "taxa_lucro_venda"]].to_string(index=False))
    print(f"\nEstudo salvo em: {raiz}")
    return raiz


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    coletas = sorted((RAIZ / "dados_adicionais").glob("PETR4_M5_mt5_*.csv"))
    p.add_argument("--csv", type=Path, default=coletas[-1] if coletas else RAIZ / "dados_processados/dados_originais.csv")
    p.add_argument("--saida", type=Path, default=RAIZ / "comparacao_rotulos")
    p.add_argument("--rotulos", default=",".join(ROTULAGENS))
    p.add_argument("--modelos", default="hist_gradient,mlp_lucro,gru_lucro")
    p.add_argument("--epocas", type=int, default=30)
    p.add_argument("--sementes", default="42,123,2026")
    p.add_argument("--custo-por-lado", type=float, default=.0002)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    c = ConfigPrevisao(epocas=a.epocas, paciencia=5, sementes=tuple(int(s) for s in a.sementes.split(",")))
    comparar(a.csv, a.saida, c, tuple(a.rotulos.split(",")), tuple(a.modelos.split(",")), a.custo_por_lado)
