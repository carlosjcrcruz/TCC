"""Simulação histórica: entrada na próxima abertura, saída no fechamento-alvo."""

import numpy as np
import pandas as pd


def simular(dados, predicoes, config):
    """Uma posição por vez; custos são cenários, não estimativas da corretora."""
    trades = []
    livre_a_partir = -1
    for p in predicoes.sort_values("time").itertuples():
        i = int(p.posicao_bruta)
        if p.classe_predita == 0 or i < livre_a_partir:
            continue
        entrada, saida = i + 1, i + config.horizonte
        if saida >= len(dados) or not dados.alvo_elegivel.iloc[i]:
            continue
        preco_entrada, preco_saida = float(dados.open.iloc[entrada]), float(dados.close.iloc[saida])
        direcao = 1 if p.classe_predita == 1 else -1
        retorno = direcao * (preco_saida / preco_entrada - 1)
        trades.append({
            "time_sinal": dados.time.iloc[i] + pd.Timedelta(minutes=config.minutos_candle),
            "time_entrada": dados.time.iloc[entrada],
            "time_saida": dados.time.iloc[saida] + pd.Timedelta(minutes=config.minutos_candle),
            "direcao": direcao, "preco_entrada": preco_entrada, "preco_saida": preco_saida,
            "retorno_bruto": retorno,
        })
        livre_a_partir = saida
    tabela = pd.DataFrame(trades, columns=["time_sinal", "time_entrada", "time_saida", "direcao", "preco_entrada", "preco_saida", "retorno_bruto"])
    metricas = []
    for custo in config.custos_por_lado:
        bruto = tabela.retorno_bruto.to_numpy()
        # Taxa sobre o notional de entrada e de saída, ambos referidos ao inicial.
        liquido = bruto - custo * (1 + tabela.preco_saida.to_numpy() / tabela.preco_entrada.to_numpy())
        patrimonio = np.r_[1.0, np.cumprod(1 + liquido)]
        pico = np.maximum.accumulate(patrimonio)
        ganhos, perdas = liquido[liquido > 0].sum(), -liquido[liquido < 0].sum()
        metricas.append({
            "custo_por_lado": float(custo), "custo_bps_por_lado": float(custo * 10000),
            "operacoes": len(tabela), "retorno_acumulado": float(patrimonio[-1] - 1),
            "max_drawdown_fechamentos": float(np.max(1 - patrimonio / pico)),
            "taxa_operacoes_positivas": float(np.mean(liquido > 0)) if len(liquido) else None,
            "retorno_medio_operacao": float(np.mean(liquido)) if len(liquido) else None,
            "profit_factor": float(ganhos / perdas) if perdas > 0 else None,
            "operacoes_compra": int((tabela.direcao == 1).sum()),
            "operacoes_venda": int((tabela.direcao == -1).sum()),
            "taxa_lucro_compra": float(np.mean(liquido[tabela.direcao.to_numpy() == 1] > 1e-12)) if (tabela.direcao == 1).any() else None,
            "taxa_lucro_venda": float(np.mean(liquido[tabela.direcao.to_numpy() == -1] > 1e-12)) if (tabela.direcao == -1).any() else None,
            "limitacao": "Drawdown medido só nas saídas; posição de 100% do patrimônio, sem aluguel, margem ou restrições de liquidez. Custos hipotéticos.",
        })
    return tabela, pd.DataFrame(metricas)
