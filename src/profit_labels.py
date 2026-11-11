"""Alvos que representam o lucro líquido de uma operação executável simulada."""

import numpy as np

from src.forecast_data import criar_alvos


def criar_alvos_lucro(dados, config, custo_por_lado=.0002):
    if not 0 <= custo_por_lado < 1:
        raise ValueError("Custo por lado inválido.")
    d = criar_alvos(dados, config)
    entrada = d.open.shift(-1)
    saida = d.close.shift(-config.horizonte)
    razao = saida / entrada
    custo = custo_por_lado * (1 + razao)
    d["preco_entrada_futuro"] = entrada
    d["preco_saida_futuro"] = saida
    d["retorno_liquido_compra"] = razao - 1 - custo
    d["retorno_liquido_venda"] = 1 - razao - custo
    compra = d.retorno_liquido_compra > 1e-12
    venda = d.retorno_liquido_venda > 1e-12
    if (compra & venda).any():
        raise ValueError("As duas direções não podem lucrar na mesma janela com custo não negativo.")
    d["rotulo"] = np.select([compra, venda], [1., 2.], default=0.)
    colunas_alvo = ["rotulo", "preco_entrada_futuro", "preco_saida_futuro", "retorno_liquido_compra", "retorno_liquido_venda"]
    d.loc[~d.alvo_elegivel, colunas_alvo] = np.nan
    return d
