"""Emissão seletiva de sinais, escolhida somente na validação temporal."""

import numpy as np
import pandas as pd


def aplicar_politica(probabilidades, politica, permissao_compra=None, permissao_venda=None):
    p = np.asarray(probabilidades, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("Escores inválidos.")
    c, v = politica.get("limiar_compra"), politica.get("limiar_venda")
    aceita_c = np.zeros(len(p), dtype=bool) if c is None else p[:, 1] >= c
    aceita_v = np.zeros(len(p), dtype=bool) if v is None else p[:, 2] >= v
    for permitida in [permissao_compra, permissao_venda]:
        if permitida is not None and len(permitida) != len(p):
            raise ValueError("Máscaras de eventos incompatíveis.")
    if permissao_compra is not None:
        aceita_c &= np.asarray(permissao_compra, dtype=bool)
    if permissao_venda is not None:
        aceita_v &= np.asarray(permissao_venda, dtype=bool)
    # Se ambos passam, usa o maior excesso relativo ao limiar. Não olha o futuro.
    excesso_c = np.full(len(p), -np.inf) if c is None else (p[:, 1] - c) / max(1 - c, 1e-12)
    excesso_v = np.full(len(p), -np.inf) if v is None else (p[:, 2] - v) / max(1 - v, 1e-12)
    return np.select([aceita_c & (~aceita_v | (excesso_c >= excesso_v)), aceita_v], [1, 2], default=0).astype(int)


def avaliar_operacoes(meta, sinais, horizonte, meta_acerto=.7, minimo=20, horizonte_venda=None):
    sinais = np.asarray(sinais, dtype=int)
    if len(sinais) != len(meta) or not meta.posicao_bruta.is_monotonic_increasing or not np.isin(sinais, [0, 1, 2]).all():
        raise ValueError("Sinais incompatíveis ou posições fora de ordem.")
    livres, escolhidos = -1, []
    posicoes = meta.posicao_bruta.to_numpy()
    for j in np.flatnonzero(sinais):
        if posicoes[j] < livres:
            continue
        escolhidos.append(j)
        livres = posicoes[j] + (horizonte if sinais[j] == 1 or horizonte_venda is None else horizonte_venda)
    usados = np.asarray(escolhidos, dtype=int)
    direcoes = sinais[usados]
    longos = meta.retorno_liquido_compra.to_numpy()[usados]
    curtos = meta.retorno_liquido_venda.to_numpy()[usados]
    liquido = np.where(direcoes == 1, longos, curtos)
    if not np.isfinite(liquido).all():
        raise ValueError("Retornos indisponíveis em operações avaliadas.")
    patrimonio = np.r_[1., np.cumprod(1 + liquido)]
    resultado = {"operacoes": len(usados), "taxa_lucro_total": float(np.mean(liquido > 1e-12)) if len(usados) else None,
                 "cobertura_sinais": float(np.mean(sinais != 0)) if len(sinais) else 0.,
                 "retorno_acumulado": float(patrimonio[-1] - 1),
                 "drawdown_saidas": float(np.max(1 - patrimonio / np.maximum.accumulate(patrimonio)))}
    for classe, nome in [(1, "compra"), (2, "venda")]:
        r = liquido[direcoes == classe]
        acertos = int((r > 1e-12).sum())
        resultado[f"operacoes_{nome}"] = len(r)
        resultado[f"lucrativas_{nome}"] = acertos
        resultado[f"taxa_lucro_{nome}"] = acertos / len(r) if len(r) else None
    resultado["suporte_suficiente"] = all(resultado[f"operacoes_{n}"] >= minimo for n in ["compra", "venda"])
    resultado["meta_70_lucro_ambas"] = bool(resultado["suporte_suficiente"] and all(
        resultado[f"taxa_lucro_{n}"] >= meta_acerto for n in ["compra", "venda"]))
    return resultado, usados


def pontuar(resultado):
    if not resultado["suporte_suficiente"]:
        return (-1., -1., resultado["operacoes"])
    taxas = [resultado["taxa_lucro_compra"], resultado["taxa_lucro_venda"]]
    return (min(taxas), sum(taxas) / 2, resultado["operacoes"])


def selecionar_politica(meta_validacao, probabilidades, horizonte, meta_acerto=.7, minimo=20,
                        permissao_compra=None, permissao_venda=None):
    p = np.asarray(probabilidades)
    permissoes = [np.ones(len(p), dtype=bool) if v is None else np.asarray(v, dtype=bool)
                  for v in [permissao_compra, permissao_venda]]
    if any(len(v) != len(p) for v in permissoes):
        raise ValueError("Máscaras de eventos incompatíveis com os escores.")
    quantis = [0., .25, .5, .65, .75, .85, .90, .95]
    limiares = [np.unique(np.quantile(p[mask, k], quantis)).tolist() if mask.any() else [None]
               for k, mask in zip([1, 2], permissoes)]
    tentativas, melhor = [], None
    for compra in limiares[0]:
        for venda in limiares[1]:
            politica = {"limiar_compra": float(compra) if compra is not None else None,
                        "limiar_venda": float(venda) if venda is not None else None}
            sinais = aplicar_politica(p, politica, *permissoes)
            m, _ = avaliar_operacoes(meta_validacao, sinais, horizonte, meta_acerto, minimo)
            tentativas.append({**politica, **m})
            if melhor is None or pontuar(m) > melhor[0]:
                melhor = pontuar(m), politica, m
    return melhor[1], melhor[2], pd.DataFrame(tentativas)
