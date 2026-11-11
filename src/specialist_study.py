"""Seleção independente de especialistas por direção, com confirmação fechada."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.binary_events import (FEATURES_EVENTOS, ajustar_modelo, carregar_prever_binario,
    criar_contexto, mascarar_eventos, permissoes, prever_scores, rotulos_binarios, salvar_binario)
from src.binary_study import _entradas, _preparar_horizonte
from src.forecast_config import ConfigPrevisao
from src.forecast_data import limpar_candles
from src.forecast_experiment import salvar_json
from src.trade_policy import avaliar_operacoes, pontuar


def selecionar_limiar_lado(meta, p, permitido, lado, horizonte, minimo=20):
    permitido = np.asarray(permitido, dtype=bool)
    if not permitido.any():
        return None
    melhor = None
    nome = "compra" if lado == 1 else "venda"
    for limiar in np.unique(np.quantile(p[permitido, lado], [0, .25, .5, .65, .75, .85, .9, .95])):
        sinais = np.where(permitido & (p[:, lado] >= limiar), lado, 0)
        m, _ = avaliar_operacoes(meta, sinais, horizonte, minimo=minimo)
        n = m[f"operacoes_{nome}"]
        score = (m[f"taxa_lucro_{nome}"] if n >= minimo else -1., n)
        if melhor is None or score > melhor[0]:
            melhor = score, float(limiar)
    return melhor[1]


def avaliar_par(compra, venda, etapa, minimo=20, meta_acerto=.7):
    ca, ve = compra[etapa], venda[etapa]
    c = ca[["time", "posicao_bruta", "retorno_liquido_compra", "score", "permitido"]].rename(columns={"score": "score_compra", "permitido": "perm_compra"})
    v = ve[["time", "posicao_bruta", "retorno_liquido_venda", "score", "permitido"]].rename(columns={"score": "score_venda", "permitido": "perm_venda"})
    uniao = c.merge(v, on=["time", "posicao_bruta"], how="outer", validate="one_to_one").sort_values("posicao_bruta").reset_index(drop=True)
    mask_c = uniao.perm_compra.eq(True).to_numpy()
    mask_v = uniao.perm_venda.eq(True).to_numpy()
    p = np.column_stack([np.zeros(len(uniao)), uniao.score_compra.fillna(0), uniao.score_venda.fillna(0)])
    pol = {"limiar_compra": compra["limiar"], "limiar_venda": venda["limiar"]}
    sinais = mascarar_eventos(p, pol, (mask_c, mask_v))
    m, usados = avaliar_operacoes(uniao, sinais, compra["horizonte"], meta_acerto, minimo,
                                  horizonte_venda=venda["horizonte"])
    uniao["sinal"] = sinais
    uniao["operacao_executada"] = False
    uniao.loc[usados, "operacao_executada"] = True
    return m, uniao


def executar_especialistas(csv, saida, plano):
    plano.validar()
    csv = Path(csv)
    bruto, _ = limpar_candles(pd.read_csv(csv))
    bruto = bruto.loc[bruto.time < pd.Timestamp(plano.fim_teste)].reset_index(drop=True)
    ctx = criar_contexto(bruto, ConfigPrevisao())
    matriz = ctx[list(FEATURES_EVENTOS)].to_numpy(dtype=np.float32)
    fit = (ctx.time < pd.Timestamp(plano.fim_treino)).to_numpy() & np.isfinite(matriz).all(axis=1)
    scaler = StandardScaler().fit(matriz[fit])
    normalizada = scaler.transform(matriz).astype(np.float32)
    pasta = Path(saida) / datetime.now(timezone.utc).strftime("especialistas_%Y%m%dT%H%M%S_%fZ")
    pasta.mkdir(parents=True, exist_ok=False)
    salvar_json(pasta / "plano_congelado.json", {"plano": asdict(plano),
        "sha256_csv": hashlib.sha256(csv.read_bytes()).hexdigest(),
        "regra": "Limiar por direção no ajuste; ranking dos especialistas e até nove pares na seleção; confirmação só se ambos passarem 70% e suporte mínimo.",
        "horizonte": "Pode ser diferente em cada direção, declarado no artefato; execução conjunta não permite posições sobrepostas."})
    linhas, qualificados = [], {1: [], 2: []}
    for h in plano.horizontes:
        meta, partes = _preparar_horizonte(ctx, matriz, h, plano)
        mp = {k: meta.iloc[v].reset_index(drop=True) for k, v in partes.items()}
        vol = ctx.vol_anterior.iloc[meta.posicao_bruta].to_numpy()
        for banda in plano.bandas:
            y = rotulos_binarios(meta, vol, h, banda)
            for familia in plano.familias:
                janela = 12 if familia == "mlp_binario" else 1
                x = {k: _entradas(normalizada, m.posicao_bruta.to_numpy(), janela) for k, m in mp.items() if k != "teste"}
                modelos, hist = ajustar_modelo(familia, x["treino"], y[partes["treino"]], x["early"], y[partes["early"]], plano)
                pa, ps = prever_scores(familia, modelos, x["ajuste"]), prever_scores(familia, modelos, x["selecao"])
                for evento in plano.eventos:
                    ma = permissoes(ctx.iloc[mp["ajuste"].posicao_bruta], evento)
                    ms = permissoes(ctx.iloc[mp["selecao"].posicao_bruta], evento)
                    for lado in [1, 2]:
                        nome = "compra" if lado == 1 else "venda"
                        corte = selecionar_limiar_lado(mp["ajuste"], pa, ma[lado-1], lado, h, plano.minimo)
                        sinais = np.where(ms[lado-1] & (ps[:, lado] >= corte), lado, 0) if corte is not None else np.zeros(len(ps), dtype=int)
                        m, _ = avaliar_operacoes(mp["selecao"], sinais, h, plano.meta, plano.minimo)
                        n, taxa = m[f"operacoes_{nome}"], m[f"taxa_lucro_{nome}"]
                        reg = {"lado": nome, "horizonte": h, "banda": banda, "familia": familia,
                            "evento": evento, "limiar": corte, "operacoes": n, "taxa_lucro": taxa,
                            "qualificado": bool(n >= plano.minimo and taxa is not None and taxa >= plano.meta)}
                        linhas.append(reg)
                        if reg["qualificado"]:
                            q = {**reg, "janela": janela, "modelos": list(modelos), "meta": meta,
                                 "partes": partes, "score_lado": lado}
                            q["selecao"] = mp["selecao"].copy()
                            q["selecao"]["score"] = ps[:, lado]
                            q["selecao"]["permitido"] = ms[lado-1]
                            qualificados[lado].append(q)
                pd.DataFrame(linhas).to_csv(pasta / "validacao_por_direcao.csv", index=False)
                logging.info("Especialistas h=%d banda=%.2f %s: qualificados compra=%d, venda=%d", h, banda, familia,
                             len(qualificados[1]), len(qualificados[2]))
    tops = {lado: sorted(valores, key=lambda q: (q["taxa_lucro"], q["operacoes"]), reverse=True)[:3] for lado, valores in qualificados.items()}
    pares, melhor = [], None
    for compra in tops[1]:
        for venda in tops[2]:
            m, pred = avaliar_par(compra, venda, "selecao", plano.minimo, plano.meta)
            pares.append({"compra": {k: compra[k] for k in ["horizonte", "banda", "familia", "evento", "limiar"]},
                          "venda": {k: venda[k] for k in ["horizonte", "banda", "familia", "evento", "limiar"]}, **m})
            if melhor is None or pontuar(m) > pontuar(melhor[0]):
                melhor = m, compra, venda
    salvar_json(pasta / "pares_validacao.json", pares)
    status = {"qualificados_compra": len(qualificados[1]), "qualificados_venda": len(qualificados[2]),
              "meta_conjunta_validacao": bool(melhor is not None and melhor[0]["meta_70_lucro_ambas"]),
              "confirmacao_avaliada": False, "meta_confirmada": False}
    if status["meta_conjunta_validacao"]:
        m, compra, venda = melhor
        salvar_json(pasta / "par_antes_da_confirmacao.json", pares[0] if len(pares) == 1 else {
            "compra": {k: compra[k] for k in ["horizonte", "banda", "familia", "evento", "limiar"]},
            "venda": {k: venda[k] for k in ["horizonte", "banda", "familia", "evento", "limiar"]}, "validacao": m})
        for q in [compra, venda]:
            teste = q["meta"].iloc[q["partes"]["teste"]].reset_index(drop=True)
            xt = _entradas(normalizada, teste.posicao_bruta.to_numpy(), q["janela"])
            pt = prever_scores(q["familia"], q["modelos"], xt)
            q["teste"] = teste
            q["teste"]["score"] = pt[:, q["score_lado"]]
            q["teste"]["permitido"] = permissoes(ctx.iloc[teste.posicao_bruta], q["evento"])[q["score_lado"]-1]
        mt, pred = avaliar_par(compra, venda, "teste", plano.minimo, plano.meta)
        pred.to_csv(pasta / "predicoes_confirmacao.csv", index=False)
        salvar_json(pasta / "resultado_confirmacao.json", mt)
        status.update({"confirmacao_avaliada": True, "meta_confirmada": bool(mt["meta_70_lucro_ambas"])})
        for lado, q in [("compra", compra), ("venda", venda)]:
            politica = {"limiar_compra": q["limiar"] if lado == "compra" else None,
                        "limiar_venda": q["limiar"] if lado == "venda" else None}
            salvar_binario(pasta / "artefato" / lado, q["modelos"], scaler,
                {k: q[k] for k in ["familia", "janela", "horizonte", "banda", "evento"]} |
                {"politica": politica, "meta_confirmada": status["meta_confirmada"]}, plano)
        salvar_json(pasta / "artefato/modelo_especialistas.json", {"tipo_modelo": "especialistas_eventos", **status})
    salvar_json(pasta / "status.json", status)
    tabela = pd.DataFrame(linhas)
    texto = ["# Especialistas independentes por direção", "",
        "Cada lado escolhe seu próprio rótulo, horizonte, modelo, evento e limiar. O lucro é líquido de 2 bps por lado, com uma posição por vez. Os horizontes podem diferir e isso é outro experimento.", "",
        "Os limiares são escolhidos no bloco de ajuste. Os especialistas são classificados na seleção. Até três qualificados por direção podem formar nove pares, novamente avaliados sem sobreposição. Só um par que alcance a meta conjunta abre a confirmação reservada.", "",
        "## Maiores taxas na validação por direção", "",
        "| Lado | Horizonte | Banda | Modelo | Evento | Operações | Taxa de lucro |",
        "|---|---:|---:|---|---|---:|---:|"]
    for lado in ["compra", "venda"]:
        top = tabela.loc[(tabela.lado == lado) & (tabela.operacoes >= plano.minimo)].sort_values(["taxa_lucro", "operacoes"], ascending=False).head(5)
        for r in top.itertuples():
            texto.append(f"| {lado} | {r.horizonte} | {r.banda} | {r.familia} | {r.evento} | {r.operacoes} | {r.taxa_lucro:.2%} |")
    texto += ["", f"Estado final: `{status}`.", "",
        "Uma taxa isolada de 70% não comprova a estratégia conjunta. Escolher muitos candidatos pode produzir otimismo na validação; a confirmação não é reaberta para procurar outra combinação. Modelos sintéticos dos testes automatizados não são incorporados ao histórico real."]
    (pasta / "LEIA_RESULTADOS.md").write_text("\n".join(texto)+"\n", encoding="utf-8")
    return pasta, tabela, status


def carregar_prever_especialistas(dados, pasta):
    pasta = Path(pasta)
    info = json.loads((pasta / "modelo_especialistas.json").read_text(encoding="utf-8"))
    compra = carregar_prever_binario(dados, pasta / "compra")
    venda = carregar_prever_binario(dados, pasta / "venda")
    sinais = [compra["classe_predita"] == 1, venda["classe_predita"] == 2]
    if all(sinais):
        ic = json.loads((pasta / "compra/modelo_binario.json").read_text())
        iv = json.loads((pasta / "venda/modelo_binario.json").read_text())
        c, v = ic["politica"]["limiar_compra"], iv["politica"]["limiar_venda"]
        escolher_compra = (compra["score_compra"]-c)/max(1-c, 1e-12) >= (venda["score_venda"]-v)/max(1-v, 1e-12)
        sinal = "compra" if escolher_compra else "venda"
    else:
        sinal = "compra" if sinais[0] else "venda" if sinais[1] else "manter"
    return {"tipo_modelo": "especialistas_eventos", "sinal": sinal,
            "compra": compra, "venda": venda, "meta_confirmada": info["meta_confirmada"]}
