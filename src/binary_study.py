"""Busca pré-definida de rótulos/horizontes sem consultar a confirmação."""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.binary_events import (FEATURES_EVENTOS, ajustar_modelo, criar_contexto,
    mascarar_eventos, permissoes, prever_scores, rotulos_binarios, salvar_binario)
from src.forecast_config import ConfigPrevisao
from src.forecast_data import limpar_candles
from src.forecast_experiment import salvar_json
from src.profit_experiment import _grafico_operacoes
from src.profit_labels import criar_alvos_lucro
from src.trade_policy import avaliar_operacoes, pontuar, selecionar_politica


def _partes(meta, plano):
    fronteiras = [pd.Timestamp(getattr(plano, c)) for c in ["fim_treino", "fim_early", "fim_ajuste", "fim_selecao", "fim_teste"]]
    partes, inicio = {}, None
    for nome, fim in zip(["treino", "early", "ajuste", "selecao", "teste"], fronteiras):
        mask = (meta.time < fim) & (meta.time_alvo < fim)
        if inicio is not None:
            mask &= meta.time >= inicio
        partes[nome] = np.flatnonzero(mask)
        inicio = fim
    if min(len(v) for v in partes.values()) < 30:
        raise ValueError("Histórico insuficiente para as fronteiras pré-definidas.")
    return partes


def _entradas(matriz_normalizada, posicoes, janela):
    if janela == 1:
        return matriz_normalizada[posicoes]
    idx = posicoes[:, None] - np.arange(janela-1, -1, -1)[None, :]
    return matriz_normalizada[idx]


def _preparar_horizonte(contexto, matriz, horizonte, plano):
    config = ConfigPrevisao(horizonte=horizonte)
    dados = criar_alvos_lucro(contexto, config, plano.custo)
    janelas_ok = pd.Series(np.isfinite(matriz).all(axis=1)).rolling(12).sum().eq(12).to_numpy()
    pos = np.flatnonzero(dados.alvo_elegivel.to_numpy() & janelas_ok)
    colunas = ["time", "time_alvo", "sessao", "retorno_liquido_compra", "retorno_liquido_venda"]
    meta = dados.iloc[pos][colunas].reset_index(drop=True)
    meta["posicao_bruta"] = pos
    return meta, _partes(meta, plano)


def executar_busca_binaria(csv, saida, plano):
    plano.validar()
    csv = Path(csv).resolve()
    bruto, qualidade = limpar_candles(pd.read_csv(csv))
    # A cauda 2024-07 a 2026 foi usada antes e não entra neste estudo.
    bruto = bruto.loc[bruto.time < pd.Timestamp(plano.fim_teste)].reset_index(drop=True)
    contexto = criar_contexto(bruto, ConfigPrevisao())
    matriz = contexto[list(FEATURES_EVENTOS)].to_numpy(dtype=np.float32)
    mask_fit = (contexto.time < pd.Timestamp(plano.fim_treino)).to_numpy() & np.isfinite(matriz).all(axis=1)
    if mask_fit.sum() < 100:
        raise ValueError("Poucos candles de treino antes da fronteira fixada.")
    scaler = StandardScaler().fit(matriz[mask_fit])
    normalizada = scaler.transform(matriz).astype(np.float32)
    pasta = Path(saida) / datetime.now(timezone.utc).strftime("estudo_%Y%m%dT%H%M%S_%fZ")
    pasta.mkdir(parents=True, exist_ok=False)
    for nome in ["historicos", "politicas", "codigo_snapshot"]:
        (pasta / nome).mkdir()
    for fonte in Path(__file__).parent.glob("*.py"):
        shutil.copy2(fonte, pasta / "codigo_snapshot" / fonte.name)
    salvar_json(pasta / "plano_congelado.json", {"plano": asdict(plano), "csv": str(csv),
        "sha256_csv": hashlib.sha256(csv.read_bytes()).hexdigest(), "qualidade_origem": qualidade,
        "dados_usados_inicio": bruto.time.iloc[0], "dados_usados_fim": bruto.time.iloc[-1],
        "features": list(FEATURES_EVENTOS),
        "regra": "Confirmar somente se a validação do modelo final atingir a meta em ambas as direções, com suporte mínimo; caso contrário não avaliar 2024."})
    registros, melhor, melhor_modelos = [], None, None
    for horizonte in plano.horizontes:
        meta, partes = _preparar_horizonte(contexto, matriz, horizonte, plano)
        meta_partes = {k: meta.iloc[idx].reset_index(drop=True) for k, idx in partes.items()}
        for banda in plano.bandas:
            vol = contexto.vol_anterior.iloc[meta.posicao_bruta].to_numpy()
            y = rotulos_binarios(meta, vol, horizonte, banda)
            for familia in plano.familias:
                janela = 12 if familia == "mlp_binario" else 1
                x = {k: _entradas(normalizada, m.posicao_bruta.to_numpy(), janela)
                     for k, m in meta_partes.items() if k != "teste"}
                nome = f"h{horizonte}_b{str(banda).replace('.', '_')}_{familia}"
                logging.info("Ajustando %s; sem avaliar a confirmação.", nome)
                modelos, hist = ajustar_modelo(familia, x["treino"], y[partes["treino"]],
                                              x["early"], y[partes["early"]], plano)
                for h in hist:
                    pd.DataFrame(h["history"]).to_csv(pasta / "historicos" / f"{nome}_{h['seed']}.csv", index=False)
                pa = prever_scores(familia, modelos, x["ajuste"])
                ps = prever_scores(familia, modelos, x["selecao"])
                for evento in plano.eventos:
                    mask_a = permissoes(contexto.iloc[meta_partes["ajuste"].posicao_bruta], evento)
                    mask_s = permissoes(contexto.iloc[meta_partes["selecao"].posicao_bruta], evento)
                    politica, ma, tabela = selecionar_politica(meta_partes["ajuste"], pa, horizonte,
                        plano.meta, plano.minimo, mask_a[0], mask_a[1])
                    sinais = mascarar_eventos(ps, politica, mask_s)
                    ms, _ = avaliar_operacoes(meta_partes["selecao"], sinais, horizonte, plano.meta, plano.minimo)
                    registro = {"candidato": nome+"_"+evento, "horizonte": horizonte, "banda": banda,
                                "familia": familia, "janela": janela, "evento": evento, **ms}
                    registros.append(registro)
                    tabela.to_csv(pasta / "politicas" / f"{nome}_{evento}.csv", index=False)
                    if melhor is None or pontuar(ms) > pontuar(melhor["metricas"]):
                        melhor = {"registro": registro, "metricas": ms, "politica": politica}
                        melhor_modelos = list(modelos)
                    logging.info("%s/%s: compra %.3f (%d), venda %.3f (%d)", nome, evento,
                        ms["taxa_lucro_compra"] or 0., ms["operacoes_compra"], ms["taxa_lucro_venda"] or 0., ms["operacoes_venda"])
                pd.DataFrame(registros).to_csv(pasta / "comparacao_validacao.csv", index=False)
    salvar_json(pasta / "escolha_antes_da_confirmacao.json", melhor)
    candidato = melhor["registro"]
    salvar_binario(pasta / "modelo_experimental", melhor_modelos, scaler,
        {**candidato, "politica": melhor["politica"], "meta_confirmada": False}, plano)
    status = {"meta_na_validacao_busca": bool(melhor["metricas"]["meta_70_lucro_ambas"]),
              "confirmacao_avaliada": False, "meta_confirmada": False}
    # Não gastar uma confirmação nova com um candidato já reprovado na validação.
    if status["meta_na_validacao_busca"]:
        h, banda, familia = candidato["horizonte"], candidato["banda"], candidato["familia"]
        meta, partes = _preparar_horizonte(contexto, matriz, h, plano)
        mp = {k: meta.iloc[idx].reset_index(drop=True) for k, idx in partes.items()}
        vol = contexto.vol_anterior.iloc[meta.posicao_bruta].to_numpy()
        y = rotulos_binarios(meta, vol, h, banda)
        x = {k: _entradas(normalizada, m.posicao_bruta.to_numpy(), candidato["janela"])
             for k, m in mp.items() if k != "teste"}
        modelos, hist = ajustar_modelo(familia, x["treino"], y[partes["treino"]], x["early"], y[partes["early"]], plano, plano.sementes_finais)
        pa = prever_scores(familia, modelos, x["ajuste"])
        masks_a = permissoes(contexto.iloc[mp["ajuste"].posicao_bruta], candidato["evento"])
        politica, _, _ = selecionar_politica(mp["ajuste"], pa, h, plano.meta, plano.minimo, *masks_a)
        ps = prever_scores(familia, modelos, x["selecao"])
        masks_s = permissoes(contexto.iloc[mp["selecao"].posicao_bruta], candidato["evento"])
        ms, _ = avaliar_operacoes(mp["selecao"], mascarar_eventos(ps, politica, masks_s), h, plano.meta, plano.minimo)
        salvar_json(pasta / "validacao_modelo_final.json", ms)
        info = {**candidato, "politica": politica, "meta_confirmada": False}
        if ms["meta_70_lucro_ambas"]:
            xt = _entradas(normalizada, mp["teste"].posicao_bruta.to_numpy(), candidato["janela"])
            pt = prever_scores(familia, modelos, xt)
            mask_t = permissoes(contexto.iloc[mp["teste"].posicao_bruta], candidato["evento"])
            sinais = mascarar_eventos(pt, politica, mask_t)
            mt, usados = avaliar_operacoes(mp["teste"], sinais, h, plano.meta, plano.minimo)
            salvar_json(pasta / "resultado_confirmacao.json", mt)
            pred = mp["teste"].copy()
            pred["sinal"] = sinais
            pred["operacao_executada"] = False
            pred.loc[usados, "operacao_executada"] = True
            pred["score_compra"], pred["score_venda"] = pt[:, 1], pt[:, 2]
            pred.to_csv(pasta / "predicoes_confirmacao.csv", index=False)
            _grafico_operacoes(mt, pasta / "confirmacao.png", "Confirmação reservada: janeiro a junho/2024")
            status.update({"confirmacao_avaliada": True, "meta_confirmada": bool(mt["meta_70_lucro_ambas"])})
            info["meta_confirmada"] = status["meta_confirmada"]
        salvar_binario(pasta / "artefato", modelos, scaler, info, plano)
    salvar_json(pasta / "status.json", status)
    _relatorio(pasta, plano, melhor, status, registros)
    salvar_json(pasta / "CONCLUIDO.json", {"concluido": True, **status})
    return pasta, pd.DataFrame(registros), status


def _relatorio(pasta, plano, melhor, status, registros):
    ordenar = sorted(registros, key=pontuar, reverse=True)
    pct = lambda x: "NA" if x is None else f"{x:.2%}"
    texto = ["# Busca de rótulos binários e horizontes", "",
        f"Meta pré-definida: {plano.meta:.0%} de operações líquidas lucrativas em compra e venda, com pelo menos {plano.minimo} operações de cada direção. Taxa fixa de {plano.custo*10000:g} bps por lado; sem spread e slippage.", "",
        f"Foram avaliados {len(registros)} candidatos na validação. Horizontes: {plano.horizontes} candles. Rótulos binários separados por direção, com bandas de clareza {plano.bandas} vezes a volatilidade passada escalada por raiz do horizonte.", "",
        "Na banda zero, toda operação tem rótulo de lucro ou não lucro. Nas bandas positivas, movimentos pequenos são desconhecidos apenas para o treinamento/Early Stopping; todos os sinais emitidos continuam sendo avaliados financeiramente, inclusive os que perdem.", "",
        "Os eventos são condições causais conhecidas ao emitir a previsão: todos os candles, choque de preço para reversão, RSI extremo ou rompimento de extremos passados. Cobertura, quantidade e resultado ficam registrados. Não há filtro baseado no lucro futuro na validação ou confirmação.", "",
        f"Treino até {plano.fim_treino}; Early Stopping até {plano.fim_early}; ajuste dos limiares até {plano.fim_ajuste}; seleção até {plano.fim_selecao}. O período de confirmação reservado termina em {plano.fim_teste}.", "",
        "## Maiores pontuações na validação", "",
        "| Horizonte | Banda | Modelo | Evento | Compras | % compra | Vendas | % venda | Retorno |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|"]
    for r in ordenar[:20]:
        texto.append(f"| {r['horizonte']} | {r['banda']} | {r['familia']} | {r['evento']} | {r['operacoes_compra']} | {pct(r['taxa_lucro_compra'])} | {r['operacoes_venda']} | {pct(r['taxa_lucro_venda'])} | {pct(r['retorno_acumulado'])} |")
    texto += ["", f"Meta atingida na validação da busca: **{status['meta_na_validacao_busca']}**.",
        f"Confirmação reservada avaliada: **{status['confirmacao_avaliada']}**.",
        f"Meta confirmada: **{status['meta_confirmada']}**.", "",
        "Se a validação não atingir a meta com suporte suficiente, a confirmação não é usada para continuar procurando uma taxa alta. Sua ausência não representa aprovação; preserva o período para outra hipótese.", "",
        "Os dois escores binários não precisam somar um e não são uma garantia de acerto. A aprovação depende de operações observadas. Uma taxa alta com poucas operações continua incerta, e taxa de vitórias não substitui retorno e drawdown.", "",
        "O arquivo de origem é um snapshot fornecido pelo MT5 em novembro/2026. Há revisão de preços no histórico do provedor, cuja causa não foi verificada. Os dados e os hashes foram preservados. Resultados históricos não estabelecem comportamento futuro garantido.", "",
        "Arquivos: `plano_congelado.json`, `comparacao_validacao.csv`, `politicas/`, `historicos/`, `escolha_antes_da_confirmacao.json`, `modelo_experimental/` e `status.json`. O modelo experimental não está aprovado pela meta. Resultados da confirmação e o artefato final só aparecem quando aplicável."]
    (pasta / "LEIA_RESULTADOS.md").write_text("\n".join(texto)+"\n", encoding="utf-8")
