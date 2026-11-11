"""Compara arquiteturas no desenvolvimento e avalia uma rede escolhida depois."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.forecast_backtest import simular
from src.forecast_config import ConfigPrevisao
from src.forecast_data import (criar_alvos, criar_atributos, extrair_entradas, indices_do_corte,
                              limpar_candles, montar_amostras, planejar_cortes)
from src.forecast_evaluation import avaliar, bootstrap_por_sessao, selecionar_modelo
from src.forecast_experiment import salvar_json, tabela_predicoes, _matriz
from src.neural_search import candidatos, treinar_candidato
from src.prediction import prever_ultima_janela, salvar_modelo


def executar(csv, referencia, saida, epocas=30, nomes=None, criterio="f1_macro"):
    config = ConfigPrevisao(epocas=epocas, paciencia=6)
    config.validar()
    specs = [s for s in candidatos() if s.criterio == criterio and (nomes is None or s.nome in nomes)]
    if not specs:
        raise ValueError("Nenhum candidato solicitado existe.")
    dados_originais, qualidade = limpar_candles(pd.read_csv(csv))
    dados = criar_alvos(criar_atributos(dados_originais, config), config)
    dados["retorno_passado"] = dados.close.pct_change(config.horizonte, fill_method=None)
    amostras = montar_amostras(dados, config)
    amostras.metadados["retorno_passado"] = dados.retorno_passado.iloc[amostras.finais_janelas].to_numpy()
    cortes = planejar_cortes(dados, config)
    base_info = json.loads((referencia / "manifesto.json").read_text(encoding="utf-8"))
    if base_info["sha256_csv"] != hashlib.sha256(Path(csv).read_bytes()).hexdigest():
        raise ValueError("A referência usa outra base de dados.")
    for chave in ["horizonte", "limiar", "janela", "folds", "minutos_candle", "fracao_teste_final", "fracao_bloco_desenvolvimento"]:
        if base_info["config"][chave] != asdict(config)[chave]:
            raise ValueError(f"A configuração de comparação difere em {chave}.")
    pasta = saida / datetime.now(timezone.utc).strftime("execucao_%Y%m%dT%H%M%S_%fZ")
    pasta.mkdir(parents=True, exist_ok=False)
    for n in ["historicos", "predicoes", "modelos", "graficos", "backtest", "codigo_snapshot"]:
        (pasta / n).mkdir()
    fontes = list((RAIZ / "src").glob("*.py")) + [Path(__file__)]
    for f in fontes:
        shutil.copy2(f, pasta / "codigo_snapshot" / f.name)
    salvar_json(pasta / "manifesto_busca.json", {"config": asdict(config), "candidatos": [asdict(s) for s in specs],
                "referencia": str(referencia), "sha256_csv": base_info["sha256_csv"], "qualidade": qualidade,
                "criterio": criterio + "_no_desenvolvimento; nunca selecionar por teste_final",
                "semente_busca": 42, "sementes_confirmacao": list(config.sementes),
                "limitacao": "O teste final já foi examinado anteriormente; avaliação retrospectiva, sem confirmação em um novo período."})
    cache_cortes = []
    for corte in cortes:
        partes = indices_do_corte(amostras.metadados, corte)
        mascara = (dados.time < corte.inicio_validacao).to_numpy() & np.isfinite(amostras.features_linhas).all(axis=1)
        scaler = StandardScaler().fit(amostras.features_linhas[mascara])
        metas = {k: amostras.metadados.iloc[v].reset_index(drop=True) for k, v in partes.items()}
        cache_cortes.append((corte, partes, scaler, metas))
    resultados = []
    for spec in specs:
        c = deepcopy(config)
        c.janela = spec.janela
        c.sementes = (42,)
        for corte, partes, scaler, metas in cache_cortes[:-1]:
            x = {k: extrair_entradas(amostras, v, scaler, c, True) for k, v in partes.items()}
            mt, mv, me = (metas[k] for k in ["treino", "validacao", "teste"])
            rede = treinar_candidato(spec, x["treino"], mt.rotulo.to_numpy(), x["validacao"], mv.rotulo.to_numpy(), c)
            out = rede.prever(x["teste"], me.retorno_passado.to_numpy())
            m, _ = avaliar(me.rotulo, out["classes"], c, out["probabilidades"])
            resultados.append({"modelo": spec.nome, "fold": corte.nome, **m})
            tabela_predicoes(me, out, c).to_csv(pasta / "predicoes" / f"{spec.nome}_{corte.nome}.csv", index=False)
            pd.DataFrame(rede.historicos[0]["history"]).to_csv(pasta / "historicos" / f"{spec.nome}_{corte.nome}.csv", index=False)
            pd.DataFrame(resultados).to_csv(pasta / "tentativas_desenvolvimento.csv", index=False)
            logging.info("%s / %s: accuracy %.3f; F1 %.3f; recall macro %.3f", spec.nome, corte.nome, m["accuracy"], m["f1_macro"], m["recall_macro"])
    if criterio == "diagonal":
        ranking = pd.DataFrame(resultados).groupby("modelo")[["recall_manter", "recall_compra", "recall_venda", "f1_macro"]].mean()
        ranking["pior_recall_medio"] = ranking[["recall_manter", "recall_compra", "recall_venda"]].min(axis=1)
        ranking = ranking.sort_values(["pior_recall_medio", "f1_macro"], ascending=False).reset_index()
        escolhido = ranking.modelo.iloc[0]
    else:
        escolhido, ranking = selecionar_modelo(pd.DataFrame(resultados))
    ranking.to_csv(pasta / "ranking_desenvolvimento.csv", index=False)
    spec = next(s for s in specs if s.nome == escolhido)
    salvar_json(pasta / "escolha_antes_do_teste.json", {"modelo": escolhido, "especificacao": asdict(spec), "criterio": criterio})
    c = deepcopy(config)
    c.janela = spec.janela
    c.modelos = ("majoritaria", escolhido)
    corte, partes, scaler, metas = cache_cortes[-1]
    x = {k: extrair_entradas(amostras, v, scaler, c, True) for k, v in partes.items()}
    mt, mv, me = (metas[k] for k in ["treino", "validacao", "teste"])
    rede = treinar_candidato(spec, x["treino"], mt.rotulo.to_numpy(), x["validacao"], mv.rotulo.to_numpy(), c)
    out = rede.prever(x["teste"], me.retorno_passado.to_numpy())
    m, _ = avaliar(me.rotulo, out["classes"], c, out["probabilidades"])
    pred = tabela_predicoes(me, out, c)
    pred.to_csv(pasta / "predicoes" / "teste_final_rede_ajustada.csv", index=False)
    salvar_modelo(rede, scaler, pasta / "modelos" / escolhido)
    salvar_json(pasta / "manifesto.json", {"config": asdict(c), "sha256_csv": base_info["sha256_csv"], "referencia": str(referencia)})
    salvar_json(pasta / "selecao_modelo.json", {"modelo_selecionado": escolhido, "origem": "busca_no_desenvolvimento"})
    salvar_json(pasta / "previsao_ultima_janela.json", prever_ultima_janela(dados_originais, rede, scaler, c))
    _matriz(me.rotulo, out["classes"], pasta / "graficos" / "matriz_rede_ajustada.png", f"Rede ajustada: {escolhido}")
    trades, bt = simular(dados, pred, c)
    trades.to_csv(pasta / "backtest" / "operacoes.csv", index=False)
    bt.to_csv(pasta / "backtest" / "metricas.csv", index=False)
    salvar_json(pasta / "incerteza_rede_ajustada.json", bootstrap_por_sessao(pred, c.bootstrap_sessoes))
    por_semente = []
    for i, h in enumerate(rede.historicos):
        pd.DataFrame(h["history"]).to_csv(pasta / "historicos" / f"final_seed_{h['seed']}.csv", index=False)
        valor = out["individuais"][i]
        if rede.ajuste_decisao is not None:
            valor = valor * rede.ajuste_decisao
        ms, _ = avaliar(me.rotulo, valor.argmax(axis=1), c)
        por_semente.append({"seed": h["seed"], **ms})
    pd.DataFrame(por_semente).to_csv(pasta / "teste_por_semente.csv", index=False)
    tabela = pd.read_csv(referencia / "metricas_teste_final.csv")
    tabela["origem"] = "previsao_sem_ruido_inicial"
    antes = pd.read_csv(RAIZ / "resultados/predicoes/predicoes_original.csv", parse_dates=["time"])
    antes = pred[["time", "classe_real"]].merge(antes, on="time", suffixes=("", "_antiga"), validate="one_to_one")
    linhas_adicionais = [{"modelo": escolhido, "origem": "rede_ajustada", **m}]
    if len(antes) == len(pred) and np.array_equal(antes.classe_real, antes.classe_real_antiga):
        antigo, _ = avaliar(antes.classe_real, antes.classe_predita, c, antes[["prob_manter", "prob_compra", "prob_venda"]].to_numpy())
        linhas_adicionais.insert(0, {"modelo": "LSTM_legada_nas_mesmas_datas", "origem": "modelo_antigo_arquivado", **antigo})
    else:
        salvar_json(pasta / "comparacao_legada_indisponivel.json", {
            "motivo": "O teste usa datas ou rótulos ausentes nas previsões antigas. Os resultados não são uma comparação pareada com a rede legada.",
            "datas_em_comum": len(antes), "amostras_teste_novo": len(pred)})
    tabela = pd.concat([tabela, pd.DataFrame(linhas_adicionais)], ignore_index=True)
    tabela.to_csv(pasta / "comparacao_antes_depois.csv", index=False)
    salvar_json(pasta / "CONCLUIDO.json", {"concluido": True, "meta_diagonal": m["meta_70_diagonal_todas_classes"], "modelo": escolhido})
    logging.info("Busca concluída: %s", pasta)
    print(tabela[["modelo", "accuracy", "f1_macro", "recall_manter", "recall_compra", "recall_venda"]].to_string(index=False))
    return pasta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=RAIZ / "dados_processados/dados_originais.csv")
    parser.add_argument("--referencia", type=Path, required=True)
    parser.add_argument("--saida", type=Path, default=RAIZ / "ajustes_redes")
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--candidatos", help="Subconjunto de nomes separados por vírgula.")
    parser.add_argument("--criterio", choices=["f1_macro", "diagonal"], default="f1_macro")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    executar(args.csv, args.referencia, args.saida, args.epocas, set(args.candidatos.split(",")) if args.candidatos else None, args.criterio)
