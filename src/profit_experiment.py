"""Previsão de operações lucrativas, com emissão seletiva e avaliação líquida."""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.forecast_backtest import simular
from src.forecast_data import (criar_atributos, extrair_entradas, indices_do_corte,
                              limpar_candles, montar_amostras, planejar_cortes)
from src.forecast_experiment import salvar_json
from src.forecast_models import ModeloPrevisao
from src.neural_search import EspecificacaoRede, treinar_candidato
from src.prediction import prever_ultima_janela, salvar_modelo
from src.profit_labels import criar_alvos_lucro
from src.label_variants import rotulos_alternativos
from src.trade_policy import aplicar_politica, avaliar_operacoes, pontuar, selecionar_politica


MODELOS = ("logistica", "hist_gradient", "extra_trees", "mlp_lucro", "gru_lucro")


def treinar_lucro(nome, entradas, metas, config):
    c = deepcopy(config)
    if nome in {"mlp_lucro", "gru_lucro"}:
        c.janela = 12 if nome == "mlp_lucro" else 24
        spec = EspecificacaoRede(nome, "mlp" if nome == "mlp_lucro" else "gru", c.janela,
                                 potencia_peso=0, criterio="log_loss")
        return treinar_candidato(spec, entradas["treino"], metas["treino"].rotulo.to_numpy(),
                                 entradas["early_stopping"], metas["early_stopping"].rotulo.to_numpy(), c)
    if nome == "logistica":
        estimador = LogisticRegression(C=.1, max_iter=1500)
    elif nome == "hist_gradient":
        estimador = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, min_samples_leaf=50,
                    l2_regularization=5., early_stopping=False, random_state=config.sementes[0])
    elif nome == "extra_trees":
        estimador = ExtraTreesClassifier(n_estimators=250, max_depth=8, min_samples_leaf=50,
                                         max_features=.8, n_jobs=4, random_state=config.sementes[0])
    else:
        raise ValueError(f"Modelo de lucro desconhecido: {nome}")
    estimador.fit(entradas["treino"], metas["treino"].rotulo.to_numpy())
    return ModeloPrevisao(nome, c, estimadores=[estimador], parametros=estimador.get_params())


def _salvar_predicoes(meta, p, sinais, destino):
    d = meta.copy()
    d["classe_predita"] = sinais
    d["score_manter"] = p[:, 0]
    d["score_compra"] = p[:, 1]
    d["score_venda"] = p[:, 2]
    d.to_csv(destino, index=False)
    return d


def _grafico_operacoes(metricas, destino, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    valores = [metricas[f"taxa_lucro_{n}"] for n in ["compra", "venda"]]
    ax.bar(["Compra", "Venda"], [v if v is not None else 0 for v in valores], color=["#167c64", "#386bb0"])
    ax.axhline(.7, color="#ba3434", linestyle="--", label="Meta: 70%")
    for i, n in enumerate(["compra", "venda"]):
        taxa = valores[i]
        label = "Sem operações" if taxa is None else f"{taxa:.1%}\n{metricas['lucrativas_'+n]}/{metricas['operacoes_'+n]} lucrativas"
        ax.text(i, (taxa or 0) + .025, label, ha="center")
    ax.set(ylim=(0, 1.15), ylabel="Proporção de operações com lucro líquido", title=titulo)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(destino, dpi=160)
    plt.close(fig)


def executar_lucro(csv, pasta_saida, config, custo_por_lado=.0002, modelos=MODELOS,
                  rotulagem="lucro_liquido", apenas_validacao=False):
    config.validar()
    config.custos_por_lado = tuple(sorted(set(config.custos_por_lado) | {custo_por_lado}))
    if not modelos or not set(modelos) <= set(MODELOS):
        raise ValueError("Modelos de lucro inválidos.")
    csv = Path(csv).resolve()
    r = pd.read_csv(csv)
    r.columns = [str(x).strip().lower() for x in r.columns]
    bruto, qualidade = limpar_candles(r)
    dados = criar_alvos_lucro(criar_atributos(bruto, config), config, custo_por_lado)
    dados["rotulo_auxiliar"] = rotulos_alternativos(dados, config, rotulagem, custo_por_lado)
    amostras = montar_amostras(dados, config)
    finais = amostras.finais_janelas
    for coluna in ["retorno_liquido_compra", "retorno_liquido_venda", "preco_entrada_futuro", "preco_saida_futuro"]:
        amostras.metadados[coluna] = dados[coluna].iloc[finais].to_numpy()
    amostras.metadados["rotulo_auxiliar"] = dados.rotulo_auxiliar.iloc[finais].to_numpy()
    corte = planejar_cortes(dados, config)[-1]
    indices = indices_do_corte(amostras.metadados, corte)
    sessoes = amostras.metadados.iloc[indices["validacao"]].sessao.unique()
    if len(sessoes) < 6:
        raise ValueError("São necessárias ao menos seis sessões de validação para separar treino e decisão.")
    grupos = np.array_split(sessoes, 3)
    for nome, dias in zip(["early_stopping", "ajuste_politica", "validacao_selecao"], grupos):
        indices[nome] = indices["validacao"][amostras.metadados.iloc[indices["validacao"]].sessao.isin(dias).to_numpy()]
    indices.pop("validacao")
    metas = {k: amostras.metadados.iloc[v].reset_index(drop=True) for k, v in indices.items()}
    if min(map(len, metas.values())) < 30:
        raise ValueError("Partição com menos de 30 exemplos.")
    linhas = amostras.features_linhas
    mascara = (dados.time < corte.inicio_validacao).to_numpy() & np.isfinite(linhas).all(axis=1)
    scaler = StandardScaler().fit(linhas[mascara])
    pasta = Path(pasta_saida) / datetime.now(timezone.utc).strftime("execucao_%Y%m%dT%H%M%S_%fZ")
    pasta.mkdir(parents=True, exist_ok=False)
    for n in ["modelos", "predicoes", "historicos", "politicas", "graficos", "backtest", "codigo_snapshot"]:
        (pasta / n).mkdir()
    for fonte in Path(__file__).parent.glob("*.py"):
        shutil.copy2(fonte, pasta / "codigo_snapshot" / fonte.name)
    salvar_json(pasta / "manifesto.json", {"config": asdict(config), "csv": str(csv),
        "sha256_csv": hashlib.sha256(csv.read_bytes()).hexdigest(), "tipo_alvo": "lucro_liquido",
        "custo_por_lado": custo_por_lado, "modelos": list(modelos),
        "rotulagem_treinamento": rotulagem, "apenas_validacao": apenas_validacao,
        "campos_legados_ignorados": ["limiar", "folds"],
        "protocolo": "Treino; early stopping; ajuste da política; seleção da estratégia; teste posterior. Datas por sessão.",
        "limitacao": "O período posterior já foi examinado em outros objetivos; estudo retrospectivo do novo rótulo, sem confirmação prospectiva nova."})
    salvar_json(pasta / "qualidade.json", qualidade)
    limites = {k: {"inicio": m.time.iloc[0], "fim_sinal": m.time.iloc[-1], "fim_alvo": m.time_alvo.max(), "amostras": len(m)} for k, m in metas.items()}
    salvar_json(pasta / "particoes.json", limites)
    distribuicao = pd.DataFrame([{"particao": k, **m.rotulo.value_counts().reindex([0, 1, 2], fill_value=0).to_dict()} for k, m in metas.items()])
    distribuicao.to_csv(pasta / "distribuicao_rotulos.csv", index=False)
    pd.DataFrame([{"particao": k, "rotulos_ambiguos_ou_indisponiveis": int(m.rotulo_auxiliar.isna().sum()),
        **m.rotulo_auxiliar.value_counts().reindex([0., 1., 2.], fill_value=0).to_dict()} for k, m in metas.items()]
        ).to_csv(pasta / "distribuicao_rotulo_auxiliar.csv", index=False)
    dados.to_csv(pasta / "dados_rotulados_lucro.csv", index=False)
    m_val, cache_modelos, cache_entradas = [], {}, {}
    for nome in modelos:
        janela = 12 if nome == "mlp_lucro" else 24 if nome == "gru_lucro" else None
        c = deepcopy(config)
        if janela is not None:
            c.janela = janela
        x = {k: extrair_entradas(amostras, idx, scaler, c, janela is not None) for k, idx in indices.items()}
        x_fit, metas_fit = dict(x), dict(metas)
        for parte in ["treino", "early_stopping"]:
            mascara_rotulo = metas[parte].rotulo_auxiliar.notna().to_numpy()
            x_fit[parte] = x[parte][mascara_rotulo]
            metas_fit[parte] = metas[parte].loc[mascara_rotulo].copy()
            metas_fit[parte]["rotulo"] = metas_fit[parte].rotulo_auxiliar.astype(int)
        logging.info("Treinando %s / %s", rotulagem, nome)
        modelo = treinar_lucro(nome, x_fit, metas_fit, c)
        modelo.tipo_alvo = "lucro_liquido"
        modelo.custo_por_lado = custo_por_lado
        modelo.rotulagem_treinamento = rotulagem
        pa = modelo.prever(x["ajuste_politica"], np.zeros(len(metas["ajuste_politica"])))["probabilidades"]
        politica, ajuste, busca = selecionar_politica(metas["ajuste_politica"], pa, config.horizonte, config.meta, config.minimo_sinais)
        modelo.politica_sinais = politica
        busca.to_csv(pasta / "politicas" / f"busca_{nome}.csv", index=False)
        salvar_json(pasta / "politicas" / f"{nome}.json", {"politica": politica, "resultado_ajuste": ajuste})
        pv = modelo.prever(x["validacao_selecao"], np.zeros(len(metas["validacao_selecao"])))
        val, _ = avaliar_operacoes(metas["validacao_selecao"], pv["classes"], config.horizonte, config.meta, config.minimo_sinais)
        m_val.append({"modelo": nome, **val})
        _salvar_predicoes(metas["validacao_selecao"], pv["probabilidades"], pv["classes"], pasta / "predicoes" / f"validacao_{nome}.csv")
        salvar_modelo(modelo, scaler, pasta / "modelos" / nome)
        salvar_json(pasta / "modelos" / nome / "parametros.json", modelo.parametros)
        for h in modelo.historicos:
            pd.DataFrame(h["history"]).to_csv(pasta / "historicos" / f"{nome}_{h['seed']}.csv", index=False)
        cache_modelos[nome], cache_entradas[nome] = modelo, x["teste"]
        logging.info("Validação %s: compra=%s (%d), venda=%s (%d)", nome, val["taxa_lucro_compra"], val["operacoes_compra"], val["taxa_lucro_venda"], val["operacoes_venda"])
    escolhido = max(m_val, key=pontuar)["modelo"]
    pd.DataFrame(m_val).to_csv(pasta / "metricas_validacao.csv", index=False)
    salvar_json(pasta / "escolha_antes_do_teste.json", {"modelo": escolhido,
        "criterio": "Maior menor taxa de lucro entre compra e venda na validação separada; suporte mínimo por direção.",
        "meta": config.meta, "minimo_operacoes_por_direcao": config.minimo_sinais})
    if apenas_validacao:
        salvar_json(pasta / "CONCLUIDO_VALIDACAO.json", {"concluido": True, "rotulagem": rotulagem,
            "modelo": escolhido, "teste_posterior_avaliado": False})
        return pasta, pd.DataFrame(m_val)
    resultados = []
    for nome, modelo in cache_modelos.items():
        o = modelo.prever(cache_entradas[nome], np.zeros(len(metas["teste"])))
        resultado, usados = avaliar_operacoes(metas["teste"], o["classes"], config.horizonte, config.meta, config.minimo_sinais)
        resultados.append({"modelo": nome, **resultado})
        p = _salvar_predicoes(metas["teste"], o["probabilidades"], o["classes"], pasta / "predicoes" / f"teste_{nome}.csv")
        operacoes, sensibilidade = simular(dados, p, config)
        operacoes.to_csv(pasta / "backtest" / f"operacoes_{nome}.csv", index=False)
        sensibilidade.to_csv(pasta / "backtest" / f"custos_{nome}.csv", index=False)
        referencia = sensibilidade.loc[np.isclose(sensibilidade.custo_por_lado, custo_por_lado)]
        if len(referencia):
            for coluna in ["operacoes", "retorno_acumulado", "taxa_lucro_compra", "taxa_lucro_venda"]:
                a, b = resultado[coluna], referencia.iloc[0][coluna]
                if a is not None and not np.isclose(a, b):
                    raise ValueError(f"Divergência entre rótulo e backtest: {nome}/{coluna}")
        _grafico_operacoes(resultado, pasta / "graficos" / f"lucro_{nome}.png", f"Lucro líquido: {nome}")
    for classe, nome in [(1, "sempre_compra"), (2, "sempre_venda")]:
        m, _ = avaliar_operacoes(metas["teste"], np.full(len(metas["teste"]), classe), config.horizonte, config.meta, config.minimo_sinais)
        resultados.append({"modelo": nome, **m})
    tabela = pd.DataFrame(resultados)
    tabela.to_csv(pasta / "metricas_teste.csv", index=False)
    modelo = cache_modelos[escolhido]
    config_salva = deepcopy(modelo.config)
    config_salva.modelos = tuple(modelos)
    manifesto = json.loads((pasta / "manifesto.json").read_text(encoding="utf-8"))
    manifesto["config"] = asdict(config_salva)
    salvar_json(pasta / "manifesto.json", manifesto)
    selecionado = tabela.loc[tabela.modelo == escolhido].iloc[0].to_dict()
    salvar_json(pasta / "selecao_modelo.json", {"modelo_selecionado": escolhido, "tipo_alvo": "lucro_liquido",
        "meta_atingida_no_teste": bool(selecionado["meta_70_lucro_ambas"]), "metricas": selecionado})
    salvar_json(pasta / "previsao_ultima_janela.json", prever_ultima_janela(bruto, modelo, scaler, modelo.config))
    _relatorio(pasta, config, custo_por_lado, escolhido, tabela, distribuicao, rotulagem)
    salvar_json(pasta / "CONCLUIDO.json", {"concluido": True, "modelo": escolhido})
    return pasta, tabela


def _relatorio(pasta, config, custo, escolhido, tabela, distribuicao, rotulagem="lucro_liquido"):
    formato = lambda x: "NA" if pd.isna(x) else f"{x:.2%}"
    linhas = ["# Operações com lucro líquido: rótulo e avaliação novos", "",
        f"Meta: pelo menos {config.meta:.0%} das compras executadas e {config.meta:.0%} das vendas executadas terminarem com lucro líquido; mínimo de {config.minimo_sinais} operações de cada direção.",
        f"Custo de referência: {custo*10000:g} pontos-base por lado. Sem spread e slippage. Modelo selecionado antes do teste: **{escolhido}**.", "",
        f"Rótulo de treinamento nesta execução: **{rotulagem}**. Ele pode ser um alvo auxiliar de movimento, trajetória ou barreiras; a taxa operacional continua contando lucro líquido após a mesma saída fixa.", "",
        f"O rótulo anterior exigia retorno de 0,8%. O novo representa lucro após entrada na próxima abertura, saída no fechamento após {config.horizonte} candles e taxas. São objetivos diferentes; não se apresentam as duas taxas como melhora no mesmo problema.", "",
        "O modelo só recebe atributos disponíveis até o fechamento do candle do sinal. Preço de entrada, saída e lucro futuro são usados exclusivamente no alvo e na avaliação.", "",
        "A validação foi dividida por sessão em três blocos: Early Stopping, ajuste da emissão dos sinais e seleção do modelo. As probabilidades não foram tratadas como garantia de acerto. Os limiares de emissão foram escolhidos no segundo bloco; o modelo foi escolhido no terceiro. O período de teste não escolheu esses parâmetros.", "",
        "O teste posterior já havia sido examinado sob outros objetivos. Esta é uma análise retrospectiva do novo rótulo e ainda precisa de confirmação prospectiva.", "",
        "| Modelo | Compras | Compras lucrativas | Acerto compra | Vendas | Vendas lucrativas | Acerto venda | Retorno acumulado |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in tabela.to_dict("records"):
        linhas.append(f"| {r['modelo']} | {r['operacoes_compra']} | {r['lucrativas_compra']} | {formato(r['taxa_lucro_compra'])} | {r['operacoes_venda']} | {r['lucrativas_venda']} | {formato(r['taxa_lucro_venda'])} | {formato(r['retorno_acumulado'])} |")
    linhas += ["", f"Somente operações realmente executadas entram no acerto. Sinais durante uma posição aberta são ignorados. Ausência de operações é NA e não 100%. Um mínimo de {config.minimo_sinais} é uma convenção de suporte, não garantia estatística; amostras pequenas continuam incertas.", "",
        "O backtest usa uma posição por vez e reinvestimento, sem aluguel de ações, margem, impacto de mercado ou drawdown intraposição. Uma taxa de acerto alta pode coexistir com prejuízo; por isso também se mostra o retorno e a sensibilidade às taxas.", "",
        "O campo manter no alvo segue a zona neutra da rotulagem de treinamento. Só no rótulo direto de lucro ele significa que nenhuma direção lucraria após custos. Na decisão, manter também pode representar confiança insuficiente para emitir um sinal; não é uma operação vencedora.", "",
        "## Arquivos", "",
        "- `dados_rotulados_lucro.csv`: alvos novos, preços futuros e lucros das duas direções.",
        "- `particoes.json` e `distribuicao_rotulos.csv`: datas, contagens e distribuição do novo alvo.",
        "- `politicas/`: limiares e todas as tentativas, usando somente a validação de ajuste.",
        "- `metricas_validacao.csv` e `escolha_antes_do_teste.json`: seleção separada do teste.",
        "- `metricas_teste.csv`: acerto líquido por direção, quantidade de operações e retorno.",
        "- `backtest/`: operações individuais e sensibilidade às taxas.",
        "- `modelos/` e `previsao_ultima_janela.json`: modelo, scaler, política de emissão e inferência sem resposta futura."]
    (pasta / "LEIA_RESULTADOS.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
