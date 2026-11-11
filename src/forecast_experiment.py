"""Previsão primeiro: desenvolvimento temporal, teste separado e simulação."""

from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import logging
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.forecast_backtest import simular
from src.forecast_data import (FEATURES, classes_do_retorno, criar_alvos, criar_atributos,
    extrair_entradas, indices_do_corte, limpar_candles, montar_amostras, planejar_cortes)
from src.forecast_evaluation import NOMES, avaliar, bootstrap_por_sessao, selecionar_modelo
from src.forecast_models import treinar
from src.prediction import prever_ultima_janela, salvar_modelo


def _json(valor):
    if isinstance(valor, (pd.Timestamp, Path)):
        return str(valor)
    if isinstance(valor, (np.generic,)):
        return valor.item()
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    raise TypeError(type(valor).__name__)


def salvar_json(caminho, valor):
    Path(caminho).write_text(json.dumps(valor, ensure_ascii=False, indent=2, default=_json, allow_nan=False), encoding="utf-8")


def tabela_predicoes(meta, saida, config):
    p = meta.copy()
    p = p.rename(columns={"rotulo": "classe_real"})
    p["instante_emissao"] = p.time + pd.Timedelta(minutes=config.minutos_candle)
    p["instante_realizacao"] = p.time_alvo + pd.Timedelta(minutes=config.minutos_candle)
    p["classe_predita"] = saida["classes"]
    if saida["probabilidades"] is not None:
        for i, nome in enumerate(NOMES):
            p[f"prob_{nome}"] = saida["probabilidades"][:, i]
    if saida["retornos"] is not None:
        p["retorno_previsto"] = saida["retornos"]
        p["preco_previsto"] = p.close * (1 + p.retorno_previsto)
    return p


def _matriz(real, previsto, destino, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(real, previsto, labels=[0, 1, 2])
    percentual = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros((3, 3)), where=cm.sum(axis=1, keepdims=True) > 0)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(percentual, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]}\n{percentual[i,j]:.1%}", ha="center", va="center", color="white" if percentual[i,j] > .55 else "black")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=NOMES, yticklabels=NOMES,
           xlabel="Classe prevista", ylabel="Classe real", title=titulo)
    fig.tight_layout()
    fig.savefig(destino, dpi=160)
    plt.close(fig)


def _resumo(pasta, tabela, ranking, selecao, qualidade, config):
    linhas = ["# Previsão sem ruído: resultado da execução", "",
        f"Modelo escolhido nos folds de desenvolvimento: **{selecao['modelo_selecionado']}**.", "",
        "O teste final é uma avaliação retrospectiva da base disponível. Esse período já foi examinado na auditoria do TCC; a confirmação em um período novo continua pendente.", "",
        f"Alvo: retorno de {config.horizonte} candles consecutivos de {config.minutos_candle} minutos dentro da mesma sessão; compra >= {config.limiar:.2%}, venda <= -{config.limiar:.2%}, manter entre as barreiras.",
        "As entradas usam candles negociados passados e podem incluir pregões anteriores. Os timestamps de candles representam a abertura; a emissão ocorre após o fechamento.", "",
        "| Modelo | Acurácia | F1 macro | Recall manter | Recall compra | Recall venda | Precisão compra | Precisão venda |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in tabela.to_dict("records"):
        ps = lambda x: "NA" if pd.isna(x) else f"{x:.2%}"
        linhas.append(f"| {r['modelo']} | {ps(r['accuracy'])} | {r['f1_macro']:.4f} | {ps(r['recall_manter'])} | {ps(r['recall_compra'])} | {ps(r['recall_venda'])} | {ps(r['precision_compra'])} | {ps(r['precision_venda'])} |")
    linhas += ["", "A matriz de confusão apresenta percentuais por linha; sua diagonal é o recall de cada classe. Acurácia de 70% e diagonal de 70% em todas as classes são critérios diferentes.", "",
        f"Critério de diagonal de pelo menos {config.meta:.0%} em todas as classes para o selecionado: **{selecao['meta_diagonal_atingida']}**.",
        f"Precisão de pelo menos {config.meta:.0%} em compra e venda: **{selecao['meta_precisao_atingida']}**.",
        f"Critérios combinados desta execução: **{selecao['criterios_combinados_atingidos']}**.", "",
        "Nenhum resultado dispara experimento de ruído automaticamente. Um resultado negativo permanece registrado, sem troca de limiar com base no teste.", "",
        "## Avaliações incluídas", "",
        "- Folds temporais com treino crescente, validação posterior e blocos de teste de desenvolvimento sem sobreposição.",
        "- Seleção de modelo pela média de F1 macro nesses folds, congelada antes de avaliar o teste final.",
        "- Referência majoritária ajustada só no treino; momentum, modelos tabulares e redes comparados nas mesmas datas.",
        "- Redes repetidas nas sementes declaradas; previsão final pela média, sem escolher a semente que se saiu melhor no teste.",
        "- Alvos contínuos previstos por Ridge e LSTM de regressão, com MAE/RMSE comparados ao retorno zero.",
        "- Bootstrap por sessão para incerteza do modelo selecionado, sem tratar janelas sobrepostas como independentes.",
        "- Simulação de uma posição por vez, entrada na abertura seguinte e saída no fechamento-alvo, com custos hipotéticos de 0, 2 e 5 bps por lado na configuração padrão.", "",
        "A simulação não usa o fechamento já conhecido como preço garantido de entrada. O resultado financeiro difere do acerto da classe e inclui cenários simplificados, sem aluguel de ações, margem, restrições de liquidez ou drawdown intraposição.", "",
        "## Arquivos", "",
        "- `manifesto.json`: configuração, origem, versões e identificação do código.",
        "- `qualidade_dados.json` e `distribuicao_classes.csv`: remoções e distribuição de classes em cada partição.",
        "- `cortes_temporais.json`: fronteiras por data, sem reutilizar cortes de bases com tamanhos diferentes.",
        "- `metricas_desenvolvimento.csv`, `ranking_desenvolvimento.csv` e `metricas_teste_final.csv`.",
        "- `metricas_por_semente.csv`: redes individuais, separadas da previsão agregada.",
        "- `predicoes/`: instante de emissão, alvo, classes reais/previstas e probabilidades ou retorno numérico.",
        "- `graficos/`: matrizes em contagem e percentuais por linha.",
        "- `backtest/`: operações e resultados por cenário de custo.",
        "- `previsao_ultima_janela.json`: inferência sobre a última janela disponível, sem rótulo futuro.", "",
        "Não foram usados dados de outros ativos nem períodos adicionais inexistentes na entrada. Alterar horizonte, barreira ou arquitetura constitui um novo experimento, que deve ser documentado e confirmado fora do desenvolvimento."]
    (pasta / "LEIA_RESULTADOS.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def executar_previsao(csv, pasta_saida, config):
    config.validar()
    csv = Path(csv).resolve()
    original = pd.read_csv(csv)
    original.columns = [str(c).strip().lower() for c in original.columns]
    limpos, qualidade = limpar_candles(original)
    dados = criar_alvos(criar_atributos(limpos, config), config)
    dados["retorno_passado"] = dados.close.pct_change(config.horizonte, fill_method=None)
    amostras = montar_amostras(dados, config)
    if len(amostras.metadados) < 100:
        raise ValueError("Menos de 100 exemplos elegíveis; forneça mais histórico.")
    amostras.metadados["retorno_passado"] = dados.retorno_passado.iloc[amostras.finais_janelas].to_numpy()
    cortes = planejar_cortes(dados, config)
    pasta = Path(pasta_saida) / datetime.now(timezone.utc).strftime("execucao_%Y%m%dT%H%M%S_%fZ")
    pasta.mkdir(parents=True, exist_ok=False)
    for nome in ["predicoes", "modelos", "historicos", "graficos", "backtest"]:
        (pasta / nome).mkdir()
    try:
        git_rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        git_rev = None
    arquivos_codigo = sorted(Path(__file__).parent.glob("forecast_*.py")) + [Path(__file__).parent / "prediction.py"]
    manifesto = {"config": asdict(config), "csv": str(csv), "sha256_csv": hashlib.sha256(csv.read_bytes()).hexdigest(),
                 "versoes": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "tensorflow", "keras"]},
                 "git_head": git_rev, "sha256_codigo": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in arquivos_codigo},
                 "politica": "Sem ruído. Modelo escolhido só nos testes de desenvolvimento; teste final retrospectivo já examinado na auditoria anterior.",
                 "convencao_time": "Abertura do candle; OHLC fica disponível time + minutos_candle."}
    salvar_json(pasta / "manifesto.json", manifesto)
    salvar_json(pasta / "cortes_temporais.json", [asdict(c) for c in cortes])
    qualidade.update({"inicio": limpos.time.iloc[0], "fim": limpos.time.iloc[-1],
                     "alvos_elegiveis": int(dados.alvo_elegivel.sum()), "amostras_com_janela": len(amostras.metadados),
                     "excluidos_sem_futuro_continuo_na_sessao": int((~dados.alvo_elegivel).sum()),
                     "contagens_alvos_elegiveis": dados.loc[dados.alvo_elegivel, "rotulo"].value_counts().sort_index().to_dict()})
    salvar_json(pasta / "qualidade_dados.json", qualidade)
    dados.to_csv(pasta / "dados_supervisionados.csv", index=False)
    metricas_dev, metricas_final, por_semente, distribuicoes, escolhas = [], [], [], [], []
    vencedor, ranking = None, None
    predicoes_finais = {}
    baselines = None
    ultima_previsao = None
    for corte in cortes:
        partes = indices_do_corte(amostras.metadados, corte)
        logging.info("%s: treino=%d, validação=%d, teste=%d", corte.nome, *(len(partes[k]) for k in ["treino", "validacao", "teste"]))
        if corte.nome == "teste_final":
            vencedor, ranking = selecionar_modelo(pd.DataFrame(metricas_dev))
            ranking.to_csv(pasta / "ranking_desenvolvimento.csv", index=False)
            salvar_json(pasta / "escolha_antes_do_teste.json", {"modelo": vencedor, "criterio": "media_f1_macro_folds_desenvolvimento"})
        matriz = amostras.features_linhas
        mascara_fit = (dados.time < corte.inicio_validacao).to_numpy() & np.isfinite(matriz).all(axis=1)
        scaler = StandardScaler().fit(matriz[mascara_fit])
        metas = {k: amostras.metadados.iloc[v].reset_index(drop=True) for k, v in partes.items()}
        for parte, meta in metas.items():
            distribuicoes.append({"fold": corte.nome, "particao": parte, "amostras": len(meta),
                **{n: int((meta.rotulo == i).sum()) for i, n in enumerate(NOMES)}})
        cache = {}
        for nome in config.modelos:
            sequencial = nome.startswith("lstm")
            if sequencial not in cache:
                cache[sequencial] = {k: extrair_entradas(amostras, indices, scaler, config, sequencial) for k, indices in partes.items()}
            x = cache[sequencial]
            mt, mv, me = (metas[k] for k in ["treino", "validacao", "teste"])
            logging.info("%s: ajustando %s...", corte.nome, nome)
            modelo = treinar(nome, x["treino"], mt.rotulo.to_numpy(), mt.retorno_futuro.to_numpy(),
                             x["validacao"], mv.rotulo.to_numpy(), mv.retorno_futuro.to_numpy(), config)
            escolhas.append({"fold": corte.nome, "modelo": nome, "parametros": modelo.parametros})
            saida = modelo.prever(x["teste"], me.retorno_passado.to_numpy())
            m, _ = avaliar(me.rotulo, saida["classes"], config, saida["probabilidades"], me.retorno_futuro, saida["retornos"])
            registro = {"fold": corte.nome, "modelo": nome, **m}
            (metricas_final if corte.nome == "teste_final" else metricas_dev).append(registro)
            p = tabela_predicoes(me, saida, config)
            p.to_csv(pasta / "predicoes" / f"{corte.nome}_{nome}.csv", index=False)
            for i, historia in enumerate(modelo.historicos):
                pd.DataFrame(historia["history"]).to_csv(pasta / "historicos" / f"{corte.nome}_{nome}_{historia['seed']}.csv", index=False)
                valor = saida["individuais"][i]
                yp = classes_do_retorno(valor, config.limiar) if modelo.regressao else np.argmax(valor, axis=1)
                ms, _ = avaliar(me.rotulo, yp, config, retorno_real=me.retorno_futuro,
                                retorno_previsto=valor if modelo.regressao else None)
                por_semente.append({"fold": corte.nome, "modelo": nome, "seed": historia["seed"], **ms})
            if corte.nome == "teste_final":
                predicoes_finais[nome] = p
                salvar_modelo(modelo, scaler, pasta / "modelos" / nome)
                _matriz(me.rotulo, saida["classes"], pasta / "graficos" / f"matriz_{nome}.png", f"Teste final: {nome}")
                trades, resumo_trades = simular(dados, p, config)
                trades.to_csv(pasta / "backtest" / f"operacoes_{nome}.csv", index=False)
                resumo_trades.to_csv(pasta / "backtest" / f"metricas_{nome}.csv", index=False)
                if nome == vencedor:
                    ultima_previsao = prever_ultima_janela(limpos, modelo, scaler, config)
                if nome == "majoritaria":
                    baselines = m
            pd.DataFrame(metricas_dev).to_csv(pasta / "metricas_desenvolvimento.csv", index=False)
            logging.info("%s / %s: accuracy=%.3f, F1 macro=%.3f, recall macro=%.3f", corte.nome, nome, m["accuracy"], m["f1_macro"], m["recall_macro"])
    tabela = pd.DataFrame(metricas_final)
    tabela.to_csv(pasta / "metricas_teste_final.csv", index=False)
    pd.DataFrame(por_semente).to_csv(pasta / "metricas_por_semente.csv", index=False)
    pd.DataFrame(distribuicoes).to_csv(pasta / "distribuicao_classes.csv", index=False)
    salvar_json(pasta / "hiperparametros_selecionados.json", escolhas)
    selecionado = tabela.loc[tabela.modelo == vencedor].iloc[0]
    selecao = {"modelo_selecionado": vencedor, "criterio": "media_f1_macro_folds_desenvolvimento",
        "meta_diagonal_atingida": bool(selecionado.meta_70_diagonal_todas_classes),
        "meta_precisao_atingida": bool(selecionado.meta_70_precisao_compra_venda),
        "superou_baseline_f1_teste": bool(selecionado.f1_macro > baselines["f1_macro"]),
        "criterios_combinados_atingidos": bool(selecionado.meta_70_acuracia and selecionado.meta_70_diagonal_todas_classes
            and selecionado.meta_70_precisao_compra_venda and selecionado.sinais_suficientes and selecionado.f1_macro > baselines["f1_macro"]),
        "confirmacao_em_periodo_novo_pendente": True, "ruido_executado": False}
    salvar_json(pasta / "selecao_modelo.json", selecao)
    salvar_json(pasta / "incerteza_modelo_selecionado.json", bootstrap_por_sessao(predicoes_finais[vencedor], config.bootstrap_sessoes))
    salvar_json(pasta / "previsao_ultima_janela.json", ultima_previsao)
    _resumo(pasta, tabela, ranking, selecao, qualidade, config)
    salvar_json(pasta / "CONCLUIDO.json", {"concluido": True, "modelo_selecionado": vencedor})
    return pasta, tabela
