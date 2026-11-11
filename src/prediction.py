"""Inferência independente: usa atributos disponíveis, nunca rótulos futuros."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.forecast_config import ConfigPrevisao
from src.forecast_data import FEATURES, criar_atributos, limpar_candles
from src.forecast_models import ModeloPrevisao, tensorflow
from src.forecast_evaluation import NOMES


def salvar_modelo(modelo, scaler, pasta):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, pasta / "scaler.joblib")
    arquivos = []
    for i, estimador in enumerate(modelo.estimadores):
        nome = f"estimador_{i}.keras" if modelo.sequencial else f"estimador_{i}.joblib"
        if modelo.sequencial:
            estimador.save(pasta / nome)
        else:
            joblib.dump(estimador, pasta / nome)
        arquivos.append(nome)
    info = {"nome": modelo.nome, "sequencial": modelo.sequencial, "regressao": modelo.regressao,
            "media_alvo": modelo.media_alvo, "escala_alvo": modelo.escala_alvo, "arquivos": arquivos,
            "features": list(FEATURES), "janela": modelo.config.janela}
    info["ajuste_decisao"] = modelo.ajuste_decisao
    info.update({"politica_sinais": modelo.politica_sinais, "tipo_alvo": modelo.tipo_alvo, "custo_por_lado": modelo.custo_por_lado})
    info["rotulagem_treinamento"] = modelo.rotulagem_treinamento
    (pasta / "modelo.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def prever_ultima_janela(dados, modelo, scaler, config):
    limpos, _ = limpar_candles(dados)
    atributos = criar_atributos(limpos, config)
    ultima = atributos.iloc[-config.janela:]
    matriz = ultima[list(FEATURES)].to_numpy(dtype=np.float32)
    if len(ultima) != config.janela or not np.isfinite(matriz).all():
        raise ValueError("Histórico insuficiente para os indicadores e a última janela. Forneça pelo menos 120 candles válidos.")
    if len(limpos) <= config.horizonte:
        raise ValueError("Histórico insuficiente para a referência de momentum.")
    normalizada = scaler.transform(matriz).astype(np.float32)
    x = normalizada[None, :, :] if modelo.sequencial else normalizada[-1:]
    retorno_passado = limpos.close.iloc[-1] / limpos.close.iloc[-1-config.horizonte] - 1
    saida = modelo.prever(x, np.array([retorno_passado]))
    classe = int(saida["classes"][0])
    # MT5/CSV: time identifica a abertura; OHLC completo é conhecido no fechamento.
    emissao = limpos.time.iloc[-1] + pd.Timedelta(minutes=config.minutos_candle)
    registro = {
        "modelo": modelo.nome, "time_ultimo_candle": str(limpos.time.iloc[-1]),
        "instante_emissao": str(emissao),
        "instante_alvo_previsto": str(emissao + pd.Timedelta(minutes=config.horizonte * config.minutos_candle)),
        "preco_referencia": float(limpos.close.iloc[-1]),
        "horizonte_minutos": config.horizonte * config.minutos_candle,
        "limiar_retorno": config.limiar, "classe_predita": classe, "sinal": NOMES[classe],
        "condicao_alvo": "Os próximos h candles precisam ser consecutivos na mesma sessão; elegibilidade futura ainda não observada.",
        "status": "Previsão experimental a partir da data do arquivo; não certifica capacidade preditiva nem representa cotação atual.",
    }
    if saida["probabilidades"] is not None:
        registro.update({f"prob_{n}": float(saida["probabilidades"][0, i]) for i, n in enumerate(NOMES)})
    if modelo.tipo_alvo == "lucro_liquido":
        registro.pop("limiar_retorno")
        registro.update({"tipo_alvo": "lucro_liquido", "custo_por_lado": modelo.custo_por_lado,
                         "regra": "Entrada na próxima abertura; saída no fechamento após h candles; lucro após taxas.",
                         "politica_sinais": modelo.politica_sinais, "objetivo_operacional": "lucro_liquido"})
        if modelo.rotulagem_treinamento in {None, "lucro_liquido"}:
            registro["prob_nao_lucrativa_ambas"] = registro.get("prob_manter")
        else:
            registro["score_classe_neutra_auxiliar"] = registro.get("prob_manter")
        if modelo.rotulagem_treinamento is not None:
            registro["rotulagem_treinamento"] = modelo.rotulagem_treinamento
            registro["natureza_scores"] = "Escores do rótulo de treinamento; lucro líquido é avaliado separadamente. Não são garantia de acerto."
    if saida["retornos"] is not None:
        r = float(saida["retornos"][0])
        registro.update({"retorno_previsto": r, "preco_previsto": float(limpos.close.iloc[-1] * (1 + r))})
    return registro


def carregar_e_prever(dados, pasta_execucao, nome_modelo=None):
    pasta = Path(pasta_execucao)
    if (pasta / "modelo_tcc_cenario.json").exists():
        from src.tcc_noise_training import carregar_prever_cenario_tcc
        return carregar_prever_cenario_tcc(dados, pasta)
    if (pasta / "modelo_meta.json").exists():
        from src.meta_filter import carregar_prever_meta
        return carregar_prever_meta(dados, pasta)
    if (pasta / "modelo_estrategia.json").exists():
        from src.bracket_strategy import carregar_prever_estrategia
        return carregar_prever_estrategia(dados, pasta)
    if (pasta / "modelo_especialistas.json").exists():
        from src.specialist_study import carregar_prever_especialistas
        return carregar_prever_especialistas(dados, pasta)
    if (pasta / "modelo_binario.json").exists():
        from src.binary_events import carregar_prever_binario
        return carregar_prever_binario(dados, pasta)
    selecao = json.loads((pasta / "selecao_modelo.json").read_text(encoding="utf-8"))
    manifesto = json.loads((pasta / "manifesto.json").read_text(encoding="utf-8"))
    config = ConfigPrevisao(**manifesto["config"])
    escolhido = nome_modelo or selecao["modelo_selecionado"]
    if escolhido not in config.modelos:
        raise ValueError("O modelo solicitado não pertence a esta execução.")
    pasta_modelo = pasta / "modelos" / escolhido
    info = json.loads((pasta_modelo / "modelo.json").read_text(encoding="utf-8"))
    if info["features"] != list(FEATURES):
        raise ValueError("A versão dos atributos mudou; use o código registrado na execução ou retreine.")
    config.janela = info.get("janela", config.janela)
    modelo = ModeloPrevisao(info["nome"], config, regressao=info["regressao"], sequencial=info["sequencial"],
                            media_alvo=info["media_alvo"], escala_alvo=info["escala_alvo"], ajuste_decisao=info.get("ajuste_decisao"),
                            politica_sinais=info.get("politica_sinais"), tipo_alvo=info.get("tipo_alvo", "retorno_limiar"),
                            custo_por_lado=info.get("custo_por_lado"), rotulagem_treinamento=info.get("rotulagem_treinamento"))
    for arquivo in info["arquivos"]:
        estimador = tensorflow().keras.models.load_model(pasta_modelo / arquivo, compile=False) if info["sequencial"] else joblib.load(pasta_modelo / arquivo)
        modelo.estimadores.append(estimador)
    scaler = joblib.load(pasta_modelo / "scaler.joblib")
    return prever_ultima_janela(dados, modelo, scaler, config)
