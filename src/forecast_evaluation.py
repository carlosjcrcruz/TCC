"""Avaliação de previsão, com a diagonal por classe e FDR indefinida sem sinais."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss


NOMES = ("manter", "compra", "venda")


def avaliar(real, previsto, config, probabilidades=None, retorno_real=None, retorno_previsto=None):
    real, previsto = np.asarray(real, dtype=int), np.asarray(previsto, dtype=int)
    if not len(real) or len(real) != len(previsto):
        raise ValueError("Vetores de avaliação vazios ou incompatíveis.")
    cm = confusion_matrix(real, previsto, labels=[0, 1, 2])
    recalls = np.divide(cm.diagonal(), cm.sum(axis=1), out=np.zeros(3), where=cm.sum(axis=1) > 0)
    precisao = np.divide(cm.diagonal(), cm.sum(axis=0), out=np.full(3, np.nan), where=cm.sum(axis=0) > 0)
    resultado = {
        "amostras": len(real), "accuracy": float(accuracy_score(real, previsto)),
        "f1_macro": float(f1_score(real, previsto, labels=[0, 1, 2], average="macro", zero_division=0)),
        "recall_macro": float(recalls.mean()),
        "meta_70_acuracia": bool(accuracy_score(real, previsto) >= config.meta),
        "meta_70_diagonal_todas_classes": bool((cm.sum(axis=1) > 0).all() and (recalls >= config.meta).all()),
    }
    for k, nome in enumerate(NOMES):
        sinais = int(cm[:, k].sum())
        resultado[f"suporte_{nome}"] = int(cm[k].sum())
        resultado[f"sinais_{nome}"] = sinais
        resultado[f"precision_{nome}"] = float(precisao[k]) if sinais else None
        resultado[f"recall_{nome}"] = float(recalls[k])
        resultado[f"FDR_{nome}"] = float(1 - precisao[k]) if sinais else None
    resultado["meta_70_precisao_compra_venda"] = bool(
        all(resultado[f"precision_{n}"] is not None and resultado[f"precision_{n}"] >= config.meta for n in ["compra", "venda"])
    )
    resultado["sinais_suficientes"] = bool(all(resultado[f"sinais_{n}"] >= config.minimo_sinais for n in ["compra", "venda"]))
    if probabilidades is not None:
        p = np.asarray(probabilidades)
        if p.shape != (len(real), 3) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
            raise ValueError("Probabilidades inválidas.")
        p = p.astype(float) / p.sum(axis=1, keepdims=True)
        resultado["log_loss"] = float(log_loss(real, p, labels=[0, 1, 2]))
        resultado["brier_multiclasse"] = float(np.mean(np.sum((p - np.eye(3)[real]) ** 2, axis=1)))
    if retorno_previsto is not None:
        r, estimado = np.asarray(retorno_real), np.asarray(retorno_previsto)
        if len(r) != len(real) or not np.isfinite(estimado).all():
            raise ValueError("Retornos previstos inválidos.")
        mse = float(np.mean((r - estimado) ** 2))
        mse_zero = float(np.mean(r ** 2))
        resultado.update({
            "mae_retorno": float(np.mean(np.abs(r - estimado))), "rmse_retorno": float(np.sqrt(mse)),
            "rmse_retorno_zero": float(np.sqrt(mse_zero)),
            "ganho_mse_vs_retorno_zero": float(1 - mse / mse_zero) if mse_zero else None,
        })
    return resultado, cm


def pontuacao_validacao(real, previsto):
    return float(f1_score(real, previsto, labels=[0, 1, 2], average="macro", zero_division=0))


def selecionar_modelo(tabela):
    """Recebe somente resultados dos folds de desenvolvimento, nunca holdout."""
    if "fold" in tabela and (tabela.fold == "teste_final").any():
        raise ValueError("O teste final não pode ser usado para selecionar o modelo.")
    medias = tabela.groupby("modelo")[["f1_macro", "recall_macro"]].mean()
    ranking = medias.sort_values(["f1_macro", "recall_macro"], ascending=False, kind="stable")
    return str(ranking.index[0]), ranking.reset_index()


def bootstrap_por_sessao(predicoes, repeticoes=300, seed=42):
    """Reamostra dias completos para preservar dependência dentro da sessão."""
    grupos = [g for _, g in predicoes.groupby("sessao", sort=True)]
    if len(grupos) < 2 or repeticoes == 0:
        return {"metodo": "bootstrap_por_sessao", "sessoes": len(grupos), "intervalos": None}
    rng = np.random.default_rng(seed)
    valores = []
    for _ in range(repeticoes):
        amostra = pd.concat([grupos[i] for i in rng.integers(0, len(grupos), len(grupos))], ignore_index=True)
        valores.append([
            accuracy_score(amostra.classe_real, amostra.classe_predita),
            f1_score(amostra.classe_real, amostra.classe_predita, labels=[0, 1, 2], average="macro", zero_division=0),
        ])
    limites = np.quantile(np.asarray(valores), [0.025, 0.975], axis=0)
    return {"metodo": "bootstrap_por_sessao", "sessoes": len(grupos), "repeticoes": repeticoes,
            "intervalos_95": {n: limites[:, i].tolist() for i, n in enumerate(["accuracy", "f1_macro"])},
            "limitacao": "Condicional aos modelos treinados; não inclui toda a incerteza de seleção ou de mudança de regime."}
