"""Métricas globais e métricas de falsos positivos por classe."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


CLASSES = (0, 1, 2)


def metricas_binarias_por_classe(matriz: np.ndarray, classe: int) -> dict:
    """Extrai TP, FP, FN, TN, FPR e FDR no esquema um-contra-todos."""
    tp = int(matriz[classe, classe])
    fp = int(matriz[:, classe].sum() - tp)
    fn = int(matriz[classe, :].sum() - tp)
    tn = int(matriz.sum() - tp - fp - fn)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fdr = fp / (fp + tp) if (fp + tp) else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "FPR": fpr, "FDR": fdr}


def avaliar_classificacao(y_real, y_predito) -> tuple[dict, np.ndarray]:
    """Calcula todas as métricas pedidas usando sempre as três classes."""
    matriz = confusion_matrix(y_real, y_predito, labels=CLASSES)
    por_classe = {
        classe: metricas_binarias_por_classe(matriz, classe) for classe in CLASSES
    }
    metricas = {
        "accuracy": accuracy_score(y_real, y_predito),
        "precision_macro": precision_score(
            y_real, y_predito, labels=CLASSES, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_real, y_predito, labels=CLASSES, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_real, y_predito, labels=CLASSES, average="macro", zero_division=0
        ),
        # FPR geral definido como a média macro dos FPRs um-contra-todos.
        "FPR_geral": float(np.mean([por_classe[c]["FPR"] for c in CLASSES])),
        "FPR_compra": por_classe[1]["FPR"],
        "FPR_venda": por_classe[2]["FPR"],
        "FDR_compra": por_classe[1]["FDR"],
        "FDR_venda": por_classe[2]["FDR"],
        "quantidade_sinais_compra": int(np.sum(np.asarray(y_predito) == 1)),
        "quantidade_sinais_venda": int(np.sum(np.asarray(y_predito) == 2)),
        "metricas_por_classe": por_classe,
    }
    return metricas, matriz

