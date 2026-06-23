"""Janelas temporais e separação cronológica dos conjuntos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LimitesTemporais:
    fim_treino: int
    fim_validacao: int
    total: int


def calcular_limites_temporais(
    total: int, proporcao_treino: float, proporcao_validacao: float
) -> LimitesTemporais:
    """Calcula cortes por posição, sem qualquer embaralhamento."""
    fim_treino = int(total * proporcao_treino)
    fim_validacao = fim_treino + int(total * proporcao_validacao)
    if fim_treino <= 0 or fim_validacao <= fim_treino or fim_validacao >= total:
        raise ValueError("Poucos dados para as proporções temporais informadas.")
    return LimitesTemporais(fim_treino, fim_validacao, total)


def criar_sequencias(
    dados: pd.DataFrame,
    colunas_features: Iterable[str],
    coluna_rotulo: str = "rotulo",
    tamanho_janela: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monta X com os últimos N candles e y na posição final da janela."""
    colunas = list(colunas_features)
    matriz = dados[colunas].to_numpy(dtype=np.float32)
    rotulos = dados[coluna_rotulo].to_numpy(dtype=np.int64)
    if len(dados) < tamanho_janela:
        raise ValueError("O dataset é menor que a janela temporal configurada.")

    x, y, indices_alvo = [], [], []
    for indice in range(tamanho_janela - 1, len(dados)):
        inicio = indice - tamanho_janela + 1
        x.append(matriz[inicio : indice + 1])
        y.append(rotulos[indice])
        indices_alvo.append(indice)
    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(indices_alvo, dtype=np.int64),
    )


def separar_sequencias_temporais(
    x: np.ndarray,
    y: np.ndarray,
    indices_alvo: np.ndarray,
    limites: LimitesTemporais,
    horizonte_rotulo: int = 0,
):
    """Separa pela posição do alvo e expurga rótulos que cruzem uma fronteira.

    Como o rótulo de uma posição depende de ``t + horizonte_rotulo``, os últimos
    alvos de treino e validação são removidos. Sem esse *purge*, uma resposta do
    período seguinte poderia entrar no treinamento ou no early stopping.
    """
    if horizonte_rotulo < 0:
        raise ValueError("O horizonte do rótulo não pode ser negativo.")
    mascara_treino = indices_alvo < (limites.fim_treino - horizonte_rotulo)
    mascara_validacao = (indices_alvo >= limites.fim_treino) & (
        indices_alvo < (limites.fim_validacao - horizonte_rotulo)
    )
    mascara_teste = indices_alvo >= limites.fim_validacao

    conjuntos = {
        "treino": (x[mascara_treino], y[mascara_treino], indices_alvo[mascara_treino]),
        "validacao": (
            x[mascara_validacao],
            y[mascara_validacao],
            indices_alvo[mascara_validacao],
        ),
        "teste": (x[mascara_teste], y[mascara_teste], indices_alvo[mascara_teste]),
    }
    if any(len(valores[0]) == 0 for valores in conjuntos.values()):
        raise ValueError(
            "Um dos conjuntos ficou vazio; aumente os dados ou reduza janela/horizonte."
        )
    return conjuntos
