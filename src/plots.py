"""Gráficos salvos em PNG para documentação dos experimentos."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Mapping

# Alguns ambientes acadêmicos/restritos não permitem gravar em ~/.matplotlib.
# O cache temporário evita avisos sem misturar arquivos internos aos resultados.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tcc_metodologia_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, PercentFormatter


NOMES_CLASSES = ["Manter", "Compra", "Venda"]


def _preparar_destino(caminho: Path | str) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    return caminho


def plotar_original_vs_ruidos(
    dados_originais: pd.DataFrame,
    cenarios: Mapping[str, pd.DataFrame],
    caminho: Path | str,
) -> None:
    """Compara o close original com todas as intensidades de ruído."""
    caminho = _preparar_destino(caminho)
    fig, eixos = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    ruidosos = [(nome, df) for nome, df in cenarios.items() if nome.startswith("Ruído")]
    for eixo, (nome, dados) in zip(eixos.flat, ruidosos):
        eixo.plot(dados_originais["time"], dados_originais["close"], label="Original", lw=1)
        eixo.plot(dados["time"], dados["close_modelo"], label=nome, lw=0.8, alpha=0.8)
        eixo.set_title(nome)
        eixo.set_ylabel("Preço")
        eixo.grid(alpha=0.25)
        eixo.legend()
    fig.suptitle("Série original versus séries com ruído gaussiano")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(caminho, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plotar_comparacao_serie(
    dados_originais: pd.DataFrame,
    dados_tratados: pd.DataFrame,
    nome_tratamento: str,
    caminho: Path | str,
) -> None:
    caminho = _preparar_destino(caminho)
    fig, eixo = plt.subplots(figsize=(14, 6))
    eixo.plot(dados_originais["time"], dados_originais["close"], label="Original", lw=1)
    eixo.plot(
        dados_tratados["time"],
        dados_tratados["close_modelo"],
        label=nome_tratamento,
        lw=1,
        alpha=0.85,
    )
    eixo.set(title=f"Série original versus {nome_tratamento}", xlabel="Tempo", ylabel="Preço")
    eixo.grid(alpha=0.25)
    eixo.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(caminho, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plotar_matriz_confusao(
    matriz: np.ndarray, nome_cenario: str, caminho: Path | str
) -> None:
    caminho = _preparar_destino(caminho)
    fig, eixo = plt.subplots(figsize=(6.5, 5.5))
    imagem = eixo.imshow(matriz, interpolation="nearest", cmap="Blues")
    fig.colorbar(imagem, ax=eixo)
    eixo.set(
        title=f"Matriz de confusão — {nome_cenario}",
        xlabel="Classe predita",
        ylabel="Classe real",
        xticks=range(3),
        yticks=range(3),
        xticklabels=NOMES_CLASSES,
        yticklabels=NOMES_CLASSES,
    )
    limite = matriz.max() / 2 if matriz.size else 0
    for linha in range(3):
        for coluna in range(3):
            valor = int(matriz[linha, coluna])
            eixo.text(
                coluna,
                linha,
                str(valor),
                ha="center",
                va="center",
                color="white" if valor > limite else "black",
            )
    fig.tight_layout()
    fig.savefig(caminho, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plotar_comparacao_fpr(tabela: pd.DataFrame, caminho: Path | str) -> None:
    caminho = _preparar_destino(caminho)
    x = np.arange(len(tabela))
    largura = 0.38
    fig, eixo = plt.subplots(figsize=(12, 6))
    barras_compra = eixo.bar(
        x - largura / 2,
        tabela["FPR_compra"],
        largura,
        label="Compra",
        color="#A3BEFA",
        edgecolor="#2E4780",
    )
    barras_venda = eixo.bar(
        x + largura / 2,
        tabela["FPR_venda"],
        largura,
        label="Venda",
        color="#F0986E",
        edgecolor="#804126",
    )
    maior_fpr = float(tabela[["FPR_compra", "FPR_venda"]].max().max())
    passo_eixo = 0.05 if maior_fpr >= 0.10 else 0.02
    limite_superior = max(
        passo_eixo,
        np.ceil((maior_fpr * 1.18) / passo_eixo) * passo_eixo,
    )
    eixo.set(
        xlabel="Cenário",
        ylabel="FPR (%)",
        xticks=x,
        xticklabels=tabela["cenario"],
        ylim=(0, limite_superior),
    )
    fig.text(
        0.08,
        0.975,
        "Taxa de falsos positivos por cenário",
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.94,
        "Valores percentuais sobre as barras; escala ajustada ao maior FPR observado.",
        ha="left",
        va="top",
        fontsize=9,
        color="#6F768A",
    )
    eixo.yaxis.set_major_locator(MultipleLocator(passo_eixo))
    eixo.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    eixo.tick_params(axis="x", rotation=30)
    eixo.grid(axis="y", alpha=0.25)
    eixo.legend()
    eixo.bar_label(
        barras_compra,
        labels=[f"{valor:.2%}" for valor in tabela["FPR_compra"]],
        padding=3,
        fontsize=8,
    )
    eixo.bar_label(
        barras_venda,
        labels=[f"{valor:.2%}" for valor in tabela["FPR_venda"]],
        padding=3,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(caminho, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plotar_comparacao_f1(tabela: pd.DataFrame, caminho: Path | str) -> None:
    caminho = _preparar_destino(caminho)
    fig, eixo = plt.subplots(figsize=(12, 6))
    barras = eixo.bar(tabela["cenario"], tabela["f1_macro"], color="#4472C4")
    eixo.set(title="F1-score macro por cenário", xlabel="Cenário", ylabel="F1 macro", ylim=(0, 1))
    eixo.tick_params(axis="x", rotation=30)
    eixo.grid(axis="y", alpha=0.25)
    eixo.bar_label(barras, fmt="%.3f", padding=3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=160, bbox_inches="tight")
    plt.close(fig)
