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
import seaborn as sns
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
    dados_ruidosos: pd.DataFrame,
    dados_tratados: pd.DataFrame,
    nome_tratamento: str,
    caminho: Path | str,
    intensidade_ruido: float,
    max_pontos: int = 600,
) -> None:
    """Compara entrada ruidosa, saída filtrada e ruído efetivamente removido."""
    caminho = _preparar_destino(caminho)
    inicio = max(0, len(dados_originais) - max_pontos)
    tempo = pd.to_datetime(dados_originais["time"].iloc[inicio:], utc=True)
    indice_candle = np.arange(len(tempo))
    original = dados_originais["close"].iloc[inicio:].to_numpy(dtype=float)
    ruidosa = dados_ruidosos["close_modelo"].iloc[inicio:].to_numpy(dtype=float)
    filtrada = dados_tratados["close_modelo"].iloc[inicio:].to_numpy(dtype=float)
    ruido_injetado = ruidosa - original
    ruido_removido = ruidosa - filtrada
    correlacao = float(np.corrcoef(ruido_injetado, ruido_removido)[0, 1])

    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": "#FCFCFD",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D7DBE7",
            "grid.color": "#E6E8F0",
            "font.family": "sans-serif",
        },
    )
    fig, eixos = plt.subplots(
        2,
        1,
        figsize=(15, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1]},
    )
    sns.lineplot(
        x=indice_candle,
        y=original,
        ax=eixos[0],
        color="#464C55",
        linewidth=1.0,
        label="Série original (referência)",
    )
    sns.lineplot(
        x=indice_candle,
        y=ruidosa,
        ax=eixos[0],
        color="#F0986E",
        linewidth=0.9,
        linestyle="--",
        label=f"Série com ruído (σ={intensidade_ruido:g})",
    )
    sns.lineplot(
        x=indice_candle,
        y=filtrada,
        ax=eixos[0],
        color="#5477C4",
        linewidth=1.0,
        label=f"Série após {nome_tratamento}",
    )
    eixos[0].set(ylabel="Preço", xlabel="")
    eixos[0].legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        frameon=False,
        ncol=3,
        borderaxespad=0,
    )

    sns.lineplot(
        x=indice_candle,
        y=ruido_injetado,
        ax=eixos[1],
        color="#F0986E",
        linewidth=0.9,
        linestyle="--",
        label="Ruído injetado: ruidosa − original",
    )
    sns.lineplot(
        x=indice_candle,
        y=ruido_removido,
        ax=eixos[1],
        color="#5477C4",
        linewidth=0.9,
        label=f"Ruído removido: ruidosa − {nome_tratamento}",
    )
    eixos[1].axhline(0, color="#464C55", linewidth=0.8, linestyle=":")
    eixos[1].set(xlabel="Índice sequencial do candle", ylabel="Resíduo em preço")
    eixos[1].legend(frameon=False, loc="upper left", ncol=2)
    eixos[1].text(
        0.99,
        0.94,
        f"Correlação entre ruído injetado e removido: {correlacao:.3f}",
        transform=eixos[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#1F2430",
        bbox={"facecolor": "white", "edgecolor": "#D7DBE7", "alpha": 0.90},
    )

    for eixo in eixos:
        eixo.spines[["top", "right"]].set_visible(False)
        eixo.grid(axis="y", alpha=0.8)
        eixo.grid(axis="x", visible=False)

    fig.suptitle(
        f"{nome_tratamento} aplicado à série com ruído gaussiano",
        x=0.08,
        y=0.995,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color="#1F2430",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(caminho, dpi=160, bbox_inches="tight")
    fig.savefig(caminho.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plotar_histograma_ruido(
    serie_original: pd.Series,
    serie_ruidosa: pd.Series,
    intensidade: float,
    caminho: Path | str,
) -> None:
    """Mostra a distribuição de ``ruidosa - original`` e sua forma relativa.

    O segundo painel é o teste visual apropriado para o modelo multiplicativo
    ``ruidosa = original * (1 + epsilon)``, no qual epsilon segue N(0, sigma²).
    """
    caminho = _preparar_destino(caminho)
    original = serie_original.to_numpy(dtype=float)
    ruidosa = serie_ruidosa.to_numpy(dtype=float)
    ruido_absoluto = ruidosa - original
    ruido_relativo = ruidosa / original - 1.0
    media = float(np.mean(ruido_relativo))
    desvio = float(np.std(ruido_relativo, ddof=1))

    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": "#FCFCFD",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D7DBE7",
            "grid.color": "#E6E8F0",
            "font.family": "sans-serif",
        },
    )
    fig, eixos = plt.subplots(1, 2, figsize=(14, 5.8))
    cor_base, cor_contorno, cor_teorica = "#A3BEFA", "#2E4780", "#804126"

    sns.histplot(
        ruido_absoluto,
        bins="fd",
        stat="density",
        color=cor_base,
        edgecolor=cor_contorno,
        linewidth=0.7,
        ax=eixos[0],
    )
    eixos[0].axvline(0, color="#1F2430", linestyle=":", linewidth=1)
    eixos[0].set(
        title="Ruído absoluto",
        xlabel="série ruidosa − série original (unidade de preço)",
        ylabel="Densidade",
    )

    sns.histplot(
        ruido_relativo,
        bins="fd",
        stat="density",
        color=cor_base,
        edgecolor=cor_contorno,
        linewidth=0.7,
        label="Ruído observado",
        ax=eixos[1],
    )
    limite = max(abs(float(np.min(ruido_relativo))), abs(float(np.max(ruido_relativo))))
    x = np.linspace(-limite, limite, 500)
    densidade_normal = np.exp(-0.5 * (x / intensidade) ** 2) / (
        intensidade * np.sqrt(2.0 * np.pi)
    )
    eixos[1].plot(
        x,
        densidade_normal,
        color=cor_teorica,
        linewidth=1.2,
        label=f"Normal teórica: N(0, {intensidade:g}²)",
    )
    eixos[1].axvline(media, color="#1F2430", linestyle=":", linewidth=1)
    eixos[1].set(
        title="Ruído relativo e distribuição teórica",
        xlabel="série ruidosa / série original − 1",
        ylabel="Densidade",
    )
    eixos[1].legend(frameon=False, loc="upper left")
    eixos[1].text(
        0.98,
        0.96,
        f"n = {len(ruido_relativo):,}\nμ observado = {media:.2e}\n"
        f"σ observado = {desvio:.6f}\nσ configurado = {intensidade:.6f}",
        transform=eixos[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#1F2430",
        bbox={"facecolor": "white", "edgecolor": "#D7DBE7", "alpha": 0.92},
    )

    for eixo in eixos:
        eixo.spines[["top", "right"]].set_visible(False)
        eixo.grid(axis="y", alpha=0.8)
        eixo.grid(axis="x", visible=False)

    fig.suptitle(
        "Distribuição do ruído gaussiano injetado",
        x=0.07,
        y=0.99,
        ha="left",
        fontsize=14,
        fontweight="semibold",
        color="#1F2430",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(caminho, dpi=180, bbox_inches="tight")
    fig.savefig(caminho.with_suffix(".svg"), bbox_inches="tight")
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
