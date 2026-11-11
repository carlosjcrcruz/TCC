r"""Reavalia as previsões existentes sem treinar redes ou alterar o experimento.

Executar na raiz: .\.venv\Scripts\python.exe scripts\auditar_predicao.py
Saídas isoladas em auditoria_predicao/. Os parâmetros são os de config.py;
o script verifica se eles reproduzem os alvos e cortes dos arquivos salvos.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from config import ConfigProjeto
from src.sequences import calcular_limites_temporais


def ler(caminho: Path) -> pd.DataFrame:
    dados = pd.read_csv(caminho)
    dados["time"] = pd.to_datetime(dados["time"], utc=True)
    if dados["time"].duplicated().any() or not dados["time"].is_monotonic_increasing:
        raise ValueError(f"Datas duplicadas ou fora de ordem: {caminho}")
    return dados


def metricas(nome: str, real: np.ndarray, previsto: np.ndarray) -> dict:
    matriz = confusion_matrix(real, previsto, labels=[0, 1, 2])
    registro = {
        "cenario": nome,
        "amostras": len(real),
        "accuracy": float(accuracy_score(real, previsto)),
        "f1_macro": float(f1_score(real, previsto, labels=[0, 1, 2], average="macro", zero_division=0)),
        "recall_macro": float(recall_score(real, previsto, labels=[0, 1, 2], average="macro", zero_division=0)),
    }
    for classe, nome_classe in [(1, "compra"), (2, "venda")]:
        tp = int(matriz[classe, classe])
        sinais = int(matriz[:, classe].sum())
        fp = sinais - tp
        negativos = int(matriz.sum() - matriz[classe, :].sum())
        registro[f"sinais_{nome_classe}"] = sinais
        registro[f"acertos_{nome_classe}"] = tp
        # Ausência de sinais não demonstra precisão perfeita: FDR indefinida.
        registro[f"FDR_{nome_classe}"] = fp / sinais if sinais else None
        registro[f"FPR_{nome_classe}"] = fp / negativos if negativos else None
    return registro


def auditar(base: Path, saida: Path) -> None:
    config = ConfigProjeto()
    bruto = ler(base / "dados_processados/dados_originais.csv")
    dados = ler(base / "dados_processados/original.csv")
    predicoes = ler(base / "resultados/predicoes/predicoes_original.csv")
    h = config.horizonte_futuro
    limites = calcular_limites_temporais(len(dados), config.proporcao_treino, config.proporcao_validacao)
    treino = dados.iloc[config.tamanho_janela - 1 : limites.fim_treino - h]
    teste = dados.iloc[limites.fim_validacao :]
    if treino.empty or teste.empty:
        raise ValueError("Treino ou teste vazios com a configuração atual.")
    if not np.array_equal(teste.time.to_numpy(), predicoes.time.to_numpy()):
        raise ValueError("Cortes atuais não reproduzem as datas de teste salvas.")
    real = predicoes.classe_real.to_numpy(dtype=int)
    if not np.array_equal(real, teste.rotulo.to_numpy(dtype=int)):
        raise ValueError("Rótulos das previsões não correspondem ao dataset.")

    retorno = bruto.close.shift(-h) / bruto.close - 1.0
    alvo = np.select([retorno >= config.limiar_retorno, retorno <= -config.limiar_retorno], [1, 2], default=0)
    referencias = pd.DataFrame({"time": bruto.time, "retorno": retorno, "rotulo_recalculado": alvo})
    conferencia = dados.merge(referencias, on="time", validate="one_to_one")
    if len(conferencia) != len(dados) or not np.allclose(conferencia.retorno_futuro, conferencia.retorno):
        raise ValueError("Horizonte atual não reproduz os retornos futuros salvos.")
    if not np.array_equal(conferencia.rotulo, conferencia.rotulo_recalculado):
        raise ValueError("Limiar atual não reproduz as classes salvas.")

    majoritaria = int(treino.rotulo.value_counts().idxmax())
    registros = [metricas("Baseline: classe majoritária do treino", real, np.full(len(real), majoritaria))]
    comparacao_salva = pd.read_csv(base / "resultados/comparacao_cenarios.csv").set_index("cenario")
    for caminho in sorted((base / "resultados/predicoes").glob("predicoes_*.csv")):
        p = ler(caminho)
        if not np.array_equal(p.time.to_numpy(), predicoes.time.to_numpy()) or not np.array_equal(p.classe_real, real):
            raise ValueError(f"Cenário sem datas/rótulos comuns: {caminho.name}")
        slug = caminho.stem.removeprefix("predicoes_")
        metadados = json.loads((base / f"resultados/metricas/metricas_{slug}.json").read_text(encoding="utf-8"))
        probabilidades = p[["prob_manter", "prob_compra", "prob_venda"]].to_numpy()
        if not np.isfinite(probabilidades).all() or (probabilidades < 0).any() or not np.allclose(probabilidades.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError(f"Probabilidades inválidas: {caminho.name}")
        if not np.array_equal(np.argmax(probabilidades, axis=1), p.classe_predita):
            raise ValueError(f"Argmax diferente da classe salva: {caminho.name}")
        registro = metricas(metadados["cenario"], real, p.classe_predita.to_numpy(dtype=int))
        for coluna in ["accuracy", "f1_macro", "recall_macro", "FDR_compra", "FDR_venda", "FPR_compra", "FPR_venda"]:
            if registro[coluna] is not None and not np.isclose(registro[coluna], comparacao_salva.loc[registro["cenario"], coluna]):
                raise ValueError(f"Métrica divergente: {registro['cenario']} / {coluna}")
        registros.append(registro)

    local = bruto.time.dt.tz_convert("America/Sao_Paulo")
    destino = local.shift(-h)
    valido = destino.notna()
    atravessa = local.dt.date != destino.dt.date
    minutos = (destino - local).dt.total_seconds() / 60
    horizontes = pd.DataFrame({"time": bruto.time[valido], "atravessa_data_local": atravessa[valido], "duracao_minutos": minutos[valido], "classe_real": alvo[valido]})
    horizontes_teste = predicoes[["time"]].merge(horizontes, on="time", validate="one_to_one")
    ruido = ler(base / "dados_processados/ruido_0_002.csv")
    if not np.array_equal(ruido.time.to_numpy(), dados.time.to_numpy()):
        raise ValueError("Dataset ruidoso com datas diferentes do original.")
    primeiro_validacao = dados.time.iloc[limites.fim_treino]
    corte_filtro = max(2, int(len(bruto) * config.proporcao_treino))
    resumo = {
        "escopo": "Auditoria retrospectiva dos arquivos existentes; não é novo teste independente nem prova de desempenho futuro.",
        "parametros_conferidos": {"horizonte": h, "limiar": config.limiar_retorno, "janela": config.tamanho_janela, "proporcao_treino": config.proporcao_treino, "proporcao_validacao": config.proporcao_validacao},
        "dados_brutos": {"quantidade": len(bruto), "inicio": str(bruto.time.iloc[0]), "fim": str(bruto.time.iloc[-1])},
        "treino_efetivo": {"quantidade": len(treino), "contagens_classes": {str(k): int(v) for k, v in treino.rotulo.value_counts().sort_index().items()}, "classe_majoritaria": majoritaria},
        "teste": {"quantidade": len(real), "contagens_classes": {str(k): int((real == k).sum()) for k in [0, 1, 2]}},
        "horizontes": {"rotulos_validos": int(valido.sum()), "atravessam_data_local": int((atravessa & valido).sum()), "minutos_maximo": float(minutos.max()), "direcionais_total": int(((alvo != 0) & valido).sum()), "direcionais_atravessam_data_local": int(((alvo != 0) & valido & atravessa).sum()), "teste_direcionais_atravessam_data_local": int(((horizontes_teste.classe_real != 0) & horizontes_teste.atravessa_data_local).sum())},
        "perturbacao_close": {"valores_fora_intervalo_high_low_original": int(((ruido.close_modelo > ruido.high) | (ruido.close_modelo < ruido.low)).sum()), "linhas_validas": len(ruido), "colunas_identicas": {c: bool(np.array_equal(dados[c], ruido[c])) for c in ["open", "high", "low", "tick_volume", "rotulo"]}},
        "fronteira_filtros": {"primeira_data_validacao_modelo": str(primeiro_validacao), "ultima_data_selecao_filtro": str(bruto.time.iloc[corte_filtro - 1]), "linhas_selecao_filtro_na_validacao_modelo": int((bruto.time.iloc[:corte_filtro] >= primeiro_validacao).sum())},
        "verificacoes": "Datas, alvos, probabilidades, argmax e métricas dos cenários conferidos; modelos não foram retreinados.",
        "limitacao": "Não há manifesto da execução antiga; a configuração atual foi conferida contra os alvos e cortes salvos, não contra todos os parâmetros do treinamento antigo.",
    }
    saida.mkdir(parents=True, exist_ok=True)
    tabela = pd.DataFrame(registros)
    tabela.to_csv(saida / "comparacao_com_baseline.csv", index=False)
    horizontes.to_csv(saida / "horizontes_reais.csv", index=False)
    (saida / "resumo_auditoria.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(tabela[["cenario", "accuracy", "f1_macro", "recall_macro"]].to_string(index=False))
    print(f"\nAuditoria salva em: {saida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=RAIZ)
    parser.add_argument("--saida", type=Path, default=RAIZ / "auditoria_predicao")
    args = parser.parse_args()
    auditar(args.base.resolve(), args.saida.resolve())
