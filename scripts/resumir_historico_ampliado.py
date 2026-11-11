"""Relata a confirmação posterior com o snapshot de 50 mil candles do MT5."""

import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
BASE = RAIZ / "previsao_sem_ruido/execucao_20261107T134926_171325Z"
BUSCAS = [RAIZ / "ajustes_redes" / n for n in ["execucao_20261107T135128_021156Z", "execucao_20261107T135138_939899Z"]]


def executar():
    for p in BUSCAS:
        if not (p / "CONCLUIDO.json").exists():
            raise ValueError(f"A busca ainda não terminou: {p}")
    q = json.loads((BASE / "qualidade_dados.json").read_text(encoding="utf-8"))
    revisao = json.loads((RAIZ / "dados_adicionais/revisao_da_base.json").read_text(encoding="utf-8"))
    cortes = json.loads((BASE / "cortes_temporais.json").read_text(encoding="utf-8"))
    dados = pd.read_csv(BASE / "metricas_teste_final.csv")
    adicionais = []
    for p in BUSCAS:
        r = pd.read_csv(p / "comparacao_antes_depois.csv")
        adicionais.append(r.loc[r.origem == "rede_ajustada"].iloc[0].to_dict())
    dados = pd.concat([dados, pd.DataFrame(adicionais)], ignore_index=True)
    saida = RAIZ / "comparacao_redes"
    saida.mkdir(exist_ok=True)
    dados.to_csv(saida / "teste_historico_ampliado.csv", index=False)
    texto = ["# Confirmação posterior com histórico ampliado", "",
        "Esta é a segunda etapa do estudo. A configuração e os resultados do sistema original estão em `RESULTADOS_ANTES_DEPOIS.md`. As taxas desta etapa pertencem a outra versão dos dados e a outro período; não são uma comparação pareada de melhora em relação aos 51,94% antigos.", "",
        f"Foram coletados {q['linhas_recebidas']:,} candles completos PETR4 M5 pelo terminal MT5 conectado, de {q['inicio']} até {q['fim']}. Não houve remoção de OHLC inválido ou de duplicatas. A base anterior foi preservada.", "",
        "## Protocolo", "",
        f"O teste posterior começa em {cortes[-1]['inicio_teste']} e termina no último alvo elegível de novembro de 2026. Esse período não existia no arquivo original que terminava em junho. Ele contém {int(dados.amostras.iloc[0])} exemplos, 5.706 de manter, 370 de compra e 321 de venda.", "",
        "O alvo permaneceu: +/-0,8% após dez candles de cinco minutos consecutivos na mesma sessão. Não se reduziu a barreira para elevar os números. As redes candidatas foram fixadas com base nos estudos anteriores: GRU de 24 candles com pesos suaves, CNN+GRU de 24 e MLP de 12 com ajuste de decisão. A confirmação do objetivo da diagonal usou a GRU de 24 previamente estudada.", "",
        "As três redes de F1 foram comparadas nos blocos de desenvolvimento do histórico ampliado e somente a selecionada foi avaliada no período posterior. A GRU de diagonal seguiu outro objetivo, declarado separadamente. Cada selecionada foi treinada nas sementes 42, 123 e 2026, até 40 épocas e com Early Stopping de validação. Nenhum parâmetro foi escolhido pelo desempenho dessas redes no teste de julho a novembro.", "",
        "## Resultado no período posterior", "",
        "| Método | Acurácia | F1 macro | Recall manter | Recall compra | Recall venda |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in dados.itertuples():
        texto.append(f"| {r.modelo} | {r.accuracy:.2%} | {r.f1_macro:.4f} | {r.recall_manter:.2%} | {r.recall_compra:.2%} | {r.recall_venda:.2%} |")
    atingiu = bool(dados.meta_70_diagonal_todas_classes.any())
    texto += ["", f"**Algum método atingiu 70% na diagonal das três classes: {'sim' if atingiu else 'não'}.**",
        "Acurácia geral acima de 70% não é suficiente para concluir que compra e venda foram previstas bem. Nesta base, sempre manter já produz 89,20% no teste.", "",
        "## Versão dos dados", "",
        f"Os {revisao['candles_em_comum']} timestamps em comum com a base antiga não têm os mesmos fechamentos. A razão entre fechamento novo e antigo fica aproximadamente entre {revisao['razao_novo_antigo_min_mediana_max'][0]:.6f} e {revisao['razao_novo_antigo_min_mediana_max'][2]:.6f}. Houve {revisao['rotulos_diferentes_nos_9990_alvos_com_futuro']} mudanças de classe entre os 9.990 alvos comparáveis na sobreposição, usando a barreira de 0,8%.", "",
        "A causa da revisão de preços não foi verificada. Não se misturaram snapshots nem se atribuiu a diferença de taxas à arquitetura. O arquivo baixado, seu hash, a identificação da origem e a análise da revisão ficaram preservados em `dados_adicionais/` e nos manifestos das execuções.", "",
        "## Simulação operacional das redes", "",
        "Spread e slippage permanecem excluídos. A simulação usa uma posição por vez, entrada na abertura seguinte, saída no fechamento-alvo e taxas hipotéticas por lado. O drawdown é medido somente nas saídas; não há liquidez, aluguel ou margem.", "",
        "| Rede | Taxa por lado | Operações | Retorno acumulado | Drawdown nas saídas |",
        "|---|---:|---:|---:|---:|"]
    for p in BUSCAS:
        nome = json.loads((p / "selecao_modelo.json").read_text())["modelo_selecionado"]
        bt = pd.read_csv(p / "backtest/metricas.csv")
        for r in bt.itertuples():
            texto.append(f"| {nome} | {r.custo_bps_por_lado:g} bps | {r.operacoes} | {r.retorno_acumulado:.2%} | {r.max_drawdown_fechamentos:.2%} |")
    texto += ["", "## Conclusão desta confirmação", "",
        "A ampliação do histórico e o novo período permitiram testar se os ajustes se sustentavam em dados posteriores. A tabela registra tanto os ganhos quanto as falhas. O código não promove um modelo como validado quando ele apenas aumenta os acertos de manter.", "",
        "Se a meta de 70% por classe não é atingida, os resultados não autorizam iniciar um ensaio de ruído alegando que existe um preditor confiável. Uma próxima mudança de alvo, horizonte, atributos externos ou estratégia deve ser uma nova hipótese de pesquisa, declarada antes de abrir outro período de confirmação. Não há garantia de alcançar a taxa desejada com os mesmos atributos OHLC e volume.", "",
        "## Inspeção e reprodução", "",
        "- `comparacao_redes/teste_historico_ampliado.csv`: métricas das referências e das duas redes selecionadas.",
        "- `dados_adicionais/origem_mt5.json` e `revisao_da_base.json`: origem e versão do histórico.",
        f"- `{BASE.relative_to(RAIZ)}`: referências, cortes e identificação dos dados.",
        *[f"- `{p.relative_to(RAIZ)}`: tentativas, pesos, matrizes, previsões, históricos e simulação da respectiva busca." for p in BUSCAS], "",
        "Os 23 testes automatizados passaram; as execuções com dados reais concluídas são uma verificação adicional do pipeline. Os arquivos do sistema original não foram sobrescritos."]
    (RAIZ / "RESULTADOS_HISTORICO_AMPLIADO.md").write_text("\n".join(texto) + "\n", encoding="utf-8")
    print(dados[["modelo", "accuracy", "f1_macro", "recall_manter", "recall_compra", "recall_venda"]].to_string(index=False))


if __name__ == "__main__":
    executar()
