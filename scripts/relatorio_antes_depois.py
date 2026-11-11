"""Consolida resultados executados e a configuração verificável do modelo antigo."""

import json
from pathlib import Path
import zipfile

import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
BASE = RAIZ / "previsao_sem_ruido/execucao_20261107T131806_643737Z"
RODADAS = [RAIZ / "ajustes_redes" / n for n in [
    "execucao_20261107T133007_451445Z", "execucao_20261107T133432_180842Z", "execucao_20261107T133952_080481Z"]]


def executar():
    saida = RAIZ / "comparacao_redes"
    saida.mkdir(exist_ok=True)
    antigo = json.loads((RAIZ / "resultados/metricas/metricas_original.json").read_text(encoding="utf-8"))
    auditoria = json.loads((RAIZ / "auditoria_predicao/resumo_auditoria.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(RAIZ / "resultados/modelos/modelo_original.keras") as z:
        arquitetura = json.loads(z.read("config.json"))
    historico = pd.read_csv(RAIZ / "resultados/historicos_treinamento/historico_original.csv")
    config_antiga = {
        "arquitetura_verificada_no_keras": [{"classe": l["class_name"], "config": l["config"]} for l in arquitetura["config"]["layers"]],
        "compilacao_verificada_no_keras": arquitetura["compile_config"],
        "configuracao_declarada_no_codigo_e_tcc": {"janela": 60, "horizonte": 10, "limiar": .008,
            "max_epocas": 50, "batch_size": 64, "paciencia": 8, "monitor": "val_loss", "seed": 42, "pesos_classes": True},
        "epocas_registradas": len(historico), "melhor_epoca_val_loss": int(historico.val_loss.idxmin() + 1),
        "pesos_reconstituidos_do_treino": {k: auditoria["treino_efetivo"]["quantidade"] / (3*v) for k, v in auditoria["treino_efetivo"]["contagens_classes"].items()},
        "limitacao": "A arquitetura/perda/otimizador foram lidos do modelo salvo. Os demais parâmetros vêm da configuração e do TCC; não havia manifesto completo da execução antiga.",
    }
    (saida / "configuracao_antiga.json").write_text(json.dumps(config_antiga, ensure_ascii=False, indent=2), encoding="utf-8")
    original = {"modelo": "LSTM antiga no teste original", "amostras": 1496, "accuracy": antigo["accuracy"], "f1_macro": antigo["f1_macro"],
                **{f"recall_{n}": antigo["matriz_confusao"][i][i]/sum(antigo["matriz_confusao"][i]) for i, n in enumerate(["manter", "compra", "venda"])},
                "universo": "teste_original_com_alvos_entre_pregoes"}
    comparacao = pd.read_csv(RODADAS[-1] / "comparacao_antes_depois.csv")
    comparacao["universo"] = "mesmas_1304_datas_intradiarias"
    linhas = [original]
    linhas += comparacao.to_dict("records")
    for pasta in RODADAS[:-1]:
        t = pd.read_csv(pasta / "comparacao_antes_depois.csv")
        linhas.append({**t.loc[t.origem == "rede_ajustada"].iloc[0].to_dict(), "universo": "mesmas_1304_datas_intradiarias"})
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(saida / "comparacao_antes_depois.csv", index=False)
    tentativas = pd.concat([pd.read_csv(p / "tentativas_desenvolvimento.csv").assign(rodada=i+1) for i, p in enumerate(RODADAS)])
    tentativas.to_csv(saida / "todas_tentativas_desenvolvimento.csv", index=False)
    resumo = tentativas.groupby(["rodada", "modelo"])[["accuracy", "f1_macro", "recall_manter", "recall_compra", "recall_venda"]].mean().reset_index()
    resumo.to_csv(saida / "resumo_configuracoes.csv", index=False)
    bt = pd.read_csv(RODADAS[1] / "backtest/metricas.csv")
    texto = ["# Resultados antes e depois dos ajustes", "",
        "**A implementação e os testes foram concluídos, mas 70% de acerto em cada classe não foi alcançado.** Houve configurações com acurácia total acima de 70%; elas continuaram falhando em compra e venda. Não é correto apresentá-las como resolução da exigência da diagonal.", "",
        "## Como a rede antiga estava configurada", "",
        "A leitura do arquivo `resultados/modelos/modelo_original.keras` confirma: entrada de 60 candles e 10 atributos, LSTM com 64 unidades, Dropout de 0,30, Dense de 32 unidades com ReLU, outro Dropout de 0,30 e saída de três classes com softmax. O otimizador era Adam com taxa aproximada de 0,001, perda categorical crossentropy e métrica accuracy.", "",
        f"O histórico salvo contém {len(historico)} épocas. A menor val_loss ocorreu na época {int(historico.val_loss.idxmin()+1)}. O código/TCC declaram até 50 épocas, batch de 64, Early Stopping por val_loss com paciência 8 e semente 42. Essa parte da configuração não tinha um manifesto completo; foi preservada em `comparacao_redes/configuracao_antiga.json` com a distinção entre o que o modelo comprova e o que está declarado.", "",
        "Os atributos antigos eram abertura, máxima, mínima, fechamento, volume de ticks, retorno percentual, SMA, EMA, volatilidade e RSI. O alvo era o retorno após dez candles com barreiras de +/-0,8%, incluindo casos entre pregões. O treino usava pesos inversos à frequência das classes.", "",
        "## Resultado antigo", "",
        f"No teste original de 1.496 exemplos: acurácia {antigo['accuracy']:.2%}, F1 macro {antigo['f1_macro']:.4f}, recall de manter {original['recall_manter']:.2%}, compra {original['recall_compra']:.2%} e venda {original['recall_venda']:.2%}.", "",
        "A rede emitiu 518 compras, das quais 34 coincidiram com a classe real, e 219 vendas, das quais 11 coincidiram com a classe real. Assim, a precisão foi 6,56% em compra e 5,02% em venda. Um falso positivo desta tarefa significa não atingir o evento de 0,8%; não é necessariamente uma operação com prejuízo.", "",
        "## Comparação nas mesmas datas", "",
        "Para comparar o desempenho, as previsões da rede antiga foram também recortadas nas mesmas 1.304 datas intradiárias do teste novo. Os rótulos coincidem nessas datas. O treino e a representação mudaram em conjunto; isto não é uma ablação causal de um único componente.", "",
        "| Método nas mesmas 1.304 datas | Acurácia | F1 macro | Recall manter | Recall compra | Recall venda |",
        "|---|---:|---:|---:|---:|---:|"]
    desejados = ["LSTM_legada_nas_mesmas_datas", "majoritaria", "lstm", "mlp12_decisao", "cnn_gru24", "gru24_diagonal"]
    nomes = {"LSTM_legada_nas_mesmas_datas": "LSTM antiga", "majoritaria": "Sempre manter", "lstm": "LSTM nova inicial",
             "mlp12_decisao": "MLP ajustada por F1", "cnn_gru24": "CNN+GRU ajustada por F1", "gru24_diagonal": "GRU ajustada pelo equilíbrio da diagonal"}
    for nome in desejados:
        r = tabela.loc[tabela.modelo == nome].iloc[0]
        texto.append(f"| {nomes[nome]} | {r.accuracy:.2%} | {r.f1_macro:.4f} | {r.recall_manter:.2%} | {r.recall_compra:.2%} | {r.recall_venda:.2%} |")
    texto += ["", "A MLP aumentou a acurácia total e a F1 macro na comparação com a rede antiga nessas datas, mas não identificou nenhuma venda. A CNN+GRU de F1 convergiu para manter em todo o teste. A GRU escolhida para equilibrar a diagonal também não chegou a 70% por classe. Nenhuma dessas redes foi certificada para servir de referência preditiva confiável ao experimento de ruído.", "",
        "## Ajustes que efetivamente foram testados", "",
        "Foram 19 configurações em três blocos temporais de desenvolvimento: 57 avaliações de desenvolvimento. Os selecionados de cada uma das três rodadas foram treinados novamente em três sementes (42, 123 e 2026), agregadas por média. Juntando os dois tipos de LSTM da avaliação inicial, houve 90 treinamentos de redes nos experimentos com a base real, sem contar os testes automatizados sintéticos.", "",
        "- LSTM, GRU, CNN, CNN+GRU, MLP e convolução com dilatação.",
        "- Janelas de 12, 24 e 60 candles; redes de 16, 32 e 64 unidades, conforme a variante.",
        "- Taxas de aprendizado de 0,001 e 0,0003, clipping de gradiente e regularização.",
        "- Sem pesos, pesos suavizados e pesos inversos à frequência das classes; perda focal em uma variante.",
        "- Atributos relativos ao preço e ao histórico, no lugar da dependência de níveis brutos de preço.",
        "- Mistura somente dos minibatches do treino, mantendo a ordem dentro de cada janela e os cortes temporais.",
        "- Regra de decisão ajustada na validação e persistida junto ao modelo.",
        "- Seleção por F1 macro em duas rodadas e seleção pelo menor recall médio entre as classes na terceira, explicitamente tratada como outro objetivo.", "",
        "A MLP de F1 usa 12 candles e 18 atributos, Flatten, Dense de 32 unidades, Dropout de 0,20, Dense de 16 e softmax de três saídas. Sua taxa é 0,0003, os pesos são balanceados e a decisão final multiplica os escores por [1, 0.5, 0.25], fator escolhido na validação. Esse ajuste não é calibração probabilística.", "",
        "## Por que a taxa antiga não chegou a 70%", "",
        "**Desbalanceamento e excesso de sinais.** No teste original, 90,84% dos exemplos eram manter. A rede emitiu compra/venda em 737 exemplos, embora só 137 exemplos pertencessem a essas classes. A maioria dos sinais operacionais não coincidia com o evento futuro definido. Aumentar o peso das classes raras não criou informação que permitisse separá-las.", "",
        "**Evento raro e horizonte misturado.** O limiar de 0,8% exigia movimentos grandes para o horizonte descrito. Na base bruta, 621 dos 1.455 alvos direcionais atravessavam datas de pregão (42,68%). O texto os tratava como aproximadamente 50 minutos. Essa mistura foi corrigida. No teste intradiário novo restaram 32 compras e 38 vendas em 1.304 exemplos: apenas 5,37% direcionais. Por isso, a regra sempre manter sobe para 94,63% sem antecipar nenhuma compra/venda.", "",
        "**Critério de treinamento diferente da meta por classe.** O modelo antigo selecionava pesos por val_loss sem exigir recall de 70% nas três classes. A comparação demonstrou que otimizar F1 ou acurácia pode continuar favorecendo manter. A rodada específica para a diagonal corrigiu esse objetivo de seleção, mas também falhou fora do treino.", "",
        "**Configuração fixa sem validação de robustez.** O resultado antigo vinha de uma configuração e uma semente, sem referências simples e sem os novos blocos de avaliação. Esses controles foram acrescentados. As tentativas mostraram que a arquitetura sozinha não resolveu o problema nesta base.", "",
        "Esses fatos explicam a falha observada e as limitações da avaliação antiga. Não demonstram uma causa única nem provam que 70% seja matematicamente impossível. Também não há evidência para prometer que mais camadas ou mais épocas alcançarão a meta.", "",
        "## Simulação operacional", "",
        "A simulação usa uma posição por vez, entrada na próxima abertura e saída no fechamento-alvo. Spread e slippage foram excluídos conforme solicitado. Os custos abaixo são cenários hipotéticos, não valores medidos da corretora.", "",
        "| MLP ajustada | Operações | Retorno acumulado | Drawdown nas saídas | Operações positivas |",
        "|---|---:|---:|---:|---:|"]
    for r in bt.itertuples():
        texto.append(f"| {r.custo_bps_por_lado:g} bps por lado | {r.operacoes} | {r.retorno_acumulado:.2%} | {r.max_drawdown_fechamentos:.2%} | {r.taxa_operacoes_positivas:.2%} |")
    texto += ["", "Esse retorno positivo em alguns cenários não comprova a meta de classificação e não estabelece rentabilidade futura. São apenas 26 operações simplificadas, sem margem, aluguel, impacto de mercado ou drawdown intraposição.", "",
        "## Validação e limite atual", "",
        "Os 23 testes automatizados passaram após as alterações finais. Eles cobrem causalidade, alvos intradiários contínuos, expurgo por timestamp de término, inferência sem resposta futura, persistência dos modelos e ajustes de decisão, arquiteturas novas e ausência de sobreposição de operações. O pipeline antigo também foi testado com dados sintéticos.", "",
        "O período final da base antiga já havia sido examinado; os resultados desta primeira etapa são retrospectivos. As configurações de cada rodada foram escolhidas no desenvolvimento, mas o desenho das rodadas foi refinado após examinar resultados anteriores. Portanto, não se trata de confirmação independente.", "",
        "A primeira tentativa de acesso ao MT5 retornou IPC timeout. Depois que o terminal ficou disponível, foram coletados 50 mil candles. Eles pertencem a outro snapshot, preservado separadamente. A confirmação posterior está em `RESULTADOS_HISTORICO_AMPLIADO.md`; os números deste documento continuam correspondendo à primeira etapa com 10 mil candles.", "",
        "Uma mudança do evento, horizonte ou informações externas constitui uma nova hipótese. Alterar o alvo produz outro experimento e deve ser identificado como tal. A busca não foi transformada em uma alegação de 70% de previsão útil.", "",
        "## Arquivos para inspeção", "",
        "- `comparacao_redes/configuracao_antiga.json`: configuração do modelo anterior, com origem de cada informação.",
        "- `comparacao_redes/comparacao_antes_depois.csv`: resultados antigos e novos, identificando o universo avaliado.",
        "- `comparacao_redes/todas_tentativas_desenvolvimento.csv`: as 57 avaliações de desenvolvimento.",
        "- `comparacao_redes/resumo_configuracoes.csv`: médias das 19 configurações, mantendo os critérios/rodadas separados.",
        "- `ajustes_redes/`: pesos, previsões, históricos, matrizes, simulações e cópias do código usado em cada rodada.",
        "- `PREVISAO_SEM_RUIDO.md`: comandos para executar a previsão e novas buscas."]
    (RAIZ / "RESULTADOS_ANTES_DEPOIS.md").write_text("\n".join(texto) + "\n", encoding="utf-8")
    print("Relatório e tabelas consolidados.")


if __name__ == "__main__":
    executar()
