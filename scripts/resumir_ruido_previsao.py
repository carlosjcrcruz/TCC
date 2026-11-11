"""Consolida achados e verificação dos ensaios de ruído já concluídos."""
import argparse
import json
from pathlib import Path
import re
import sys

import pandas as pd

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))


def gerar(pasta,arquivo):
    pasta=Path(pasta)
    fim=json.loads((pasta/"CONCLUIDO.json").read_text(encoding="utf-8"))
    if not fim["concluido"]:
        raise ValueError("O ensaio ainda não concluiu.")
    plano=json.loads((pasta/"plano_congelado.json").read_text(encoding="utf-8"))
    filtros=json.loads((pasta/"filtros_selecionados.json").read_text(encoding="utf-8"))
    resumo=pd.read_csv(pasta/"resumo_sementes.csv")
    ic=pd.read_csv(pasta/"intervalos_pareados.csv")
    dados=pd.read_csv(pasta/"resultados_cenarios.csv")
    sigma=plano["plano"]["intensidade_filtros"]
    escopo="rede_evento" if "rede_evento" in resumo.escopo.unique() else resumo.escopo.iloc[0]
    bloco=resumo[resumo.escopo.eq(escopo)]
    original=bloco[bloco.intensidade.eq(0)&bloco.tratamento.eq("sem_tratamento")].iloc[0]
    alvo=bloco[(bloco.intensidade.eq(sigma))]
    if alvo.empty:
        sigma=float(bloco[bloco.intensidade.gt(0)].intensidade.iloc[0])
        alvo=bloco[bloco.intensidade.eq(sigma)]
    pct=lambda x:"NA" if pd.isna(x) else f"{x:.2%}"
    texto=["# Resultado dos experimentos novos de ruído", "",
        f"Foram executadas **{len(dados)} avaliações financeiras pareadas** em {len(plano['plano']['escopos'])} escopos, com {len(plano['plano']['sementes'])} sementes de ruído. O cenário original reproduziu a referência: compra {original.compra_media:.2%}, venda {original.venda_media:.2%}, retorno {original.retorno_medio:.2%}. A rede, o scaler e os limiares ficaram congelados.", "",
        f"A fonte padrão é o snapshot MT5 de PETR4 M5. Usa-se seu passado para inicializar os indicadores e escolher os filtros antes de {plano['fim_treino_filtros']}; a avaliação vai de {plano['plano']['inicio']} até antes de {plano['plano']['fim']}. Os preços posteriores ao fim não entram neste ensaio. Há **{fim['amostras_comuns']} amostras comuns**; quantidade de amostras não é quantidade de operações, pois há abstenção e controle de sobreposição.", "",
        f"## Comparação principal — sigma {sigma:g} ({sigma*100:g}%), escopo `{escopo}`", "",
        "Médias entre sementes sobre o mesmo período. Não se somam as repetições como se fossem mercados independentes. O escopo inclui perturbação do previsor e do evento, com risco e execução mantidos no referencial original.", "",
        "| Condição | Compras médias | % lucrativas compra | Vendas médias | % lucrativas venda | Retorno médio | Sementes com suporte mínimo |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    linhas=[("Original, sem ruído ou tratamento",original)]
    for tratamento in ["sem_tratamento","EMA","Wavelet"]:
        r=alvo[alvo.tratamento.eq(tratamento)].iloc[0]
        linhas.append(("Ruído" if tratamento=="sem_tratamento" else f"Ruído + {tratamento}",r))
    for nome,r in linhas:
        texto.append(f"| {nome} | {r.compras_media:.1f} | {pct(r.compra_media)} | {r.vendas_media:.1f} | {pct(r.venda_media)} | {r.retorno_medio:.2%} | {r.sementes_suporte}/{r.execucoes} |")
    ruido=alvo[alvo.tratamento.eq("sem_tratamento")].iloc[0]
    texto += ["", "## Reconstrução da entrada e previsão são resultados distintos", "",
        "| Tratamento | RMSE relativo médio do fechamento | Redução contra o ruído sem tratamento |",
        "|---|---:|---:|"]
    for tratamento in ["sem_tratamento","EMA","Wavelet"]:
        r=alvo[alvo.tratamento.eq(tratamento)].iloc[0]
        reducao=1-r.rmse_relativo/ruido.rmse_relativo
        texto.append(f"| {tratamento} | {r.rmse_relativo:.4%} | {reducao:.2%} |")
    texto += ["", "As reduções acima comparam as médias do erro de reconstrução, não a remoção comprovada do ruído natural do mercado. Melhor reconstrução não garante melhores sinais ou retorno. Como a rede foi treinada em entradas originais, tratar suas entradas também muda a distribuição que ela recebe. O resultado é específico ao modelo congelado; não demonstra que EMA/Wavelet sejam inúteis em qualquer pipeline treinado para usá-las.", "",
        "## Diferenças de taxa contra a mesma realização sem tratamento", "",
        "Intervalos descritivos de 95% por reamostragem pareada de sessões, expressos em pontos percentuais. Não corrigem a multiplicidade de comparações nem a seleção histórica do modelo.", "",
        "| Tratamento | Direção | Limite inferior da diferença | Limite superior |",
        "|---|---|---:|---:|"]
    q=ic[ic.escopo.eq(escopo)&ic.intensidade.eq(sigma)&ic.comparacao.eq("mesmo_ruido_sem_tratamento")&ic.tratamento.ne("sem_tratamento")]
    for r in q.itertuples():
        texto.append(f"| {r.tratamento} | {r.direcao} | {r.delta_taxa_inferior*100:.2f} p.p. | {r.delta_taxa_superior*100:.2f} p.p. |")
    texto += ["", f"Parâmetros escolhidos novamente no treino: `{filtros}`.", "",
        f"Em todas as condições, {int(dados.meta_70_lucro_ambas.sum())}/{len(dados)} avaliações atingiram 70% nas duas direções com suporte mínimo; {int((dados.meta_70_lucro_ambas & dados.retorno_acumulado.gt(0)).sum())} dessas tiveram retorno acumulado positivo. **{int(dados.aprovado.sum())} avaliações cumpriram também o critério econômico completo**, que exige lucro médio positivo em cada direção e retorno acumulado positivo. As médias das condições continuam negativas. As realizações compartilham datas e não são confirmações independentes.", "",
        "Não houve retreinamento de uma rede por cenário. Este é um ensaio de sensibilidade a erro de observação, com preços, ATR de risco e respostas financeiras comuns. O escopo `rede` mantém também o evento original; `rede_evento` recalcula o evento, mas não contamina todos os subsistemas da plataforma.", "",
        "O histórico já foi inspecionado. Nenhuma realização de ruído é promovida como estratégia validada, e `meta_confirmada` permanece falsa. Os controles de EMA e Wavelet sem ruído artificial, as quatro intensidades e os dois escopos estão nas tabelas completas abaixo.", "",
        "## Verificação e arquivos", "",
        f"A auditoria confirmou modelo, scaler e política inalterados, referência reproduzida e {fim['amostras_comuns']} amostras com os mesmos alvos e preços de execução. Os CSVs registram cada semente, operação potencial, sinal e operação efetivamente executada.", ""]
    log=(RAIZ/"TESTES_PREVISAO.log").read_text(encoding="utf-8")
    testes=re.search(r"Ran (\d+) tests in ([\d.]+)s\s+OK",log)
    if testes:
        texto += [f"**{testes[1]} testes automatizados passaram**, em {float(testes[2]):.1f} segundos. Isso verifica o código, não certifica lucro ou taxa futura.", ""]
    relativo=pasta.relative_to(RAIZ) if pasta.is_absolute() and pasta.is_relative_to(RAIZ) else pasta
    texto += [f"Execução: `{relativo.as_posix()}`.", "",
        f"![Taxas de lucro com ruído e tratamento]({relativo.as_posix()}/taxas_rede_evento.png)", "",
        f"![Erro de reconstrução e retorno]({relativo.as_posix()}/reconstrucao_e_retorno.png)", "",
        "---", "", (pasta/"LEIA_RESULTADOS.md").read_text(encoding="utf-8")]
    Path(arquivo).write_text("\n".join(texto)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("pasta",type=Path)
    p.add_argument("--arquivo",type=Path,default=RAIZ/"RESULTADOS_RUIDO_NOVA_PREVISAO.md")
    a=p.parse_args()
    gerar(a.pasta,a.arquivo)
    print(a.arquivo)
