"""Audita seleção congelada e publica resultados sem selecionar pelo teste."""
import argparse
import json
import re
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from src.binary_events import FEATURES_EVENTOS, PlanoBinario, criar_contexto, prever_scores
from src.binary_study import _entradas, _partes
from src.bracket_strategy import (SaidaBarreiras, avaliar_barreiras, criar_operacoes_barreiras,
    permissoes_estrategia, reservar_confirmacao)
from src.forecast_config import ConfigPrevisao
from src.forecast_data import limpar_candles
from src.forecast_experiment import salvar_json
from src.forecast_models import tensorflow
from src.trade_policy import aplicar_politica


def auditar_selecao(pasta):
    plano_doc=json.loads((pasta/"plano_congelado.json").read_text(encoding="utf-8"))
    dados=plano_doc["plano"]
    for k in ["horizontes","bandas","familias","eventos","sementes_finais"]:
        dados[k]=tuple(dados[k])
    plano=PlanoBinario(**dados)
    escolhido=json.loads((pasta/"escolha_antes_da_confirmacao.json").read_text(encoding="utf-8"))
    info=json.loads((pasta/"modelo_experimental/modelo_binario.json").read_text(encoding="utf-8"))
    bruto,_=limpar_candles(pd.read_csv(plano_doc["csv"]))
    bruto=bruto[bruto.time<pd.Timestamp(plano.fim_teste)].reset_index(drop=True)
    ctx=criar_contexto(bruto,ConfigPrevisao())
    matriz=ctx[list(FEATURES_EVENTOS)].to_numpy(dtype=np.float32)
    regra=SaidaBarreiras(**{k:escolhido[k] for k in ["horizonte","tipo","alvo","razao_stop"]})
    meta=criar_operacoes_barreiras(ctx,regra,plano.custo)
    ok=pd.Series(np.isfinite(matriz).all(axis=1)).rolling(12).sum().eq(12).to_numpy()
    meta=meta.loc[ok[meta.posicao_bruta]].reset_index(drop=True)
    selecao=meta.iloc[_partes(meta,plano)["selecao"]].reset_index(drop=True)
    artefato=pasta/"modelo_experimental"
    scaler=joblib.load(artefato/"scaler.joblib")
    normalizada=scaler.transform(matriz).astype(np.float32)
    x=_entradas(normalizada,selecao.posicao_bruta.to_numpy(),info["janela"])
    modelos=[joblib.load(artefato/f) if f.endswith("joblib") else tensorflow().keras.models.load_model(artefato/f,compile=False) for f in info["arquivos"]]
    p=prever_scores(info["familia"],modelos,x)
    masks=permissoes_estrategia(ctx.iloc[selecao.posicao_bruta],escolhido["evento"])
    sinais=aplicar_politica(p,info["politica"],*masks)
    m,usados=avaliar_barreiras(selecao,sinais,plano.minimo,plano.meta)
    for campo in ["operacoes_compra","lucrativas_compra","operacoes_venda","lucrativas_venda","retorno_acumulado"]:
        if not np.isclose(m[campo],escolhido[campo],rtol=1e-6,atol=1e-8):
            raise ValueError(f"Auditoria divergente em {campo}: {m[campo]} x {escolhido[campo]}")
    selecao["sinal"]=sinais
    selecao["operacao_executada"]=False
    selecao.loc[usados,"operacao_executada"]=True
    selecao["score_compra"],selecao["score_venda"]=p[:,1],p[:,2]
    selecao.to_csv(pasta/"operacoes_selecao_auditadas.csv",index=False)
    salvar_json(pasta/"auditoria_selecao.json",{"reproduziu_resultados":True,"metricas":m})
    status=json.loads((pasta/"status.json").read_text(encoding="utf-8"))
    if status["confirmacao_avaliada"]:
        # Registra também a execução inicial, anterior à inclusão do marcador.
        reservar_confirmacao(plano_doc["csv"],pasta.parent,plano,pasta,escolhido)
        teste=meta.iloc[_partes(meta,plano)["teste"]].reset_index(drop=True)
        mc,mv=permissoes_estrategia(ctx.iloc[teste.posicao_bruta],escolhido["evento"])
        referencias=[]
        # Mesma saída congelada. Estas referências diagnosticam o resultado;
        # não entram em nova seleção nem autorizam escolher outro modelo.
        for nome,sinais in [("sempre_compra",np.ones(len(teste),dtype=int)),
                            ("sempre_venda",np.full(len(teste),2)),
                            ("evento_sem_modelo",np.select([mc&~mv,mv&~mc],[1,2],default=0))]:
            metrica,_=avaliar_barreiras(teste,sinais,plano.minimo,plano.meta)
            referencias.append({"referencia":nome,**metrica})
        salvar_json(pasta/"referencias_confirmacao.json",referencias)
    return escolhido,status


def intervalo_por_sessao(predicoes, repeticoes=2000):
    dias=pd.Index(predicoes.sessao.unique())
    execs=predicoes[predicoes.operacao_executada].copy()
    rng=np.random.default_rng(42)
    amostras=rng.integers(0,len(dias),size=(repeticoes,len(dias)))
    out={"metodo":"bootstrap por sessão; 2.000 reamostragens; semente 42; percentis 2,5/97,5; intervalo descritivo, não corrigido pela busca"}
    for lado,nome in [(1,"compra"),(2,"venda")]:
        a=execs[execs.sinal==lado]
        n=a.groupby("sessao").size().reindex(dias,fill_value=0).to_numpy()
        k=a.assign(ganhou=a[f"retorno_liquido_{nome}"]>1e-12).groupby("sessao").ganhou.sum().reindex(dias,fill_value=0).to_numpy()
        nn=n[amostras].sum(axis=1)
        kk=k[amostras].sum(axis=1)
        taxas=kk[nn>0]/nn[nn>0]
        out[nome]=[float(x) for x in np.quantile(taxas,[.025,.975])] if len(taxas) else None
    return out


def gerar(pastas,arquivo):
    texto=["# Estratégias com alvo, stop e prazo máximo", "",
        "Foram implementadas e executadas duas buscas: previsão de lucro por classificação e previsão do retorno líquido por regressão. A meta de taxa de acerto continua sendo 70% em compras e vendas executadas, separadamente, com pelo menos vinte operações por direção. A aprovação econômica adicional exige retorno acumulado e lucro médio positivos em cada direção. Todos os valores são de simulação, líquidos de 2 pontos-base por lado, sem spread ou slippage.", "",
        "## Mudanças executadas", "",
        "A posição pode sair pelo alvo, pelo stop ou pelo fechamento no prazo máximo. Os limites são fixos ou baseados no ATR de vinte candles já fechados. Há seis condições causais de entrada: todos os candles, choque de reversão, RSI extremo, rompimento, reversão à média ponderada por ticks e recuo na tendência. Cada estratégia aprende alvos que correspondem à sua própria regra de saída; os resultados não são diretamente comparáveis à antiga saída sempre após cinquenta minutos.", "",
        "A entrada é no open seguinte. Quando o candle toca ambos os limites sem ordem conhecida, conta-se o stop. Um gap de stop executa no open, mesmo sendo pior; um gap de alvo recebe somente o preço limite. A avaliação respeita uma posição por vez até a saída efetiva. O prazo inteiro precisa ser contínuo na mesma sessão para a amostra ser elegível, mesmo quando a barreira sai antes. Esta é uma simulação com OHLC e preços contínuos do snapshot, sem modelar fila, liquidez, arredondamento de ordens ou microestrutura.", "",
        "A classificação compara boosting e MLP (duas saídas sigmoid); a regressão usa duas cabeças de boosting que aprendem retorno líquido. Seus escores são rankings, não probabilidades calibradas. A normalização usa só o treino. Os modelos usam semente 42 e permanecem congelados depois da seleção, sem reajuste ao ver o teste.", "",
        "## Protocolo temporal", "",
        "Treino de março/2022 até fevereiro/2023; Early Stopping em março–abril/2023; limiares em maio–agosto/2023; seleção em setembro–dezembro/2023. Alvos que atravessam fronteiras são expurgados. A confirmação reservada é janeiro–junho/2024. O histórico posterior a junho/2024 não entra no treinamento nem na escolha. O snapshot de 99.998 candles foi fornecido pelo MT5 em novembro/2026 e apresenta revisões do provedor em relação à base original.", "",
        "Os candidatos e a escolha são gravados antes da confirmação. Somente uma configuração aprovada pode abrir o período reservado. O marcador em `busca_estrategias/confirmacoes_utilizadas/` impede reabrir o mesmo snapshot/período dentro desta busca. Mudar a pasta de saída não torna um período já observado independente novamente.", "",
        "## Resultados das buscas", ""]
    confirmou=False
    for pasta in pastas:
        escolhido,status=auditar_selecao(pasta)
        tabela=pd.read_csv(pasta/"comparacao_validacao.csv")
        plano=json.loads((pasta/"plano_congelado.json").read_text(encoding="utf-8"))
        objetivo=plano.get("objetivo","taxa")
        texto += [f"### Busca `{pasta.name}` — objetivo `{objetivo}`", "",
            f"{len(plano['regras'])} regras de saída, {len(plano['plano']['familias'])} família(s), {len(plano['eventos'])} eventos: **{len(tabela)} candidatos**. Na seleção, {int(tabela.meta_70_lucro_ambas.sum())} atingiram 70% nas duas direções com suporte, e {int(tabela.aprovado.sum())} também passaram os critérios econômicos.", "",
            f"Escolhido: `{escolhido['familia']}`, evento `{escolhido['evento']}`, prazo {escolhido['horizonte']} candles, barreira `{escolhido['tipo']}`, alvo {escolhido['alvo']:g}, stop {escolhido['razao_stop']:g} vezes o alvo. Os limiares são {escolhido['limiar_compra']:.6f} para compra e {escolhido['limiar_venda']:.6f} para venda.", "",
            "| Etapa | Compras lucrativas / total | % compra | Vendas lucrativas / total | % venda | Retorno acumulado | Profit factor | Drawdown nas saídas |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
        linhas=[("Seleção",escolhido)]
        if status["confirmacao_avaliada"]:
            confirmou=True
            linhas.append(("Confirmação",json.loads((pasta/"resultado_confirmacao.json").read_text(encoding="utf-8"))))
            pred=pd.read_csv(pasta/"operacoes_confirmacao.csv")
            ci=intervalo_por_sessao(pred)
            salvar_json(pasta/"incerteza_confirmacao.json",ci)
        for etapa,r in linhas:
            pf="NA" if r['profit_factor'] is None else f"{r['profit_factor']:.3f}"
            texto.append(f"| {etapa} | {r['lucrativas_compra']}/{r['operacoes_compra']} | {r['taxa_lucro_compra']:.2%} | {r['lucrativas_venda']}/{r['operacoes_venda']} | {r['taxa_lucro_venda']:.2%} | {r['retorno_acumulado']:.2%} | {pf} | {r['drawdown_saidas']:.2%} |")
        texto += ["",f"Meta confirmada: **{status['meta_confirmada']}**. Aprovação econômica na confirmação: **{status['aprovado_na_confirmacao']}**.", ""]
        if status["confirmacao_avaliada"]:
            texto += [f"Intervalos descritivos de 95% por bootstrap de sessões: compra {ci['compra'][0]:.2%} a {ci['compra'][1]:.2%}; venda {ci['venda'][0]:.2%} a {ci['venda'][1]:.2%}. Há dependência temporal e seleção entre muitos candidatos; esses intervalos não corrigem o efeito da busca e não garantem taxa futura.", ""]
            refs=json.loads((pasta/"referencias_confirmacao.json").read_text(encoding="utf-8"))
            texto += ["Referências diagnósticas na confirmação, com a mesma saída congelada. Não foram usadas para escolher outro modelo depois de observar o teste. `evento_sem_modelo` opera a direção permitida pelo evento, sem os escores; não opera quando ambas são permitidas.", "",
                "| Referência | Compras | % compra | Vendas | % venda | Retorno |",
                "|---|---:|---:|---:|---:|---:|"]
            for r in refs:
                c="NA" if r['taxa_lucro_compra'] is None else f"{r['taxa_lucro_compra']:.2%}"
                v="NA" if r['taxa_lucro_venda'] is None else f"{r['taxa_lucro_venda']:.2%}"
                texto.append(f"| {r['referencia']} | {r['operacoes_compra']} | {c} | {r['operacoes_venda']} | {v} | {r['retorno_acumulado']:.2%} |")
            texto += ["", "Alvo menor que stop permite que as referências também tenham taxas altas sem previsão aprendida. Superar uma taxa isolada não comprova valor adicional da rede; as comparações usam conjuntos de operações distintos e exigiriam uma análise específica para atribuir diferenças ao filtro aprendido.", ""]
        else:
            texto += ["Esta busca não avaliou a confirmação. Ausência de avaliação não significa aprovação.", ""]
        suportados=tabela[tabela.suporte_suficiente].copy()
        suportados["menor_taxa"]=suportados[["taxa_lucro_compra","taxa_lucro_venda"]].min(axis=1)
        maximo=suportados.sort_values("menor_taxa",ascending=False).iloc[0]
        texto += [f"A maior taxa mínima entre as duas direções na seleção foi {maximo.taxa_lucro_compra:.2%} em compras ({maximo.lucrativas_compra}/{maximo.operacoes_compra}) e {maximo.taxa_lucro_venda:.2%} em vendas ({maximo.lucrativas_venda}/{maximo.operacoes_venda}), com retorno {maximo.retorno_acumulado:.2%}. Maximizar somente a taxa pode escolher uma estratégia com prejuízo.", "",
            f"Arquivos: [{pasta.name}](busca_estrategias/{pasta.name}/), incluindo plano, comparação, escolha congelada, modelos, auditoria reproduzida da seleção e operações quando houve confirmação.", ""]
    texto += ["## Interpretação e reprodução", ""]
    if confirmou:
        texto += ["O período de janeiro–junho/2024 agora foi observado. Procurar novas configurações até vencer nesse mesmo período produziria otimismo. Trabalhos adicionais podem usar esses dados como desenvolvimento, mas precisam de outra confirmação ainda não inspecionada ou acompanhamento prospectivo.", ""]
    texto += ["As taxas altas desta etapa decorrem também da mudança de payoff: alvos menores que stops podem aumentar vitórias e tornar cada perda mais cara. É por isso que taxa de acerto, quantidade, tamanho dos ganhos/perdas, retorno e drawdown são registrados juntos. Os resultados anteriores de saída fixa permanecem em `RESULTADOS_META70.md`.", "",
        "```powershell", ".\\.venv\\Scripts\\python.exe scripts\\buscar_estrategias.py --somente-validacao",
        ".\\.venv\\Scripts\\python.exe scripts\\buscar_estrategias.py --familias hist_binario --objetivo valor --horizontes 10,20,40 --alvos 0.002,0.003,0.004 --atr 1,1.5 --razoes-stop 1,1.5,2 --somente-validacao", "```", "",
        "Os testes verificam execução de alvos/stops, gaps, ambiguidades intrabar, taxas, causalidade, sobreposição, persistência, bloqueio da confirmação e rejeição de alta taxa com prejuízo. Fixtures sintéticas são exclusivamente testes de software; seus resultados não foram incluídos nas taxas de mercado. O log completo está em `TESTES_PREVISAO.log`."]
    confirmacoes=[p for p in pastas if (p/"resultado_confirmacao.json").exists()]
    if confirmacoes:
        r=json.loads((confirmacoes[0]/"resultado_confirmacao.json").read_text(encoding="utf-8"))
        resumo=f"**Resultado no período reservado:** compra {r['taxa_lucro_compra']:.2%} ({r['lucrativas_compra']}/{r['operacoes_compra']}), venda {r['taxa_lucro_venda']:.2%} ({r['lucrativas_venda']}/{r['operacoes_venda']}), retorno líquido {r['retorno_acumulado']:.2%}. A meta nas duas direções não foi confirmada." if not r["meta_70_lucro_ambas"] else f"**Resultado no período reservado:** compra {r['taxa_lucro_compra']:.2%}, venda {r['taxa_lucro_venda']:.2%}, retorno {r['retorno_acumulado']:.2%}."
        texto[2:2]=[resumo,""]
    log=RAIZ/"TESTES_PREVISAO.log"
    if log.exists():
        conteudo=log.read_text(encoding="utf-8")
        testes=re.search(r"Ran (\d+) tests in ([\d.]+)s\s+OK",conteudo)
        if testes:
            texto += ["",f"Verificação completa: **{testes[1]} testes passaram**, em {float(testes[2]):.1f} segundos. A emissão pela CLI carregou o modelo com alvo/stop e preservou `meta_confirmada: false`. Os testes verificam o software, não certificam a taxa de mercado."]
    Path(arquivo).write_text("\n".join(texto)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("pastas",nargs="+",type=Path)
    p.add_argument("--arquivo",type=Path,default=RAIZ/"RESULTADOS_ESTRATEGIAS.md")
    a=p.parse_args()
    gerar(a.pastas,a.arquivo)
    print(a.arquivo)
