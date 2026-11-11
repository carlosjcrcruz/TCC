"""Filtro de oportunidades com contexto M15/H1 e reestimação cronológica.

O histórico deste experimento já foi inspecionado. Walk-forward evita treinar
com o futuro de cada bloco, mas não restaura um teste independente da pesquisa.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.binary_events import FEATURES_EVENTOS, PlanoBinario, criar_contexto
from src.bracket_strategy import (SaidaBarreiras, avaliar_barreiras,
    criar_operacoes_barreiras, distancias_barreiras, permissoes_estrategia,
    pontuar_barreiras, selecionar_politica_barreiras)
from src.forecast_config import ConfigPrevisao
from src.forecast_data import limpar_candles
from src.forecast_experiment import salvar_json
from src.trade_policy import aplicar_politica


VARIANTES = ("logistica", "boosting", "extra_trees", "ensemble")


@dataclass(frozen=True)
class PlanoMeta:
    inicio_avaliacao: str = "2024-01-01T00:00:00Z"
    meses_treino: int = 18
    meses_calibracao: int = 3
    meses_bloco: int = 3
    minimo: int = 20
    custo: float = .0002
    meta: float = .7
    evento: str = "reversao_media"
    variantes: tuple = VARIANTES

    def validar(self):
        if pd.Timestamp(self.inicio_avaliacao).tzinfo is None:
            raise ValueError("A data inicial precisa de fuso.")
        if min(self.meses_treino,self.meses_calibracao,self.meses_bloco,self.minimo) < 1:
            raise ValueError("Janelas e suporte devem ser positivos.")
        if not 0 <= self.custo < 1 or not 0 < self.meta <= 1:
            raise ValueError("Custo ou meta inválidos.")
        if not self.variantes or not set(self.variantes) <= set(VARIANTES):
            raise ValueError("Variantes inválidas.")
        if self.evento not in {"reversao_media","choque_reversao","rsi_extremo","rompimento","recuo_tendencia"}:
            raise ValueError("O evento precisa determinar uma direção.")


def atributos_multiescala(ctx, evento="reversao_media"):
    """Agregações só se tornam disponíveis quando o candle maior fechou."""
    x=ctx[list(FEATURES_EVENTOS)].copy()
    emissao=ctx.time+pd.Timedelta(minutes=5)
    for minutos in [15,60]:
        b=ctx.set_index("time").resample(f"{minutos}min",closed="left",label="right").agg(
            fechamento=("close","last"),maxima=("high","max"),minima=("low","min"),n=("close","count"))
        b=b[b.n.eq(minutos//5)].copy()
        prefixo=f"m{minutos}_"
        f=pd.DataFrame(index=b.index)
        f[prefixo+"retorno_1"]=b.fechamento.pct_change(fill_method=None)
        f[prefixo+"retorno_3"]=b.fechamento.pct_change(3,fill_method=None)
        f[prefixo+"distancia_media8"]=b.fechamento/b.fechamento.rolling(8).mean()-1
        f[prefixo+"vol12"]=b.fechamento.pct_change(fill_method=None).rolling(12).std()
        f[prefixo+"amplitude"]=(b.maxima-b.minima)/b.fechamento
        f=f.reset_index().rename(columns={"time":"disponivel_em"})
        # Mesmo construído com o CSV inteiro, um bloco futuro não entra no asof.
        unido=pd.merge_asof(pd.DataFrame({"emissao":emissao}),f,left_on="emissao",right_on="disponivel_em",direction="backward")
        for coluna in f.columns.drop("disponivel_em"):
            x[coluna]=unido[coluna].to_numpy()
    mc,mv=permissoes_estrategia(ctx,evento)
    lado=np.select([mc&~mv,mv&~mc],[1,-1],default=0)
    x["lado_regra"]=lado
    for coluna in ["retorno_1","retorno_5","retorno_20","desvio_media_ticks","distancia_ema20",
                   "m15_retorno_3","m60_retorno_3","m60_distancia_media8"]:
        x["orientado_"+coluna]=x[coluna]*lado
    x["rsi_orientado"]=(ctx.rsi14-.5)*lado
    return x.replace([np.inf,-np.inf],np.nan),lado


def treinar_variantes(x,y):
    if len(x)<50 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Poucas oportunidades ou atributos/rótulos inválidos.")
    bases={"logistica":LogisticRegression(C=.1,max_iter=1000,random_state=42),
        "boosting":HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=7,min_samples_leaf=60,
            l2_regularization=10.,early_stopping=False,random_state=42),
        "extra_trees":ExtraTreesClassifier(n_estimators=300,max_depth=8,min_samples_leaf=30,
            max_features=.7,n_jobs=4,random_state=42)}
    modelos={}
    for nome,estimador in bases.items():
        if len(np.unique(y))<2:
            estimador=DummyClassifier(strategy="prior")
        modelo=make_pipeline(StandardScaler(),estimador)
        modelo.fit(x,y)
        modelos[nome]=[modelo]
    modelos["ensemble"]=[modelos[n][0] for n in ["logistica","boosting","extra_trees"]]
    return modelos


def scores_meta(modelos,x,lado):
    p=[]
    for modelo in modelos:
        pr=modelo.predict_proba(x)
        classe=np.flatnonzero(np.asarray(modelo.classes_)==1)
        p.append(pr[:,classe[0]] if len(classe) else np.zeros(len(x)))
    s=np.mean(p,axis=0)
    return np.column_stack([np.zeros(len(x)),np.where(lado==1,s,0.),np.where(lado==-1,s,0.)])


def separar_janelas(meta,inicio,meses_treino=18,meses_calibracao=3,meses_bloco=3,fim_dados=None):
    inicio=pd.Timestamp(inicio)
    cal_inicio=inicio-pd.DateOffset(months=meses_calibracao)
    treino_inicio=cal_inicio-pd.DateOffset(months=meses_treino)
    fim=inicio+pd.DateOffset(months=meses_bloco)
    if fim_dados is not None:
        fim=min(fim,pd.Timestamp(fim_dados))
    idx={"treino":np.flatnonzero((meta.time>=treino_inicio)&(meta.time<cal_inicio)&(meta.time_alvo<cal_inicio)),
         "calibracao":np.flatnonzero((meta.time>=cal_inicio)&(meta.time<inicio)&(meta.time_alvo<inicio)),
         "avaliacao":np.flatnonzero((meta.time>=inicio)&(meta.time<fim)&(meta.time_alvo<fim))}
    return idx,{"inicio_treino":str(treino_inicio),"inicio_calibracao":str(cal_inicio),
                "inicio_avaliacao":str(inicio),"fim_avaliacao":str(fim)}


def registrar_predicoes(meta,sinais,p):
    d=meta.copy()
    m,usados=avaliar_barreiras(d,sinais,minimo=1)
    d["sinal"]=sinais
    d["score_compra"],d["score_venda"]=p[:,1],p[:,2]
    d["operacao_executada"]=False
    d.loc[usados,"operacao_executada"]=True
    return d


def executar_metafiltro(csv,saida,plano=None,regra=None):
    plano=plano or PlanoMeta()
    regra=regra or SaidaBarreiras(60,"atr",.5,2.)
    plano.validar()
    regra.validar()
    csv=Path(csv)
    bruto,qualidade=limpar_candles(pd.read_csv(csv))
    ctx=criar_contexto(bruto,ConfigPrevisao())
    features,lado=atributos_multiescala(ctx,plano.evento)
    matriz=features.to_numpy(dtype=np.float32)
    meta=criar_operacoes_barreiras(ctx,regra,plano.custo)
    meta=meta.loc[np.isfinite(matriz[meta.posicao_bruta]).all(axis=1)].reset_index(drop=True)
    pos=meta.posicao_bruta.to_numpy()
    x=matriz[pos]
    direcao=lado[pos]
    y=(np.where(direcao==1,meta.retorno_liquido_compra,meta.retorno_liquido_venda)>1e-12).astype(int)
    pasta=Path(saida)/datetime.now(timezone.utc).strftime("metafiltro_%Y%m%dT%H%M%S_%fZ")
    pasta.mkdir(parents=True,exist_ok=False)
    (pasta/"codigo_snapshot").mkdir()
    for f in Path(__file__).parent.glob("*.py"):
        shutil.copy2(f,pasta/"codigo_snapshot"/f.name)
    salvar_json(pasta/"plano_congelado.json",{"plano":asdict(plano),"regra":asdict(regra),
        "features":list(features.columns),"csv":str(csv),"sha256_csv":hashlib.sha256(csv.read_bytes()).hexdigest(),
        "qualidade":qualidade,"natureza":"Exploratório: histórico já inspecionado; não constitui nova confirmação independente.",
        "selecao":"Treino até maio/2023, limiares maio-agosto/2023, seleção setembro-dezembro/2023. Escolher uma variante antes do walk-forward.",
        "aprendizado":"Apenas oportunidades da regra primária; alvo binário é seu lucro líquido; três classificadores e média das probabilidades."})
    # Seleção inicial usa somente dados anteriores ao início da avaliação.
    inicio=pd.Timestamp(plano.inicio_avaliacao)
    corte_selecao=inicio-pd.DateOffset(months=4)
    corte_calibracao=inicio-pd.DateOffset(months=8)
    treino=np.flatnonzero((meta.time<corte_calibracao)&(meta.time_alvo<corte_calibracao)&(direcao!=0))
    ajuste=np.flatnonzero((meta.time>=corte_calibracao)&(meta.time<corte_selecao)&(meta.time_alvo<corte_selecao))
    selecao=np.flatnonzero((meta.time>=corte_selecao)&(meta.time<inicio)&(meta.time_alvo<inicio))
    if min(len(treino),len(ajuste),len(selecao))<50:
        raise ValueError("Histórico insuficiente para selecionar o metafiltro.")
    modelos=treinar_variantes(x[treino],y[treino])
    registros,melhor=[],None
    for variante in plano.variantes:
        pa=scores_meta(modelos[variante],x[ajuste],direcao[ajuste])
        masks=(direcao[ajuste]==1,direcao[ajuste]==-1)
        pol,am=selecionar_politica_barreiras(meta.iloc[ajuste],pa,masks,plano.minimo,plano.meta)
        ps=scores_meta(modelos[variante],x[selecao],direcao[selecao])
        sinais=aplicar_politica(ps,pol,direcao[selecao]==1,direcao[selecao]==-1)
        m,_=avaliar_barreiras(meta.iloc[selecao],sinais,plano.minimo,plano.meta)
        r={"variante":variante,**pol,**m}
        registros.append(r)
        if melhor is None or pontuar_barreiras(m)>pontuar_barreiras(melhor):
            melhor=r
            inicial_pred=registrar_predicoes(meta.iloc[selecao].reset_index(drop=True),sinais,ps)
    pd.DataFrame(registros).to_csv(pasta/"comparacao_selecao.csv",index=False)
    salvar_json(pasta/"escolha_antes_walkforward.json",melhor)
    inicial_pred.to_csv(pasta/"predicoes_selecao.csv",index=False)
    variante=melhor["variante"]
    resultados,predicoes,referencias=[],[],[]
    atual=inicio
    fim_dados=ctx.time.iloc[-1]+pd.Timedelta(minutes=5)
    ultimo=None
    while atual<fim_dados:
        idx,fronteiras=separar_janelas(meta,atual,plano.meses_treino,plano.meses_calibracao,plano.meses_bloco,fim_dados)
        it=idx["treino"][direcao[idx["treino"]]!=0]
        ia,iv=idx["calibracao"],idx["avaliacao"]
        if len(iv)==0:
            atual+=pd.DateOffset(months=plano.meses_bloco)
            continue
        if min(len(it),len(ia))<50:
            raise ValueError("O bloco tem treino ou calibração insuficientes.")
        modelos_bloco=treinar_variantes(x[it],y[it])[variante]
        pa=scores_meta(modelos_bloco,x[ia],direcao[ia])
        pol,am=selecionar_politica_barreiras(meta.iloc[ia],pa,(direcao[ia]==1,direcao[ia]==-1),plano.minimo,plano.meta)
        ps=scores_meta(modelos_bloco,x[iv],direcao[iv])
        sinais=aplicar_politica(ps,pol,direcao[iv]==1,direcao[iv]==-1)
        m,_=avaliar_barreiras(meta.iloc[iv],sinais,plano.minimo,plano.meta)
        bloco={**fronteiras,"variante":variante,"oportunidades_treino":len(it),**pol,**m}
        resultados.append(bloco)
        pred=registrar_predicoes(meta.iloc[iv].reset_index(drop=True),sinais,ps)
        pred["bloco_inicio"]=str(atual)
        predicoes.append(pred)
        sinal_regra=np.select([direcao[iv]==1,direcao[iv]==-1],[1,2],default=0)
        referencias.append(registrar_predicoes(meta.iloc[iv].reset_index(drop=True),sinal_regra,np.zeros_like(ps)))
        ultimo={"modelos":modelos_bloco,"politica":pol,"fronteiras":fronteiras}
        pd.DataFrame(resultados).to_csv(pasta/"resultados_blocos.csv",index=False)
        logging.info("Meta %s: compra %.2f%% (%d), venda %.2f%% (%d), retorno %.2f%%",str(atual.date()),
            100*(m["taxa_lucro_compra"] or 0),m["operacoes_compra"],100*(m["taxa_lucro_venda"] or 0),m["operacoes_venda"],100*m["retorno_acumulado"])
        atual+=pd.DateOffset(months=plano.meses_bloco)
    if not predicoes:
        raise ValueError("Nenhum bloco posterior para avaliar.")
    total=pd.concat(predicoes,ignore_index=True)
    base=pd.concat(referencias,ignore_index=True)
    m,usados=avaliar_barreiras(total,total.sinal,plano.minimo,plano.meta)
    mb,ub=avaliar_barreiras(base,base.sinal,plano.minimo,plano.meta)
    total["operacao_executada"]=False
    total.loc[usados,"operacao_executada"]=True
    base["operacao_executada"]=False
    base.loc[ub,"operacao_executada"]=True
    total.to_csv(pasta/"predicoes_walkforward.csv",index=False)
    base.to_csv(pasta/"regra_sem_filtro.csv",index=False)
    # Recorte fixo para comparar com o resultado anterior, já observado.
    recorte=total[total.time<pd.Timestamp("2024-07-01T00:00:00Z")].reset_index(drop=True)
    mr,_=avaliar_barreiras(recorte,recorte.sinal,plano.minimo,plano.meta)
    resumo={"variante":variante,"exploratorio":True,"meta_confirmada":False,
        "walkforward":m,"regra_sem_filtro":mb,"janeiro_junho_2024":mr,
        "blocos_com_meta":sum(int(r["meta_70_lucro_ambas"]) for r in resultados),"blocos":len(resultados)}
    salvar_json(pasta/"resultado_exploratorio.json",resumo)
    artefato=pasta/"modelo_experimental"
    artefato.mkdir()
    joblib.dump(ultimo["modelos"],artefato/"estimadores.joblib")
    salvar_json(artefato/"modelo_meta.json",{"features":list(features.columns),"plano":asdict(plano),"regra":asdict(regra),
        "variante":variante,"politica":ultimo["politica"],"fronteiras":ultimo["fronteiras"],"meta_confirmada":False})
    _relatorio(pasta,resumo,melhor,resultados,plano,regra)
    return pasta,pd.DataFrame(resultados),resumo


def _relatorio(pasta,resumo,selecao,blocos,plano,regra):
    r=resumo["walkforward"]
    pct=lambda a:"NA" if a is None else f"{a:.2%}"
    linhas=["# Filtro de oportunidades com atualização temporal", "",
        f"**Resultado agregado:** compras {pct(r['taxa_lucro_compra'])} ({r['lucrativas_compra']}/{r['operacoes_compra']}), vendas {pct(r['taxa_lucro_venda'])} ({r['lucrativas_venda']}/{r['operacoes_venda']}), retorno líquido {r['retorno_acumulado']:.2%}. A meta nas duas direções foi observada: **{r['meta_70_lucro_ambas']}**. O modelo anterior foi preservado.", "",
        "**Experimento exploratório.** Os períodos já foram avaliados em pesquisas anteriores. Treinar só com o passado de cada bloco evita vazamento no ajuste local, mas não torna esta uma confirmação independente.", "",
        f"Variante escolhida antes do walk-forward: **{resumo['variante']}**. A regra primária é `{plano.evento}`; saída ATR × {regra.alvo:g}, stop {regra.razao_stop:g} vezes maior e prazo {regra.horizonte} candles M5. Taxas de {plano.custo*10000:g} pontos-base por lado, sem spread ou slippage. Uma posição por vez, stop primeiro em candle ambíguo e execução conservadora de gaps.", "",
        "## Método", "",
        "A primeira etapa determina a direção usando o evento causal. A segunda aprende somente com as oportunidades desse evento, em vez de todos os candles. Seu alvo é executar ou rejeitar a operação segundo o lucro líquido que teria ocorrido. O modelo compartilha informações de compra e venda, com atributos orientados pela direção, e recebe contexto de M15 e H1. Candles maiores só entram após seu fechamento completo.", "",
        "São comparadas regressão logística, boosting, Extra Trees e a média dos três escores. A seleção inicial usa treino anterior a maio/2023, ajuste de limiares em maio–agosto e comparação em setembro–dezembro/2023. Não se escolhe uma nova variante depois de ver o walk-forward. São reestimados somente seus parâmetros e limiares usando janelas móveis predefinidas.", "",
        f"Cada bloco tem {plano.meses_bloco} meses; o treino usa até {plano.meses_treino} meses anteriores aos {plano.meses_calibracao} meses de calibração. O scaler é ajustado somente nas oportunidades do treino. Alvos que atravessam as fronteiras são expurgados. O último bloco pode ser parcial. O limite de 70% exige ao menos {plano.minimo} operações em cada direção. A aprovação econômica também exige lucro médio positivo nos dois lados e retorno acumulado positivo.", "",
        "## Resultados", "",
        "| Recorte | Compra lucrativa / total | % compra | Venda lucrativa / total | % venda | Retorno líquido | Profit factor |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for nome,r in [("Seleção",selecao),("Jan–jun/2024 (reutilizado)",resumo["janeiro_junho_2024"]),("Walk-forward completo",resumo["walkforward"]),("Regra sem filtro, mesmo período",resumo["regra_sem_filtro"])]:
        pf="NA" if r["profit_factor"] is None else f"{r['profit_factor']:.3f}"
        linhas.append(f"| {nome} | {r['lucrativas_compra']}/{r['operacoes_compra']} | {pct(r['taxa_lucro_compra'])} | {r['lucrativas_venda']}/{r['operacoes_venda']} | {pct(r['taxa_lucro_venda'])} | {r['retorno_acumulado']:.2%} | {pf} |")
    linhas += ["",f"Blocos com 70% nas duas direções e suporte suficiente: **{resumo['blocos_com_meta']}/{resumo['blocos']}**. Meta observada no agregado exploratório: **{resumo['walkforward']['meta_70_lucro_ambas']}**. Aprovação econômica no agregado: **{resumo['walkforward']['aprovado']}**. Meta confirmada independentemente: **False**.", "",
        "Os resultados são sensíveis ao período e à quantidade de operações. A regra sem filtro e o modelo executam conjuntos de operações diferentes; diferenças de retorno não devem ser atribuídas somente à capacidade de prever. A taxa de vitória não substitui a relação entre ganhos e perdas. Não foram eliminadas perdas da avaliação com base no futuro.", "",
        "## Blocos cronológicos", "",
        "| Início | Compras | % compra | Vendas | % venda | Retorno |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in blocos:
        linhas.append(f"| {r['inicio_avaliacao'][:10]} | {r['operacoes_compra']} | {pct(r['taxa_lucro_compra'])} | {r['operacoes_venda']} | {pct(r['taxa_lucro_venda'])} | {r['retorno_acumulado']:.2%} |")
    linhas += ["", "## Reprodução", "", "```powershell", ".\\.venv\\Scripts\\python.exe scripts\\testar_metafiltro.py", "```", "",
        "O artefato `modelo_experimental/` é o modelo do último bloco, com pesos e limiares ajustados somente antes desse bloco. A emissão usa o último candle completo do CSV, não uma cotação ao vivo, e não envia ordens. `meta_confirmada` permanece falsa. Fonte, hash, regras e código foram congelados em `plano_congelado.json` e `codigo_snapshot/`. As operações, inclusive sinais rejeitados por sobreposição, estão nos CSVs."]
    (pasta/"LEIA_RESULTADOS.md").write_text("\n".join(linhas)+"\n",encoding="utf-8")


def carregar_prever_meta(dados,pasta):
    pasta=Path(pasta)
    info=json.loads((pasta/"modelo_meta.json").read_text(encoding="utf-8"))
    plano=PlanoMeta(**{**info["plano"],"variantes":tuple(info["plano"]["variantes"])})
    regra=SaidaBarreiras(**info["regra"])
    bruto,_=limpar_candles(dados)
    ctx=criar_contexto(bruto,ConfigPrevisao())
    features,lado=atributos_multiescala(ctx,plano.evento)
    if list(features.columns)!=info["features"]:
        raise ValueError("Atributos do metafiltro incompatíveis.")
    x=features.iloc[-1:].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Histórico insuficiente para os candles M15/H1 fechados.")
    modelos=joblib.load(pasta/"estimadores.joblib")
    p=scores_meta(modelos,x,lado[-1:])
    sinal=int(aplicar_politica(p,info["politica"],lado[-1:]==1,lado[-1:]==-1)[0])
    alvo,stop=distancias_barreiras(ctx,regra,plano.custo)
    return {"tipo_modelo":"metafiltro_temporal","variante":info["variante"],"sinal":["manter","compra","venda"][sinal],
        "classe_predita":sinal,"direcao_regra_primaria":int(lado[-1]),"score_compra":float(p[0,1]),"score_venda":float(p[0,2]),
        "politica":info["politica"],"alvo_fracao":float(alvo[-1]),"stop_fracao":float(stop[-1]),
        "prazo_maximo_minutos":regra.horizonte*5,"custo_por_lado":plano.custo,
        "instante_emissao":str(ctx.time.iloc[-1]+pd.Timedelta(minutes=5)),
        "meta_confirmada":False,"status":"Modelo experimental; escores não garantem lucro. Entrada na abertura seguinte; horizonte condicionado à continuidade da sessão."}
