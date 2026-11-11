"""Sete cenários do TCC, com treinamento independente da nova MLP de lucro.

Preserva a perturbação exclusiva de close_modelo. OHL e volume são originais;
o alvo financeiro é comum, calculado antes de qualquer transformação.
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
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from config import ConfigProjeto
from src.binary_events import FEATURES_EVENTOS, PlanoBinario, ajustar_modelo, criar_contexto, prever_scores, salvar_binario
from src.binary_study import _entradas
from src.bracket_strategy import SaidaBarreiras, avaliar_barreiras, criar_operacoes_barreiras, distancias_barreiras, permissoes_estrategia, selecionar_politica_barreiras
from src.filter_analysis import calcular_estatisticas_ruido, calcular_metricas_filtragem
from src.forecast_config import ConfigPrevisao
from src.forecast_data import limpar_candles
from src.forecast_experiment import salvar_json
from src.forecast_models import tensorflow
from src.noise import adicionar_ruido_gaussiano, suavizar_ema, wavelet_denoising
from src.trade_policy import aplicar_politica


CENARIOS = ("Original","Ruido_0_0005","Ruido_0_001","Ruido_0_002","Ruido_0_005","EMA","Wavelet")


@dataclass(frozen=True)
class PlanoTCCRuido:
    proporcao_treino: float = .70
    proporcao_validacao: float = .15
    janela: int = 12
    epocas: int = 30
    paciencia: int = 5
    sementes_redes: tuple = (42,123,2026)
    semente_ruido: int = 42
    custo: float = .0002
    minimo: int = 20
    meta: float = .70
    evento: str = "reversao_media"
    ema_spans: tuple = (2,3,5,8,13,20)
    wavelets: tuple = ("db2","db4","sym4")
    niveis: tuple = (1,2,3)
    janelas_wavelet: tuple = (64,128,256)
    escalas_wavelet: tuple = (.5,1.)

    def validar(self):
        if not 0<self.proporcao_treino<1 or not 0<self.proporcao_validacao<1 or self.proporcao_treino+self.proporcao_validacao>=1:
            raise ValueError("Proporções inválidas.")
        if self.janela<1 or self.epocas<1 or self.paciencia<1 or self.minimo<1 or not 0<=self.custo<1 or not 0<self.meta<=1:
            raise ValueError("Parâmetros de treino ou avaliação inválidos.")
        if not self.sementes_redes or len(set(self.sementes_redes))!=len(self.sementes_redes):
            raise ValueError("Informe sementes de rede distintas.")
        if not all([self.ema_spans,self.wavelets,self.niveis,self.janelas_wavelet,self.escalas_wavelet]) or min(self.ema_spans)<1 or min(self.niveis)<1 or min(self.janelas_wavelet)<2 or min(self.escalas_wavelet)<=0:
            raise ValueError("Grade de filtros inválida.")


def wavelet_causal_lotes(close,wavelet="db2",nivel=2,janela_causal=64,threshold_scale=.5,threshold_mode="soft"):
    """Mesma transformação causal do TCC; janelas completas em lotes NumPy.

    A implementação de referência trata as primeiras janelas incompletas.
    Nenhum lote compartilha coeficientes entre timestamps.
    """
    import pywt
    arr=close.to_numpy(dtype=float)
    n,w=len(arr),janela_causal
    parametros={"wavelet":wavelet,"nivel":nivel,"janela_causal":w,
        "threshold_scale":threshold_scale,"threshold_mode":threshold_mode}
    saida=arr.copy()
    inicio=min(n,w-1)
    if inicio:
        saida[:inicio]=wavelet_denoising(close.iloc[:inicio],**parametros).to_numpy()
    if n<w:
        return pd.Series(saida,index=close.index)
    nivel_usado=min(nivel,pywt.dwt_max_level(w,pywt.Wavelet(wavelet).dec_len))
    if nivel_usado<1:
        return pd.Series(saida,index=close.index)
    janelas=np.lib.stride_tricks.sliding_window_view(arr,w)
    for j in range(0,len(janelas),2048):
        bloco=janelas[j:j+2048]
        coefs=pywt.wavedec(bloco,wavelet,level=nivel_usado,axis=-1)
        sigma=np.median(np.abs(coefs[-1]),axis=1)/.6745
        limiar=threshold_scale*sigma*np.sqrt(2*np.log(w))
        tratados=[coefs[0]]+[pywt.threshold(c,limiar[:,None],mode=threshold_mode) for c in coefs[1:]]
        reconstruida=pywt.waverec(tratados,wavelet,axis=-1)
        saida[w-1+j:w-1+j+len(bloco)]=reconstruida[:,w-1]
    return pd.Series(saida,index=close.index)


def selecionar_filtros_tcc(close,fim_treino,plano):
    """Mesma grade e razão de RMSE do legado, calculadas só no treino."""
    treino=close.iloc[:fim_treino]
    if len(treino)<128:
        raise ValueError("Treino insuficiente para os filtros.")
    ruido=adicionar_ruido_gaussiano(treino,.002,plano.semente_ruido)
    rmse_ruido=float(np.sqrt(np.mean((ruido-treino)**2)))
    linhas=[]
    for span in plano.ema_spans:
        param={"span":span}
        filtrada=suavizar_ema(ruido,**param)
        rmse=float(np.sqrt(np.mean((filtrada-treino)**2)))
        linhas.append({"metodo":"EMA","parametros":param,"rmse_treino":rmse,"razao_rmse":rmse/rmse_ruido})
    for w in plano.wavelets:
        for nivel in plano.niveis:
            for janela in plano.janelas_wavelet:
                for escala in plano.escalas_wavelet:
                    param={"wavelet":w,"nivel":nivel,"janela_causal":janela,"threshold_scale":escala,"threshold_mode":"soft"}
                    filtrada=wavelet_causal_lotes(ruido,**param)
                    rmse=float(np.sqrt(np.mean((filtrada-treino)**2)))
                    linhas.append({"metodo":"Wavelet","parametros":param,"rmse_treino":rmse,"razao_rmse":rmse/rmse_ruido})
    melhores={metodo:min([r for r in linhas if r["metodo"]==metodo],key=lambda r:r["razao_rmse"])["parametros"] for metodo in ["EMA","Wavelet"]}
    return melhores,linhas


def criar_series_tcc(close,filtros,semente=42):
    series={"Original":close.copy()}
    for nome,intensidade in zip(CENARIOS[1:5],[.0005,.001,.002,.005]):
        series[nome]=adicionar_ruido_gaussiano(close,intensidade,semente)
    series["EMA"]=suavizar_ema(series["Ruido_0_002"],**filtros["EMA"])
    series["Wavelet"]=wavelet_causal_lotes(series["Ruido_0_002"],**filtros["Wavelet"])
    return series


def contexto_cenario(bruto,close_modelo):
    d=bruto.copy()
    d["close"]=close_modelo.to_numpy()
    # Não validar/remover OHLC após perturbar: o TCC altera só a observação do close.
    return criar_contexto(d,ConfigPrevisao())


def cortes_tcc(meta,plano):
    n=len(meta)
    a,b=int(n*plano.proporcao_treino),int(n*(plano.proporcao_treino+plano.proporcao_validacao))
    if not 0<a<b<n:
        raise ValueError("Base insuficiente para os cortes.")
    c=a+(b-a)//2
    limites={"inicio_validacao":meta.time.iloc[a],"inicio_ajuste":meta.time.iloc[c],"inicio_teste":meta.time.iloc[b]}
    iv,ia,it=[limites[k] for k in ["inicio_validacao","inicio_ajuste","inicio_teste"]]
    idx={"treino":np.flatnonzero((meta.time<iv)&(meta.time_alvo<iv)),
        "early":np.flatnonzero((meta.time>=iv)&(meta.time<ia)&(meta.time_alvo<ia)),
        "ajuste":np.flatnonzero((meta.time>=ia)&(meta.time<it)&(meta.time_alvo<it)),
        "teste":np.flatnonzero(meta.time>=it)}
    if min(len(x) for x in idx.values())<30:
        raise ValueError("Poucas amostras depois do expurgo temporal.")
    return idx,limites


def metricas_binarias(meta,sinais):
    saida={}
    for classe,nome in [(1,"compra"),(2,"venda")]:
        real=(meta[f"retorno_liquido_{nome}"].to_numpy()>1e-12).astype(int)
        previsto=(np.asarray(sinais)==classe).astype(int)
        cm=confusion_matrix(real,previsto,labels=[0,1])
        tn,fp,fn,tp=[int(x) for x in cm.ravel()]
        saida[nome]={"matriz":cm.tolist(),"TN":tn,"FP":fp,"FN":fn,"TP":tp,
            "precision":tp/(tp+fp) if tp+fp else None,"recall":tp/(tp+fn) if tp+fn else None,
            "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            "FPR":fp/(fp+tn) if fp+tn else None,"FDR":fp/(tp+fp) if tp+fp else None}
    return saida


def executar_tcc_ruido(csv,saida,plano=None,regra=None):
    plano=plano or PlanoTCCRuido()
    regra=regra or SaidaBarreiras(60,"atr",.5,2.)
    plano.validar()
    regra.validar()
    csv=Path(csv)
    bruto,qualidade=limpar_candles(pd.read_csv(csv))
    referencia=criar_contexto(bruto,ConfigPrevisao())
    meta=criar_operacoes_barreiras(referencia,regra,plano.custo)
    # Planeja os limites somente pela referência; nenhuma métrica decide o corte.
    matriz_ref=referencia[list(FEATURES_EVENTOS)].to_numpy(dtype=np.float32)
    validas=pd.Series(np.isfinite(matriz_ref).all(axis=1)).rolling(plano.janela).sum().eq(plano.janela).to_numpy()
    meta=meta.loc[validas[meta.posicao_bruta]].reset_index(drop=True)
    idx,limites=cortes_tcc(meta,plano)
    treino_fim=int(bruto.time.searchsorted(limites["inicio_validacao"]))
    pasta=Path(saida)/datetime.now(timezone.utc).strftime("tcc_nova_predicao_%Y%m%dT%H%M%S_%fZ")
    for sub in ["modelos","predicoes","historicos","series","graficos","codigo_snapshot"]:
        (pasta/sub).mkdir(parents=True,exist_ok=False)
    for f in Path(__file__).parent.glob("*.py"):
        shutil.copy2(f,pasta/"codigo_snapshot"/f.name)
    salvar_json(pasta/"plano_congelado.json",{"plano":asdict(plano),"regra":asdict(regra),"csv":str(csv),
        "csv_sha256":hashlib.sha256(csv.read_bytes()).hexdigest(),"inicio_dados":bruto.time.iloc[0],"fim_dados":bruto.time.iloc[-1],
        "qualidade":qualidade,"limites":limites,"cenarios":CENARIOS,"features":list(FEATURES_EVENTOS),
        "fonte_metodo":"TCC_Carlos_V12 (1).pdf, seções 3.1.3 e 3.6, páginas PDF 38 e 42–44.",
        "preservado":"Sete cenários, close_modelo perturbado, OHL/volume originais, respostas comuns, treino independente e scaler próprio só no treino, cortes 70/15/15.",
        "adaptado":"MLP 12x31 e dois alvos de lucro da previsão nova; custos e saída ATR comuns; validação dividida em Early Stopping e ajuste de limiares.",
        "natureza":"Exploratório: dados já inspecionados; não é uma confirmação independente nem réplica numérica da antiga tarefa de 0,8%."})
    logging.info("Selecionando a grade de filtros do TCC somente no treino: %d candles.",treino_fim)
    filtros,busca=selecionar_filtros_tcc(bruto.close,treino_fim,plano)
    salvar_json(pasta/"filtros_selecionados.json",filtros)
    salvar_json(pasta/"busca_filtros_treino.json",busca)
    series=criar_series_tcc(bruto.close,filtros,plano.semente_ruido)
    contextos={nome:contexto_cenario(bruto,serie) for nome,serie in series.items()}
    matrizes={nome:ctx[list(FEATURES_EVENTOS)].to_numpy(dtype=np.float32) for nome,ctx in contextos.items()}
    pos=meta.posicao_bruta.to_numpy()
    janelas=pos[:,None]-np.arange(plano.janela-1,-1,-1)[None,:]
    for nome,matriz in matrizes.items():
        if not np.isfinite(matriz[janelas]).all():
            raise ValueError(f"Cenário {nome} tem atributo inválido; não remover silenciosamente amostras.")
    y=(meta[["retorno_liquido_compra","retorno_liquido_venda"]].to_numpy()>1e-12).astype(np.float32)
    meta.to_csv(pasta/"referencia_alvos.csv",index=False)
    particoes=np.full(len(meta),"expurgado",dtype=object)
    for nome,indices in idx.items():
        particoes[indices]=nome
    pd.DataFrame({"time":meta.time,"time_alvo":meta.time_alvo,"posicao_bruta":pos,"particao":particoes}).to_csv(pasta/"particoes.csv",index=False)
    mascaras=permissoes_estrategia(referencia.iloc[pos],plano.evento)
    linhas_filtros=[]
    for nome in ["EMA","Wavelet"]:
        original_filtrada=suavizar_ema(bruto.close,**filtros[nome]) if nome=="EMA" else wavelet_causal_lotes(bruto.close,**filtros[nome])
        for part,inicio,fim in [("treino",0,treino_fim),
            ("validacao",treino_fim,int(bruto.time.searchsorted(limites["inicio_teste"]))),
            ("teste",int(bruto.time.searchsorted(limites["inicio_teste"])),len(bruto))]:
            metricas=calcular_metricas_filtragem(bruto.close,series["Ruido_0_002"],series[nome],original_filtrada,inicio,fim)
            linhas_filtros.append({"metodo":nome,"particao":part,**metricas})
    pd.DataFrame(linhas_filtros).to_csv(pasta/"diagnostico_filtros.csv",index=False)
    salvar_json(pasta/"estatisticas_ruido_0_002.json",calcular_estatisticas_ruido(bruto.close,series["Ruido_0_002"],.002))
    tarefas=[]
    treino_linhas=np.unique(janelas[idx["treino"]].ravel())
    for nome,matriz in matrizes.items():
        para_salvar=bruto.copy()
        para_salvar["close_modelo"]=series[nome]
        para_salvar.to_csv(pasta/"series"/f"{nome}.csv.gz",index=False,compression="gzip")
        scaler=StandardScaler().fit(matriz[treino_linhas])
        normalizada=scaler.transform(matriz).astype(np.float32)
        xt=_entradas(normalizada,pos[idx["treino"]],plano.janela)
        xe=_entradas(normalizada,pos[idx["early"]],plano.janela)
        xa=_entradas(normalizada,pos[idx["ajuste"]],plano.janela)
        for seed in plano.sementes_redes:
            logging.info("Treinando do zero: %s, rede semente %d.",nome,seed)
            modelos,historicos=ajustar_modelo("mlp_binario",xt,y[idx["treino"]],xe,y[idx["early"]],plano,(seed,))
            pa=prever_scores("mlp_binario",modelos,xa)
            masks=tuple(m[idx["ajuste"]] for m in mascaras)
            politica,ma=selecionar_politica_barreiras(meta.iloc[idx["ajuste"]],pa,masks,plano.minimo,plano.meta)
            artefato=pasta/"modelos"/nome/f"semente_{seed}"
            salvar_binario(artefato,modelos,scaler,{"familia":"mlp_binario","janela":plano.janela,"horizonte":regra.horizonte,
                "evento":"todos","banda":0.,"politica":politica,"meta_confirmada":False},plano)
            salvar_json(artefato/"modelo_estrategia.json",{"regra":asdict(regra),"evento":plano.evento,"meta_confirmada":False,"aprovado_na_confirmacao":False})
            salvar_json(artefato/"modelo_tcc_cenario.json",{"cenario":nome,"semente_rede":seed,"semente_ruido":plano.semente_ruido,
                "filtros":filtros,"inicio_dados":bruto.time.iloc[0],"politica":politica,"limites":limites,"meta_confirmada":False})
            salvar_json(artefato/"ajuste_validacao.json",ma)
            for h in historicos:
                pd.DataFrame(h["history"]).to_csv(pasta/"historicos"/f"{nome}_semente_{seed}.csv",index=False)
            tarefas.append({"cenario":nome,"semente":seed,"artefato":str(artefato),"politica":politica,
                "epocas_executadas":len(historicos[0]["history"]["loss"]),"scaler_media":scaler.mean_.tolist(),"linhas_fit_scaler":len(treino_linhas)})
    # Todos os modelos e limiares são congelados antes de abrir qualquer teste.
    salvar_json(pasta/"modelos_antes_teste.json",tarefas)
    resultados=[]
    mt=meta.iloc[idx["teste"]].reset_index(drop=True)
    for tarefa in tarefas:
        nome,seed=tarefa["cenario"],tarefa["semente"]
        artefato=Path(tarefa["artefato"])
        scaler=joblib.load(artefato/"scaler.joblib")
        modelo=tensorflow().keras.models.load_model(artefato/"rede_0.keras",compile=False)
        normalizada=scaler.transform(matrizes[nome]).astype(np.float32)
        x=_entradas(normalizada,pos[idx["teste"]],plano.janela)
        p=prever_scores("mlp_binario",[modelo],x)
        sinais=aplicar_politica(p,tarefa["politica"],*(m[idx["teste"]] for m in mascaras))
        m,usados=avaliar_barreiras(mt,sinais,plano.minimo,plano.meta)
        classificacao=metricas_binarias(mt,sinais)
        registro={"cenario":nome,"semente":seed,"epocas":tarefa["epocas_executadas"],**tarefa["politica"],**m}
        for lado in ["compra","venda"]:
            for medida in ["FP","TP","FN","TN","precision","recall","f1","FPR","FDR"]:
                registro[f"sinais_{medida}_{lado}"]=classificacao[lado][medida]
            taxa=m[f"taxa_lucro_{lado}"]
            registro[f"FDR_operacoes_{lado}"]=None if taxa is None else 1-taxa
        resultados.append(registro)
        salvar_json(artefato/"metricas_teste.json",{"financeiro":m,"classificacao_sinais":classificacao,"exploratorio":True})
        pred=mt.copy()
        pred["sinal"]=sinais
        pred["score_compra"],pred["score_venda"]=p[:,1],p[:,2]
        pred["operacao_executada"]=False
        pred.loc[usados,"operacao_executada"]=True
        pred.to_csv(pasta/"predicoes"/f"{nome}_semente_{seed}.csv.gz",index=False,compression="gzip")
        pd.DataFrame(resultados).to_csv(pasta/"comparacao_cenarios.csv",index=False)
        if seed==plano.sementes_redes[0]:
            _plot_matrizes(classificacao,nome,pasta/"graficos"/f"matrizes_{nome}.png")
        logging.info("Teste %s/%d: compra %.2f%% (%d), venda %.2f%% (%d), retorno %.2f%%",nome,seed,
            100*(m["taxa_lucro_compra"] or 0),m["operacoes_compra"],100*(m["taxa_lucro_venda"] or 0),m["operacoes_venda"],100*m["retorno_acumulado"])
    tabela=pd.DataFrame(resultados)
    resumo=tabela.groupby("cenario",sort=False).agg(compra_media=("taxa_lucro_compra","mean"),compra_desvio=("taxa_lucro_compra","std"),
        venda_media=("taxa_lucro_venda","mean"),venda_desvio=("taxa_lucro_venda","std"),retorno_medio=("retorno_acumulado","mean"),
        retorno_desvio=("retorno_acumulado","std"),compras_media=("operacoes_compra","mean"),vendas_media=("operacoes_venda","mean"),
        sementes_meta70=("meta_70_lucro_ambas","sum"),sementes_aprovadas=("aprovado","sum")).reset_index()
    resumo.to_csv(pasta/"resumo_sementes.csv",index=False)
    _graficos(pasta,tabela,series,bruto,limites)
    _relatorio(pasta,tabela,resumo,plano,regra,limites,filtros,idx)
    salvar_json(pasta/"CONCLUIDO.json",{"concluido":True,"modelos_independentes":len(tarefas),"cenarios":len(CENARIOS),
        "sementes_redes":plano.sementes_redes,"ruido_somente_close_modelo":True,"alvos_comuns":True,
        "meta_confirmada":False,"exploratorio":True})
    return pasta,tabela,resumo


def _plot_matrizes(metricas,nome,caminho):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(9,4))
    for ax,lado in zip(axs,["compra","venda"]):
        cm=np.asarray(metricas[lado]["matriz"])
        den=cm.sum(axis=1,keepdims=True)
        proporcao=np.divide(cm,den,out=np.zeros((2,2)),where=den>0)
        ax.imshow(proporcao,cmap="Blues",vmin=0,vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j,i,f"{cm[i,j]}\n{proporcao[i,j]:.1%}",ha="center",va="center",color="white" if proporcao[i,j]>.6 else "black")
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Não emitir","Emitir"],yticklabels=["Não lucrativa","Lucrativa"],
            xlabel="Sinal antes da sobreposição",ylabel="Resultado potencial real",title=lado.capitalize())
    fig.suptitle(nome)
    fig.tight_layout()
    fig.savefig(caminho,dpi=150)
    plt.close(fig)


def _graficos(pasta,tabela,series,bruto,limites):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,nome in zip(axs,["compra","venda"]):
        for seed,sub in tabela.groupby("semente",sort=False):
            ax.plot(np.arange(7),sub[f"taxa_lucro_{nome}"]*100,"o-",label=f"Rede {seed}")
        ax.axhline(70,color="gray",linestyle="--")
        ax.set(xticks=np.arange(7),xticklabels=["Original","0,0005","0,001","0,002","0,005","EMA","Wavelet"],title=nome.capitalize(),ylim=(0,100))
        ax.tick_params(axis="x",rotation=30)
        ax.grid(alpha=.2)
    axs[0].set_ylabel("Operações lucrativas (%)")
    axs[1].legend(fontsize=8)
    fig.suptitle("Redes treinadas independentemente por cenário")
    fig.tight_layout()
    fig.savefig(pasta/"graficos/taxas_cenarios.png",dpi=160)
    plt.close(fig)
    indices=np.flatnonzero((bruto.time>=limites["inicio_teste"]).to_numpy())[:200]
    fig,ax=plt.subplots(figsize=(11,4))
    for nome in ["Original","Ruido_0_002","EMA","Wavelet"]:
        ax.plot(np.arange(len(indices)),series[nome].iloc[indices],label=nome,linewidth=1)
    ax.set(xlabel="Candles iniciais do teste",ylabel="Fechamento de entrada",title="Série ruidosa 0,002 e tratamentos causais")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(pasta/"graficos/series_teste.png",dpi=160)
    plt.close(fig)


def _relatorio(pasta,tabela,resumo,plano,regra,limites,filtros,idx):
    pct=lambda x:"NA" if pd.isna(x) else f"{x:.2%}"
    linhas=["# Protocolo do TCC aplicado à nova previsão de lucro", "",
        "**Agora cada cenário tem redes e normalizadores próprios, treinados do zero.** Não é o ensaio anterior de perturbar somente a entrada de uma rede congelada. Este experimento preserva os sete cenários e a contaminação exclusiva de `close_modelo` descritos no TCC, adaptando o alvo e a arquitetura à previsão nova. Os resultados são exploratórios porque o histórico já foi inspecionado.", "",
        "## Correspondência com o TCC", "",
        "Preservados: PETR4 M5; original, quatro ruídos gaussianos (0,0005/0,001/0,002/0,005), EMA e Wavelet sobre a mesma realização 0,002; OHL e volume originais; indicadores recalculados; resposta comum; normalização e rede independentes por cenário; treino/validação/teste cronológicos 70%/15%/15% antes dos expurgos.", "",
        f"Adaptados à previsão nova: MLP de {plano.janela} candles e 31 atributos, duas saídas sigmoid de lucro líquido; alvo ATR × {regra.alvo:g}, stop {regra.razao_stop:g} vezes maior e prazo {regra.horizonte} candles. Taxas {plano.custo*10000:g} pontos-base por lado, uma posição por vez, entrada na próxima abertura e execução original. O evento `{plano.evento}` e as regras de risco são comuns; os indicadores desse controle usam a referência original.", "",
        "As duas respostas binárias podem ser lucrativas na mesma trajetória de barreiras, por isso não se inventa uma classe única para formar uma matriz 3×3. São calculadas matrizes 2×2 por direção, FPR/FDR/precisão/recall/F1 dos sinais potenciais e, separadamente, a taxa de lucro e FDR das operações executadas sem sobreposição.", "",
        "## Divisão e treinamento", "",
        f"Início da validação: {limites['inicio_validacao']}. Metade interna para ajustar decisões começa em {limites['inicio_ajuste']}. Início do teste: {limites['inicio_teste']}. Quantidades após expurgo: { {k:len(v) for k,v in idx.items()} }.", "",
        "A primeira metade da validação escolhe a época por Early Stopping; a segunda ajusta os limiares pela mesma grade e critério em todos os cenários. Nenhum limiar usa o teste. Todos os modelos e políticas são gravados em `modelos_antes_teste.json` antes da primeira avaliação. A proporção 70/15/15 é aplicada ao universo comum de amostras elegíveis; os expurgos reduzem os conjuntos. O horizonte financeiro exige continuidade na mesma sessão; as janelas de entrada podem conter sessões anteriores.", "",
        f"Semente de ruído: {plano.semente_ruido}, a mesma sequência escalada nas quatro intensidades, inclusive treino/validação/teste. Sementes das redes: {plano.sementes_redes}, pareadas entre os sete cenários. A primeira reproduz o desenho de uma semente do TCC; as demais são repetições de inicialização, não novas amostras de mercado nem sementes adicionais de ruído. Arquitetura: Flatten, Dense 64 ReLU, Dropout 0,25, Dense 32 ReLU, Dense 2 sigmoid, Adam 0,0003, batch 128, até {plano.epocas} épocas, paciência {plano.paciencia}. Sem escolher a melhor semente.", "",
        f"Os filtros foram selecionados novamente somente antes do início da validação, pela razão do RMSE absoluto de reconstrução contra a série 0,002, com a grade do código do TCC. Parâmetros escolhidos: `{filtros}`. A Wavelet em lotes é numericamente equivalente à rotina causal de referência; os testes verificam essa equivalência.", "",
        "## Resultado da primeira semente de rede", "",
        "| Cenário | Compras lucrativas/total | % compra | Vendas lucrativas/total | % venda | Retorno líquido |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in tabela[tabela.semente.eq(plano.sementes_redes[0])].itertuples():
        linhas.append(f"| {r.cenario} | {r.lucrativas_compra}/{r.operacoes_compra} | {pct(r.taxa_lucro_compra)} | {r.lucrativas_venda}/{r.operacoes_venda} | {pct(r.taxa_lucro_venda)} | {r.retorno_acumulado:.2%} |")
    linhas += ["", "## Médias entre sementes de rede", "",
        "| Cenário | % compra média ± desvio | % venda média ± desvio | Retorno médio | Sementes com 70% e suporte | Sementes com aprovação econômica |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in resumo.itertuples():
        linhas.append(f"| {r.cenario} | {pct(r.compra_media)} ± {pct(r.compra_desvio)} | {pct(r.venda_media)} ± {pct(r.venda_desvio)} | {r.retorno_medio:.2%} | {r.sementes_meta70}/{len(plano.sementes_redes)} | {r.sementes_aprovadas}/{len(plano.sementes_redes)} |")
    linhas += ["", "## Limites e artefatos", "",
        "O fechamento de entrada perturbado pode ficar fora de low/high originais, pois o TCC altera só esse atributo e não cria um candle integralmente contaminado. Esses casos são mantidos e não são descartados da avaliação. Existem canais de preço originais preservados; esta é uma característica do protocolo replicado. Não é a perturbação OHLC coerente do ensaio anterior.", "",
        "Os preços, alvos, ATR de risco, durações potenciais e datas são originais e iguais nos sete cenários. As diferenças refletem o pipeline inteiro treinado por condição, incluindo scaler, pesos e limiar ajustado na validação; não isolam apenas o efeito de passar um filtro em pesos fixos. O teste é outro período em relação ao ensaio congelado de janeiro–junho/2024, portanto as taxas não são comparações pareadas com aquele relatório.", "",
        "As métricas de sinais antes da sobreposição não são iguais às taxas das operações. A meta exige ao menos vinte operações por direção. `aprovado` também exige lucro médio positivo em cada direção e retorno acumulado positivo. Taxas altas em uma semente não certificam previsão futura, e `meta_confirmada` permanece falsa.", "",
        "Arquivos: plano, partições e alvos comuns, filtros e diagnósticos de reconstrução, modelos/scalers por cenário/semente, históricos de treino, políticas congeladas antes do teste, sinais e operações por timestamp, métricas completas de falsos positivos, matrizes e gráficos. O PDF e os resultados antigos foram preservados.", "",
        "```powershell", ".\\.venv\\Scripts\\python.exe scripts\\treinar_ruido_tcc.py", "```"]
    (pasta/"LEIA_RESULTADOS.md").write_text("\n".join(linhas)+"\n",encoding="utf-8")


def carregar_prever_cenario_tcc(dados,pasta):
    from src.binary_events import carregar_prever_binario
    pasta=Path(pasta)
    info=json.loads((pasta/"modelo_tcc_cenario.json").read_text(encoding="utf-8"))
    base=json.loads((pasta/"modelo_binario.json").read_text(encoding="utf-8"))
    estr=json.loads((pasta/"modelo_estrategia.json").read_text(encoding="utf-8"))
    bruto,_=limpar_candles(dados)
    if bruto.time.iloc[0]!=pd.Timestamp(info["inicio_dados"]):
        raise ValueError("Use o histórico desde a data original para preservar o alinhamento da realização de ruído.")
    serie=criar_series_tcc(bruto.close,info["filtros"],info["semente_ruido"])[info["cenario"]]
    ctx=contexto_cenario(bruto,serie)
    valores=ctx[list(FEATURES_EVENTOS)].iloc[-base["janela"]:].to_numpy(dtype=np.float32)
    if len(valores)!=base["janela"] or not np.isfinite(valores).all():
        raise ValueError("Histórico insuficiente.")
    scaler=joblib.load(pasta/"scaler.joblib")
    x=scaler.transform(valores).astype(np.float32)[None,:,:]
    modelo=tensorflow().keras.models.load_model(pasta/"rede_0.keras",compile=False)
    p=prever_scores("mlp_binario",[modelo],x)
    referencia=criar_contexto(bruto,ConfigPrevisao())
    masks=permissoes_estrategia(referencia.iloc[-1:],estr["evento"])
    sinal=int(aplicar_politica(p,info["politica"],*masks)[0])
    alvo,stop=distancias_barreiras(referencia,SaidaBarreiras(**estr["regra"]),base["plano"]["custo"])
    return {"tipo_modelo":"tcc_retreino_cenario","cenario":info["cenario"],"semente_rede":info["semente_rede"],
        "sinal":["manter","compra","venda"][sinal],"classe_predita":sinal,"score_compra":float(p[0,1]),"score_venda":float(p[0,2]),
        "alvo_fracao":float(alvo[-1]),"stop_fracao":float(stop[-1]),"instante_emissao":str(bruto.time.iloc[-1]+pd.Timedelta(minutes=5)),
        "meta_confirmada":False,"status":"Emissão experimental da condição treinada; o ruído sintético é reaplicado à entrada, sem ordens reais."}
