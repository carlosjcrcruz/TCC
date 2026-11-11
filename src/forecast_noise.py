"""Perturbação pareada de entradas de um previsor de operações já congelado.

Não reestima pesos/scaler/limiares, não altera preços de execução ou respostas,
e não constitui nova confirmação da estratégia. Ruído é erro de observação.
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

from src.binary_events import FEATURES_EVENTOS, criar_contexto, prever_scores
from src.binary_study import _entradas
from src.bracket_strategy import SaidaBarreiras, avaliar_barreiras, criar_operacoes_barreiras, permissoes_estrategia
from src.forecast_config import ConfigPrevisao
from src.forecast_data import limpar_candles
from src.forecast_experiment import salvar_json
from src.forecast_models import tensorflow
from src.noise import suavizar_ema, wavelet_denoising
from src.trade_policy import aplicar_politica


OHLC = ("open","high","low","close")


@dataclass(frozen=True)
class PlanoRuidoPrevisao:
    inicio: str = "2024-01-01T00:00:00Z"
    fim: str = "2024-07-01T00:00:00Z"
    intensidades: tuple = (.0005,.001,.002,.005)
    sementes: tuple = (42,123,2026,7,17,31,73,101,211,509)
    semente_filtros: int = 314159
    intensidade_filtros: float = .002
    ema_spans: tuple = (2,3,5,8)
    wavelet_opcoes: tuple = (("db2",1,64,.5),("db2",2,64,.5),
        ("db2",2,128,.5),("db4",2,64,.5),("db2",2,64,1.))
    escopos: tuple = ("rede","rede_evento")
    minimo: int = 20
    bootstrap: int = 2000

    def validar(self):
        a,b=pd.Timestamp(self.inicio),pd.Timestamp(self.fim)
        if a.tzinfo is None or b.tzinfo is None or a>=b:
            raise ValueError("Intervalo de avaliação inválido ou sem fuso.")
        if not self.intensidades or any(not np.isfinite(x) or not 0<x<.1 for x in self.intensidades):
            raise ValueError("Intensidades precisam estar entre zero e 0,1.")
        if not self.sementes or len(set(self.sementes))!=len(self.sementes) or self.semente_filtros in self.sementes:
            raise ValueError("Sementes de avaliação distintas e separadas da seleção de filtros são obrigatórias.")
        if not self.ema_spans or min(self.ema_spans)<1 or not self.wavelet_opcoes:
            raise ValueError("Grade de filtros vazia ou inválida.")
        if not self.escopos or not set(self.escopos)<={"rede","rede_evento"} or self.minimo<1 or self.bootstrap<100:
            raise ValueError("Escopos, suporte ou reamostragens inválidos.")
        if not np.isfinite(self.intensidade_filtros) or not 0<self.intensidade_filtros<.1:
            raise ValueError("Intensidade de seleção dos filtros inválida.")


def perturbar_ohlc(dados,intensidade,semente):
    if not np.isfinite(intensidade) or not 0<=intensidade<.1:
        raise ValueError("Intensidade inválida.")
    ruido=np.random.default_rng(semente).standard_normal(len(dados))
    fator=1+intensidade*ruido
    if np.any(fator<=0):
        raise ValueError("Ruído gerou fator não positivo.")
    d=dados.copy()
    d[list(OHLC)]=d[list(OHLC)].mul(fator,axis=0)
    return d


def tratar_ohlc(dados,tratamento,parametros):
    if tratamento=="sem_tratamento":
        return dados.copy()
    if tratamento=="EMA":
        close=suavizar_ema(dados.close,**parametros)
    elif tratamento=="Wavelet":
        close=wavelet_denoising(dados.close,**parametros)
    else:
        raise ValueError("Tratamento desconhecido.")
    if not np.isfinite(close).all() or (close<=0).any():
        raise ValueError("Filtro gerou fechamento inválido.")
    fator=close/dados.close
    d=dados.copy()
    # Preserva o formato relativo do candle e a ordem low <= open/close <= high.
    d[list(OHLC)]=d[list(OHLC)].mul(fator,axis=0)
    return d


def selecionar_filtros(dados,fim_treino,plano):
    treino=dados.loc[dados.time<pd.Timestamp(fim_treino)].reset_index(drop=True)
    if len(treino)<128:
        raise ValueError("Poucos candles de treino para selecionar filtros.")
    ruidoso=perturbar_ohlc(treino,plano.intensidade_filtros,plano.semente_filtros)
    ignorar=max(128,max(int(x[2]) for x in plano.wavelet_opcoes))
    if len(treino)<=ignorar:
        raise ValueError("Treino insuficiente após o aquecimento dos filtros.")
    registros=[]
    for nome,opcoes in [("EMA",[{"span":x} for x in plano.ema_spans]),
        ("Wavelet",[{"wavelet":w,"nivel":n,"janela_causal":j,"threshold_scale":s,"threshold_mode":"soft"} for w,n,j,s in plano.wavelet_opcoes])]:
        for parametros in opcoes:
            filtrado=tratar_ohlc(ruidoso,nome,parametros)
            erro=(filtrado.close.iloc[ignorar:]/treino.close.iloc[ignorar:]-1).to_numpy()
            registros.append({"tratamento":nome,"parametros":parametros,"rmse_relativo_treino":float(np.sqrt(np.mean(erro**2)))})
    escolhidos={nome:min([r for r in registros if r["tratamento"]==nome],key=lambda r:r["rmse_relativo_treino"])["parametros"] for nome in ["EMA","Wavelet"]}
    return escolhidos,registros


def _hashes(pasta):
    arquivos=[pasta/"modelo_binario.json",pasta/"modelo_estrategia.json",pasta/"scaler.joblib"]
    info=json.loads(arquivos[0].read_text(encoding="utf-8"))
    arquivos += [pasta/x for x in info["arquivos"]]
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in arquivos}


def _predizer(dados,meta,info,modelos,scaler):
    contexto=criar_contexto(dados,ConfigPrevisao())
    matriz=contexto[list(FEATURES_EVENTOS)].to_numpy(dtype=np.float32)
    pos=meta.posicao_bruta.to_numpy()
    idx=pos[:,None]-np.arange(info["janela"]-1,-1,-1)[None,:]
    if not np.isfinite(matriz[idx]).all():
        raise ValueError("Cenário perdeu atributos válidos; não é permitido remover amostras difíceis.")
    normalizada=scaler.transform(matriz).astype(np.float32)
    x=_entradas(normalizada,pos,info["janela"])
    return prever_scores(info["familia"],modelos,x),contexto


def _bootstrap_pareado(predicoes,meta,plano):
    """Efeito sobre dias, com a mesma reamostragem para todos os cenários."""
    dias=pd.Index(meta.sessao.unique())
    rng=np.random.default_rng(42)
    indices=rng.integers(0,len(dias),size=(plano.bootstrap,len(dias)))
    saida=[]
    for escopo in plano.escopos:
        original=predicoes[(escopo,0.,-1,"sem_tratamento")]
        for sigma in plano.intensidades:
            for tratamento in ["sem_tratamento","EMA","Wavelet"]:
                referencias=[predicoes[(escopo,float(sigma),seed,tratamento)] for seed in plano.sementes]
                ruidosos=[predicoes[(escopo,float(sigma),seed,"sem_tratamento")] for seed in plano.sementes]
                for lado,nome in [(1,"compra"),(2,"venda")]:
                    def diarios(d):
                        a=d[d.operacao_executada & d.sinal.eq(lado)].copy()
                        nn=a.groupby("sessao").size().reindex(dias,fill_value=0).to_numpy()
                        kk=a.assign(venceu=a[f"retorno_liquido_{nome}"]>1e-12).groupby("sessao").venceu.sum().reindex(dias,fill_value=0).to_numpy()
                        return nn,kk
                    on,ok=diarios(original)
                    def reamostrar(conjuntos):
                        quantidades,acertos=zip(*[diarios(d) for d in conjuntos])
                        n=np.stack(quantidades)[:,indices].sum(axis=2)
                        k=np.stack(acertos)[:,indices].sum(axis=2)
                        return np.divide(k,n,out=np.full(k.shape,np.nan),where=n>0)
                    taxas=reamostrar(referencias)
                    ruido=reamostrar(ruidosos)
                    n0,k0=on[indices].sum(axis=1),ok[indices].sum(axis=1)
                    t0=np.divide(k0,n0,out=np.full(len(n0),np.nan),where=n0>0)
                    # Não trata as dez sementes como dez histórias de mercado independentes.
                    for comparacao,diferencas in [("original",taxas-t0[None,:]),("mesmo_ruido_sem_tratamento",taxas-ruido)]:
                        validos=np.isfinite(diferencas).all(axis=0)
                        dist=diferencas[:,validos].mean(axis=0)
                        ci=[float(v) for v in np.quantile(dist,[.025,.975])] if len(dist) else [None,None]
                        saida.append({"escopo":escopo,"intensidade":float(sigma),"tratamento":tratamento,"direcao":nome,
                            "comparacao":comparacao,"delta_taxa_inferior":ci[0],"delta_taxa_superior":ci[1],"reamostragens_validas":int(validos.sum())})
    return saida


def executar_ruido_previsao(csv,artefato,saida,plano=None):
    plano=plano or PlanoRuidoPrevisao()
    plano.validar()
    csv,artefato=Path(csv),Path(artefato)
    hashes=_hashes(artefato)
    info=json.loads((artefato/"modelo_binario.json").read_text(encoding="utf-8"))
    estrategia=json.loads((artefato/"modelo_estrategia.json").read_text(encoding="utf-8"))
    if info["features"]!=list(FEATURES_EVENTOS):
        raise ValueError("Atributos do modelo incompatíveis.")
    fim_treino=info["plano"]["fim_treino"]
    if pd.Timestamp(fim_treino)>=pd.Timestamp(plano.inicio):
        raise ValueError("Avaliação deve começar depois do treino dos filtros/modelo.")
    regra=SaidaBarreiras(**estrategia["regra"])
    custo=info["plano"]["custo"]
    bruto,qualidade=limpar_candles(pd.read_csv(csv))
    bruto=bruto.loc[bruto.time<pd.Timestamp(plano.fim)].reset_index(drop=True)
    ctx=criar_contexto(bruto,ConfigPrevisao())
    matriz=ctx[list(FEATURES_EVENTOS)].to_numpy(dtype=np.float32)
    validas=pd.Series(np.isfinite(matriz).all(axis=1)).rolling(info["janela"]).sum().eq(info["janela"]).to_numpy()
    meta=criar_operacoes_barreiras(ctx,regra,custo)
    mask=(meta.time>=pd.Timestamp(plano.inicio))&(meta.time<pd.Timestamp(plano.fim))&(meta.time_alvo<pd.Timestamp(plano.fim))
    meta=meta.loc[mask & validas[meta.posicao_bruta]].reset_index(drop=True)
    if len(meta)<30:
        raise ValueError("Poucas amostras de avaliação.")
    pos=meta.posicao_bruta.to_numpy()
    referencia_eventos=permissoes_estrategia(ctx.iloc[pos],estrategia["evento"])
    modelos=[joblib.load(artefato/f) if f.endswith("joblib") else tensorflow().keras.models.load_model(artefato/f,compile=False) for f in info["arquivos"]]
    scaler=joblib.load(artefato/"scaler.joblib")
    pasta=Path(saida)/datetime.now(timezone.utc).strftime("ruido_previsao_%Y%m%dT%H%M%S_%fZ")
    for sub in ["predicoes","series","codigo_snapshot"]:
        (pasta/sub).mkdir(parents=True,exist_ok=False)
    for f in Path(__file__).parent.glob("*.py"):
        shutil.copy2(f,pasta/"codigo_snapshot"/f.name)
    salvar_json(pasta/"plano_congelado.json",{"plano":asdict(plano),"csv":str(csv),"csv_sha256":hashlib.sha256(csv.read_bytes()).hexdigest(),
        "artefato":str(artefato),"hashes_artefato":hashes,"qualidade":qualidade,"regra":asdict(regra),"evento":estrategia["evento"],
        "politica":info["politica"],"custo_por_lado":custo,"fim_treino_filtros":fim_treino,
        "natureza":"Exploratório em histórico já observado. Ensaio de sensibilidade de modelo congelado; não há retreinamento.",
        "hipotese_ruido":"Fator 1+sigma*z independente entre candles e comum às quatro colunas OHLC; volume/tempo preservados.",
        "risco_e_execucao":"OHLC original determina preços, ATR, barreiras, duração e lucro. São idênticos em todos os cenários.",
        "escopos":{"rede":"Perturba só as entradas da rede; mantém evento de referência.",
                   "rede_evento":"Perturba entradas da rede e recalcula o evento. O controlador de risco continua no referencial original."}})
    logging.info("Selecionando filtros exclusivamente antes de %s.",fim_treino)
    filtros,busca=selecionar_filtros(bruto,fim_treino,plano)
    salvar_json(pasta/"filtros_selecionados.json",filtros)
    salvar_json(pasta/"busca_filtros_treino.json",busca)
    meta.to_csv(pasta/"referencia_operacoes.csv",index=False)
    p0,_=_predizer(bruto,meta,info,modelos,scaler)
    base_sinais={e:aplicar_politica(p0,info["politica"],*referencia_eventos) for e in plano.escopos}
    # Se disponível no mesmo artefato, verifica reprodução do benchmark conhecido.
    esperado=artefato.parent/"resultado_confirmacao.json"
    auditoria={"benchmark_anterior_compativel":False}
    if esperado.exists() and plano.inicio=="2024-01-01T00:00:00Z" and plano.fim=="2024-07-01T00:00:00Z":
        anterior=json.loads(esperado.read_text(encoding="utf-8"))
        m,_=avaliar_barreiras(meta,base_sinais[plano.escopos[0]],plano.minimo)
        for campo in ["operacoes_compra","lucrativas_compra","operacoes_venda","lucrativas_venda","retorno_acumulado"]:
            if not np.isclose(m[campo],anterior[campo],atol=1e-8,rtol=1e-6):
                raise ValueError(f"O cenário original não reproduziu o benchmark em {campo}.")
        auditoria["benchmark_anterior_compativel"]=True
    registros,predicoes=[],{}
    eval_bruto=(bruto.time>=pd.Timestamp(plano.inicio)).to_numpy()
    def avaliar_cenario(observado,sigma,seed,tratamento):
        p,contexto=_predizer(observado,meta,info,modelos,scaler)
        erro=observado.close.to_numpy()[eval_bruto]/bruto.close.to_numpy()[eval_bruto]-1
        rmse=float(np.sqrt(np.mean(erro**2)))
        nome=f"s{sigma:g}_seed{seed}_{tratamento}"
        serie=observado.loc[eval_bruto,["time",*OHLC]].copy()
        serie.to_csv(pasta/"series"/f"{nome}.csv.gz",index=False,compression="gzip")
        for escopo in plano.escopos:
            masks=referencia_eventos if escopo=="rede" else permissoes_estrategia(contexto.iloc[pos],estrategia["evento"])
            sinais=aplicar_politica(p,info["politica"],*masks)
            m,usados=avaliar_barreiras(meta,sinais,plano.minimo)
            reg={"escopo":escopo,"intensidade":float(sigma),"semente":int(seed),"tratamento":tratamento,
                "rmse_relativo_close":rmse,"fracao_sinais_alterados":float(np.mean(sinais!=base_sinais[escopo])),
                "rmse_scores":float(np.sqrt(np.mean((p[:,1:]-p0[:,1:])**2))),**m}
            registros.append(reg)
            d=meta.copy()
            d["sinal"]=sinais
            d["score_compra"],d["score_venda"]=p[:,1],p[:,2]
            d["operacao_executada"]=False
            d.loc[usados,"operacao_executada"]=True
            d.to_csv(pasta/"predicoes"/f"{escopo}_{nome}.csv.gz",index=False,compression="gzip")
            predicoes[(escopo,float(sigma),int(seed),tratamento)]=d[["sessao","sinal","operacao_executada","retorno_liquido_compra","retorno_liquido_venda"]]
    avaliar_cenario(bruto,0.,-1,"sem_tratamento")
    for tratamento in ["EMA","Wavelet"]:
        avaliar_cenario(tratar_ohlc(bruto,tratamento,filtros[tratamento]),0.,-1,tratamento)
    for seed in plano.sementes:
        for sigma in plano.intensidades:
            ruidoso=perturbar_ohlc(bruto,sigma,seed)
            avaliar_cenario(ruidoso,sigma,seed,"sem_tratamento")
            for tratamento in ["EMA","Wavelet"]:
                avaliar_cenario(tratar_ohlc(ruidoso,tratamento,filtros[tratamento]),sigma,seed,tratamento)
            pd.DataFrame(registros).to_csv(pasta/"resultados_cenarios.csv",index=False)
            logging.info("Ruído %g semente %d concluído (rede e evento conforme plano).",sigma,seed)
    tabela=pd.DataFrame(registros)
    resumo=tabela.groupby(["escopo","intensidade","tratamento"],sort=True).agg(
        execucoes=("semente","size"),compra_media=("taxa_lucro_compra","mean"),compra_desvio=("taxa_lucro_compra","std"),
        venda_media=("taxa_lucro_venda","mean"),venda_desvio=("taxa_lucro_venda","std"),
        compras_media=("operacoes_compra","mean"),vendas_media=("operacoes_venda","mean"),
        retorno_medio=("retorno_acumulado","mean"),retorno_desvio=("retorno_acumulado","std"),
        drawdown_medio=("drawdown_saidas","mean"),rmse_relativo=("rmse_relativo_close","mean"),
        sinais_alterados=("fracao_sinais_alterados","mean"),semente_meta70=("meta_70_lucro_ambas","sum"),
        semente_aprovada=("aprovado","sum"),sementes_suporte=("suporte_suficiente","sum")).reset_index()
    resumo.to_csv(pasta/"resumo_sementes.csv",index=False)
    ic=_bootstrap_pareado(predicoes,meta,plano)
    pd.DataFrame(ic).to_csv(pasta/"intervalos_pareados.csv",index=False)
    if hashes!=_hashes(artefato):
        raise ValueError("O modelo de referência mudou durante o experimento.")
    auditoria.update({"modelo_scaler_politica_inalterados":True,"amostras_comuns":len(meta),"linhas_resultado":len(tabela),
        "filtros_ajustados_so_no_treino":True,"precos_execucao_e_alvos_comuns":True,"meta_confirmada":False})
    salvar_json(pasta/"auditoria.json",auditoria)
    _graficos(pasta,resumo)
    _relatorio(pasta,resumo,tabela,plano,filtros,ic)
    salvar_json(pasta/"CONCLUIDO.json",{"concluido":True,"exploratorio":True,"meta_confirmada":False,**auditoria})
    return pasta,tabela,resumo


def _graficos(pasta,resumo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for escopo in resumo.escopo.unique():
        sub=resumo[resumo.escopo==escopo]
        fig,eixos=plt.subplots(1,2,figsize=(11,4),sharey=True)
        for ax,lado in zip(eixos,["compra","venda"]):
            for tratamento in ["sem_tratamento","EMA","Wavelet"]:
                d=sub[(sub.intensidade>0)&sub.tratamento.eq(tratamento)].sort_values("intensidade")
                xx=d.intensidade.to_numpy()*100
                yy=d[f"{lado}_media"].to_numpy()*100
                sd=d[f"{lado}_desvio"].fillna(0).to_numpy()*100
                ax.plot(xx,yy,"o-",label=tratamento)
                ax.fill_between(xx,yy-sd,yy+sd,alpha=.12)
            b=sub[sub.intensidade.eq(0)&sub.tratamento.eq("sem_tratamento")].iloc[0]
            ax.axhline(b[f"{lado}_media"]*100,color="black",linestyle="--",label="Original")
            ax.axhline(70,color="gray",linestyle=":",label="70%")
            ax.set(xlabel="Desvio-padrão do ruído (%)",title=lado.capitalize(),ylim=(0,100))
            ax.grid(alpha=.2)
        eixos[0].set_ylabel("Operações lucrativas (%)")
        eixos[1].legend(fontsize=8)
        fig.suptitle(f"Modelo congelado — {escopo}; faixa: ±1 desvio entre sementes")
        fig.tight_layout()
        fig.savefig(pasta/f"taxas_{escopo}.png",dpi=160)
        plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    s=resumo[resumo.escopo.eq(resumo.escopo.iloc[0])&resumo.intensidade.gt(0)]
    for tratamento in ["sem_tratamento","EMA","Wavelet"]:
        d=s[s.tratamento.eq(tratamento)].sort_values("intensidade")
        axs[0].plot(d.intensidade*100,d.rmse_relativo*100,"o-",label=tratamento)
        axs[1].plot(d.intensidade*100,d.retorno_medio*100,"o-",label=tratamento)
    axs[0].set(ylabel="RMSE relativo do fechamento (%)",xlabel="Desvio-padrão do ruído (%)",title="Reconstrução da entrada")
    axs[1].set(ylabel="Retorno líquido médio (%)",xlabel="Desvio-padrão do ruído (%)",title="Resultado financeiro — rede")
    for ax in axs:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(pasta/"reconstrucao_e_retorno.png",dpi=160)
    plt.close(fig)


def _relatorio(pasta,resumo,tabela,plano,filtros,intervalos):
    pct=lambda x:"NA" if pd.isna(x) else f"{x:.2%}"
    texto=["# Ruído e tratamentos na previsão de compras e vendas", "",
        "**Ensaio exploratório de sensibilidade de um modelo congelado.** O modelo de referência não atingiu 70% nas duas direções na confirmação anterior. O experimento mede perturbação e recuperação das entradas; não certifica um preditor lucrativo nem constitui nova confirmação independente.", "",
        f"Período: {plano.inicio} até antes de {plano.fim}. São {len(plano.sementes)} sementes de ruído, quatro intensidades por padrão e três condições por intensidade (sem tratamento, EMA, Wavelet), além dos controles sem ruído artificial. A mesma sequência gaussiana é escalada nas intensidades de uma semente. As sementes são repetições de perturbação sobre o mesmo mercado, não novas amostras independentes.", "",
        "## Protocolo", "",
        "A referência é a MLP preservada com doze candles e 31 atributos, previsão binária de lucro em cada direção, entrada seguinte e saída por ATR/alvo/stop/prazo. Pesos, scaler e limiares de emissão não são ajustados. Há taxas fixas de 0,02% por lado, uma posição por vez e execução em OHLC original, sem spread ou slippage. Todos os alvos, níveis de risco, durações potenciais e preços de execução são os mesmos nos cenários.", "",
        "O ruído é multiplicativo, 1 + sigma × z, com um fator comum a abertura, máxima, mínima e fechamento de cada candle. Isso preserva a coerência OHLC e representa erro comum de nível, não erros independentes em cada preço. Volume de ticks e horários não são alterados. Os atributos são recalculados. EMA/Wavelet tratam o fechamento observado e deslocam proporcionalmente as quatro colunas do candle, preservando sua geometria relativa.", "",
        f"Os filtros foram selecionados por menor RMSE relativo de reconstrução somente no treino original do modelo, com sigma {plano.intensidade_filtros:g} e semente separada {plano.semente_filtros}. São congelados para todas as intensidades e sementes de avaliação: `{filtros}`. Não foram herdados os parâmetros escolhidos pelo estudo antigo de 2026.", "",
        "São dois escopos: `rede` mantém o evento de entrada original para isolar a sensibilidade da rede; `rede_evento` também recalcula a condição de entrada com a observação contaminada/tratada. Em ambos, o controlador de risco e a execução ficam no referencial original. Portanto, o segundo escopo não simula corrupção integral de todos os subsistemas de uma plataforma. A alteração de quais operações são executadas é parte do resultado, inclusive seus efeitos de sobreposição.", "",
        "Original+EMA e Original+Wavelet são controles: mostram a distorção que o filtro provoca sem a perturbação artificial. O preço observado não é tratado como o sinal latente verdadeiro do mercado. Os rótulos de lucro nunca são calculados a partir de preços perturbados ou filtrados.", "",
        "## Resultados médios entre sementes", "",
        "Quantidades são médias de operações por execução; taxas são médias das taxas por semente, não uma soma artificial de operações sobre o mesmo período. A última coluna mostra quantas sementes atingiram 70% com pelo menos vinte compras e vinte vendas.", ""]
    for escopo in plano.escopos:
        texto += [f"### Escopo `{escopo}`", "",
            "| Ruído sigma | Tratamento | Compras médias | % compra | Vendas médias | % venda | Retorno médio | Sementes com meta |",
            "|---:|---|---:|---:|---:|---:|---:|---:|"]
        for r in resumo[resumo.escopo.eq(escopo)].itertuples():
            texto.append(f"| {r.intensidade:.4f} | {r.tratamento} | {r.compras_media:.1f} | {pct(r.compra_media)} | {r.vendas_media:.1f} | {pct(r.venda_media)} | {r.retorno_medio:.2%} | {r.semente_meta70}/{r.execucoes} |")
    texto += ["", "## Incerteza e limites", "",
        f"`intervalos_pareados.csv` contém {plano.bootstrap} reamostragens pareadas de sessões. Compara diferenças de taxa contra a referência original e contra a mesma realização sem tratamento, usando os mesmos dias e a média entre sementes. Réplicas sem operações em alguma semente/direção são excluídas e contadas. São intervalos descritivos condicionais a este histórico, sem correção pela pesquisa anterior ou multiplicidade de comparações.", "",
        "Reduzir RMSE da entrada não garante melhorar previsão ou lucro. Uma taxa acima de 70% com poucas operações não aprova a meta; uma taxa acima de 70% também pode coexistir com prejuízo. O arquivo detalhado registra suporte, lucro médio, ganhos/perdas, profit factor, drawdown nas saídas, sinais alterados e aprovação econômica.", "",
        "Não houve retreinamento por cenário, ajuste dos limiares pela avaliação nem escolha da melhor semente para promoção do modelo. Este ensaio responde à sensibilidade de um previsor existente; comparar pipelines treinados independentemente em entradas ruidosas é outro experimento.", "",
        "## Arquivos e reprodução", "",
        "`plano_congelado.json`, `filtros_selecionados.json`, `busca_filtros_treino.json`, `referencia_operacoes.csv`, `resultados_cenarios.csv`, `resumo_sementes.csv`, `intervalos_pareados.csv`, `auditoria.json`, `series/`, `predicoes/` e os gráficos PNG preservam o ensaio. Os modelos originais não são sobrescritos. `codigo_snapshot/` contém o código usado.", "",
        "```powershell", ".\\.venv\\Scripts\\python.exe scripts\\experimentar_ruido_previsao.py", "```"]
    (pasta/"LEIA_RESULTADOS.md").write_text("\n".join(texto)+"\n",encoding="utf-8")
