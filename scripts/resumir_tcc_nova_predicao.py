"""Audita a réplica adaptada do TCC e consolida os resultados já executados."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import joblib
import numpy as np
import pandas as pd

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from src.binary_events import FEATURES_EVENTOS
from src.bracket_strategy import avaliar_barreiras
from src.forecast_data import limpar_candles
from src.tcc_noise_training import contexto_cenario,metricas_binarias


def auditar(pasta):
    doc=json.loads((pasta/"plano_congelado.json").read_text(encoding="utf-8"))
    tabela=pd.read_csv(pasta/"comparacao_cenarios.csv")
    ref=pd.read_csv(pasta/"referencia_alvos.csv")
    part=pd.read_csv(pasta/"particoes.csv")
    tarefas=json.loads((pasta/"modelos_antes_teste.json").read_text(encoding="utf-8"))
    assert len(tabela)==len(tarefas)==7*len(doc["plano"]["sementes_redes"])
    assert np.array_equal(ref.posicao_bruta,part.posicao_bruta)
    teste=ref.loc[part.particao.eq("teste")].reset_index(drop=True)
    financeiro=["posicao_bruta","time","time_alvo","preco_entrada","preco_saida_compra","preco_saida_venda",
        "retorno_liquido_compra","retorno_liquido_venda","duracao_compra","duracao_venda","alvo_fracao","stop_fracao"]
    hashes={}
    antes=(pasta/"modelos_antes_teste.json").stat().st_mtime
    for linha in tabela.to_dict("records"):
        nome,seed=linha["cenario"],linha["semente"]
        pred=pd.read_csv(pasta/"predicoes"/f"{nome}_semente_{seed}.csv.gz")
        pd.testing.assert_frame_equal(pred[financeiro],teste[financeiro])
        m,usados=avaliar_barreiras(pred,pred.sinal,doc["plano"]["minimo"],doc["plano"]["meta"])
        for k,v in m.items():
            assert (v is None and pd.isna(linha[k])) or np.isclose(v,linha[k],atol=1e-10,rtol=1e-9),(nome,seed,k)
        assert np.array_equal(usados,np.flatnonzero(pred.operacao_executada))
        cls=metricas_binarias(pred,pred.sinal)
        for lado in ["compra","venda"]:
            for k in ["FP","FN","TP","TN","precision","recall","f1","FPR","FDR"]:
                v=cls[lado][k]
                assert (v is None and pd.isna(linha[f"sinais_{k}_{lado}"])) or np.isclose(v,linha[f"sinais_{k}_{lado}"],atol=1e-10,rtol=1e-9)
        modelo=pasta/"modelos"/nome/f"semente_{seed}/rede_0.keras"
        assert modelo.stat().st_mtime<=antes
        hashes[f"{nome}/{seed}"]=hashlib.sha256(modelo.read_bytes()).hexdigest()
    bruto,_=limpar_candles(pd.read_csv(doc["csv"]))
    treino_pos=ref.loc[part.particao.eq("treino"),"posicao_bruta"].to_numpy()
    fit=np.unique((treino_pos[:,None]-np.arange(doc["plano"]["janela"]-1,-1,-1)[None,:]).ravel())
    assert bruto.time.iloc[fit].max()<pd.Timestamp(doc["limites"]["inicio_validacao"])
    for nome in doc["cenarios"]:
        d=pd.read_csv(pasta/"series"/f"{nome}.csv.gz")
        for col in ["open","high","low","close","tick_volume"]:
            np.testing.assert_allclose(d[col],bruto[col],atol=1e-12,rtol=1e-12)
        ctx=contexto_cenario(bruto,d.close_modelo)
        matriz=ctx[list(FEATURES_EVENTOS)].to_numpy(dtype=np.float32)
        for seed in doc["plano"]["sementes_redes"]:
            scaler=joblib.load(pasta/"modelos"/nome/f"semente_{seed}/scaler.joblib")
            # StandardScaler usa acumulação em float64 mesmo com entrada float32.
            np.testing.assert_allclose(scaler.mean_,matriz[fit].astype(np.float64).mean(axis=0),atol=1e-9,rtol=1e-9)
            assert int(scaler.n_samples_seen_)==len(fit)
    auditoria={"modelos_verificados":len(tarefas),"testes_com_mesmos_alvos":True,"ohl_volume_originais":True,
        "scalers_reproduzidos_so_no_treino":True,"metricas_financeiras_e_fp_reproduzidas":True,
        "todos_modelos_salvos_antes_de_abrir_teste":True,"amostras_teste":len(teste),"hashes_modelos":hashes,"meta_confirmada":False}
    (pasta/"auditoria.json").write_text(json.dumps(auditoria,ensure_ascii=False,indent=2),encoding="utf-8")
    return auditoria


def grafico_fp(pasta,tabela):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m=tabela.groupby("cenario",sort=False).mean(numeric_only=True)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    x=np.arange(len(m))
    for ax,metrica in zip(axs,["FPR","FDR"]):
        for lado,delta in [("compra",-.18),("venda",.18)]:
            ax.bar(x+delta,m[f"sinais_{metrica}_{lado}"]*100,width=.36,label=lado)
        ax.set(xticks=x,xticklabels=m.index,ylabel=f"{metrica} dos sinais (%)",title=f"{metrica} — média entre sementes")
        ax.tick_params(axis="x",rotation=35,labelsize=8)
        ax.legend()
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("Antes do controle de sobreposição; não confundir FPR com FDR")
    fig.tight_layout()
    fig.savefig(pasta/"graficos/falsos_positivos.png",dpi=160)
    plt.close(fig)


def gerar(pasta,arquivo):
    pasta=Path(pasta)
    audit=auditar(pasta)
    doc=json.loads((pasta/"plano_congelado.json").read_text(encoding="utf-8"))
    tabela=pd.read_csv(pasta/"comparacao_cenarios.csv")
    resumo=pd.read_csv(pasta/"resumo_sementes.csv")
    diag=pd.read_csv(pasta/"diagnostico_filtros.csv")
    grafico_fp(pasta,tabela)
    pct=lambda x:"NA" if pd.isna(x) else f"{x:.2%}"
    texto=["# Experimento do TCC com treinamento da nova previsão", "",
        f"**Concluído: sete cenários, {len(doc['plano']['sementes_redes'])} sementes de rede, {audit['modelos_verificados']} treinamentos independentes.** Cada condição tem seu próprio scaler, pesos, Early Stopping e limiares ajustados na validação. Não foi reutilizada uma única rede congelada entre os cenários.", "",
        "A arquitetura preservada é a MLP da previsão nova, não a LSTM da tarefa antiga de ±0,8%. O alvo continua sendo lucro líquido de compra/venda com as regras de entrada e saída novas. O procedimento experimental do TCC foi transferido, com as adaptações explicitadas abaixo.", "",
        f"Fonte: {doc['qualidade']['linhas_validas']} candles PETR4 M5, de {doc['inicio_dados']} a {doc['fim_dados']}, no snapshot MT5 preservado. Divisão cronológica 70%/15%/15% do universo comum antes dos expurgos. O teste começa em **{doc['limites']['inicio_teste']}** e tem {audit['amostras_teste']} amostras. Portanto, não é o mesmo teste de janeiro–junho/2024 dos ensaios anteriores, e as taxas não representam comparação direta de melhora/piora com aqueles números.", "",
        "## Resultado médio entre sementes de rede", "",
        "| Cenário | % compras lucrativas | % vendas lucrativas | Retorno líquido médio |",
        "|---|---:|---:|---:|"]
    for r in resumo.itertuples():
        texto.append(f"| {r.cenario} | {pct(r.compra_media)} | {pct(r.venda_media)} | {r.retorno_medio:.2%} |")
    texto += ["",f"Execuções que atingiram 70% nas duas direções com suporte: {int(tabela.meta_70_lucro_ambas.sum())}/{len(tabela)}. Aprovação econômica: {int(tabela.aprovado.sum())}/{len(tabela)}. As repetições usam o mesmo mercado e a mesma realização de ruído; variam a inicialização da rede. Não foram escolhidas sementes por desempenho no teste.", "",
        "## Falsos positivos dos sinais potenciais", "",
        "FPR = FP/(FP+TN); FDR = FP/(TP+FP). São médias dos indicadores por rede, antes do controle de sobreposição. A FDR das operações efetivamente executadas é calculada separadamente e equivale a um menos sua taxa de lucro. As duas respostas de lucro são binárias, não uma classificação 3×3 artificial.", "",
        "| Cenário | FPR compra | FDR compra | FPR venda | FDR venda |",
        "|---|---:|---:|---:|---:|"]
    m=tabela.groupby("cenario",sort=False).mean(numeric_only=True)
    for nome,r in m.iterrows():
        texto.append(f"| {nome} | {pct(r.sinais_FPR_compra)} | {pct(r.sinais_FDR_compra)} | {pct(r.sinais_FPR_venda)} | {pct(r.sinais_FDR_venda)} |")
    texto += ["", "## O que os filtros fizeram", ""]
    for r in diag[diag.particao.eq("teste")].itertuples():
        texto.append(f"- {r.metodo}: redução de {r.reducao_rmse_percentual:.2f}% no RMSE de reconstrução contra a série ruidosa 0,002.")
    texto += ["", "No teste desta execução, a redução de RMSE não se traduziu em melhora das médias de acerto de compra/venda em relação ao ruído 0,002 sem filtro. O experimento mede o efeito do pipeline completo reaprendido por condição, incluindo pesos, normalização e política de emissão; não isola o efeito de um filtro em pesos fixos.", "",
        "## Auditoria e verificações", "",
        "Foram reproduzidos os normalizadores a partir das linhas de treino, os resultados financeiros e as métricas de falsos positivos. Todos os modelos estavam salvos antes de abrir o teste. As datas, respostas, OHL/volume e preços de execução foram conferidos entre os sete cenários. O fechamento perturbado fica exclusivamente em `close_modelo`; o fechamento original é preservado nos CSVs.", ""]
    log=(RAIZ/"TESTES_PREVISAO.log").read_text(encoding="utf-8")
    testes=re.search(r"Ran (\d+) tests in ([\d.]+)s\s+OK",log)
    if testes:
        texto += [f"**{testes[1]} testes automatizados passaram**, em {float(testes[2]):.1f} segundos, incluindo equivalência numérica/causal da Wavelet em lotes, geração dos sete cenários, preservação dos dados originais, treinamento independente e recarga do cenário correto.", ""]
    relativo=pasta.relative_to(RAIZ) if pasta.is_absolute() and pasta.is_relative_to(RAIZ) else pasta
    texto += [f"Execução completa: `{relativo.as_posix()}`. `auditoria.json` registra as verificações e hashes dos modelos. O histórico já foi inspecionado, portanto os resultados são exploratórios e não certificam uma nova estratégia independente.", "",
        f"![Falsos positivos por cenário]({relativo.as_posix()}/graficos/falsos_positivos.png)", "",
        f"![Taxas de operações lucrativas por cenário]({relativo.as_posix()}/graficos/taxas_cenarios.png)", "",
        "---", "",(pasta/"LEIA_RESULTADOS.md").read_text(encoding="utf-8")]
    Path(arquivo).write_text("\n".join(texto)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("pasta",type=Path)
    p.add_argument("--arquivo",type=Path,default=RAIZ/"RESULTADOS_TCC_NOVA_PREDICAO.md")
    a=p.parse_args()
    gerar(a.pasta,a.arquivo)
    print(a.arquivo)
