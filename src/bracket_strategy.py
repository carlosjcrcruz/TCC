"""Previsão de lucro com saídas por alvo, stop ou prazo, sem ordens reais."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from src.binary_events import (FEATURES_EVENTOS, PlanoBinario, ajustar_modelo,
    criar_contexto, permissoes, prever_scores, salvar_binario)
from src.binary_study import _entradas, _partes
from src.forecast_config import ConfigPrevisao
from src.forecast_data import criar_alvos, limpar_candles
from src.forecast_experiment import salvar_json
from src.trade_policy import aplicar_politica


EVENTOS_ESTRATEGIA = ("todos", "choque_reversao", "rsi_extremo", "rompimento",
                     "reversao_media", "recuo_tendencia")


@dataclass
class CabecaValor:
    """Adapta retorno esperado para um escore de ranking, não probabilidade."""
    estimador: object
    escala: float

    @property
    def classes_(self):
        return np.array([0, 1])

    def predict_proba(self, x):
        z = np.clip(self.estimador.predict(x)/self.escala, -30., 30.)
        s = 1/(1+np.exp(-z))
        return np.column_stack([1-s, s])


def ajustar_valor(xt, retornos):
    cabecas = []
    for lado in range(2):
        y = retornos[:,lado]
        escala = max(float(np.std(y)), 1e-6)
        modelo = HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=15,
            min_samples_leaf=50, l2_regularization=5., early_stopping=False, random_state=42)
        modelo.fit(xt, y/escala)
        # O regressor prevê em unidades de desvio-padrão do retorno de treino.
        cabecas.append(CabecaValor(modelo, 1.))
    return cabecas


@dataclass(frozen=True)
class SaidaBarreiras:
    horizonte: int = 20
    tipo: str = "fixa"
    alvo: float = .002
    razao_stop: float = 2.

    def validar(self):
        if self.horizonte < 1 or self.tipo not in {"fixa", "atr"}:
            raise ValueError("Prazo ou tipo de barreira inválido.")
        if not np.isfinite(self.alvo) or not np.isfinite(self.razao_stop) or self.alvo <= 0 or self.razao_stop <= 0:
            raise ValueError("Alvo e razão de stop devem ser positivos e finitos.")
        if self.tipo == "fixa" and self.alvo * max(1, self.razao_stop) >= 1:
            raise ValueError("Barreiras de preço inválidas.")


def amplitude_passada(ctx):
    anterior = ctx.close.shift(1)
    tr = pd.concat([ctx.high-ctx.low, (ctx.high-anterior).abs(), (ctx.low-anterior).abs()], axis=1).max(axis=1)
    # Inclui apenas o candle já fechado no instante da previsão.
    return tr.rolling(20).mean() / ctx.close


def distancias_barreiras(ctx, regra, custo=.0002):
    regra.validar()
    alvo = np.full(len(ctx), regra.alvo) if regra.tipo == "fixa" else amplitude_passada(ctx).to_numpy() * regra.alvo
    # Um take-profit precisa pagar as taxas inclusive na direção mais onerosa.
    piso = max(.0008, 2*custo/(1-custo) + .0001)
    alvo = np.maximum(alvo, piso)
    return alvo, alvo * regra.razao_stop


def criar_operacoes_barreiras(ctx, regra, custo=.0002):
    """OHLC: gap de stop executa no open; alvo limita preço; empate intrabar perde.

    O universo exige todo o prazo contínuo na mesma sessão, inclusive quando
    uma barreira teria sido atingida antes. Isso evita selecionar elegibilidade
    pelo próprio resultado da operação.
    """
    regra.validar()
    if not 0 <= custo < 1:
        raise ValueError("Custo inválido.")
    h, n = regra.horizonte, len(ctx)
    base = criar_alvos(ctx, ConfigPrevisao(horizonte=h))
    alvo, stop = distancias_barreiras(ctx, regra, custo)
    entrada = ctx.open.shift(-1).to_numpy()
    elegivel = base.alvo_elegivel.to_numpy() & np.isfinite(alvo) & (np.maximum(alvo, stop) < 1)
    m = base[["time", "time_alvo", "sessao"]].copy()
    m["posicao_bruta"] = np.arange(n)
    m["preco_entrada"] = entrada
    m["alvo_fracao"], m["stop_fracao"] = alvo, stop
    for lado, nome in [(1, "compra"), (-1, "venda")]:
        tp = entrada * (1 + lado*alvo)
        sl = entrada * (1 - lado*stop)
        ativo = elegivel.copy()
        saida = ctx.close.shift(-h).to_numpy().copy()
        duracao = np.full(n, h, dtype=int)
        motivo = np.full(n, "prazo", dtype=object)
        ambiguo = np.zeros(n, dtype=bool)
        for k in range(1, h+1):
            op = ctx.open.shift(-k).to_numpy()
            hi = ctx.high.shift(-k).to_numpy()
            lo = ctx.low.shift(-k).to_numpy()
            gap_stop = ativo & ((op <= sl) if lado == 1 else (op >= sl))
            gap_alvo = ativo & ~gap_stop & ((op >= tp) if lado == 1 else (op <= tp))
            toca_stop = (lo <= sl) if lado == 1 else (hi >= sl)
            toca_alvo = (hi >= tp) if lado == 1 else (lo <= tp)
            restante = ativo & ~gap_stop & ~gap_alvo
            stop_barra = restante & toca_stop
            alvo_barra = restante & ~toca_stop & toca_alvo
            ambiguo |= restante & toca_stop & toca_alvo
            saida[gap_stop] = op[gap_stop]
            saida[gap_alvo | alvo_barra] = tp[gap_alvo | alvo_barra]
            saida[stop_barra] = sl[stop_barra]
            encerra = gap_stop | gap_alvo | stop_barra | alvo_barra
            duracao[encerra] = k
            motivo[gap_stop | stop_barra] = "stop"
            motivo[gap_alvo | alvo_barra] = "alvo"
            ativo[encerra] = False
        q = saida / entrada
        m[f"retorno_liquido_{nome}"] = lado * (q-1) - custo*(1+q)
        m[f"preco_saida_{nome}"] = saida
        m[f"duracao_{nome}"] = duracao
        m[f"motivo_{nome}"] = motivo
        m[f"ambiguo_{nome}"] = ambiguo
    return m.loc[elegivel].reset_index(drop=True)


def permissoes_estrategia(ctx, evento):
    if evento in {"todos", "choque_reversao", "rsi_extremo", "rompimento"}:
        return permissoes(ctx, evento)
    if evento == "reversao_media":
        vol = ctx.vol_anterior
        return ((ctx.desvio_media_ticks < -vol) & (ctx.rsi14 < .4)).to_numpy(), ((ctx.desvio_media_ticks > vol) & (ctx.rsi14 > .6)).to_numpy()
    if evento == "recuo_tendencia":
        return ((ctx.retorno_20 > 0) & (ctx.retorno_1 < 0) & (ctx.rsi14 < .6)).to_numpy(), ((ctx.retorno_20 < 0) & (ctx.retorno_1 > 0) & (ctx.rsi14 > .4)).to_numpy()
    raise ValueError("Evento desconhecido.")


def avaliar_barreiras(meta, sinais, minimo=20, meta_acerto=.7):
    sinais = np.asarray(sinais)
    if len(sinais) != len(meta) or not np.isin(sinais, [0, 1, 2]).all() or not meta.posicao_bruta.is_monotonic_increasing:
        raise ValueError("Sinais ou posições inválidos.")
    fim, usados = -1, []
    pos = meta.posicao_bruta.to_numpy()
    dc, dv = meta.duracao_compra.to_numpy(), meta.duracao_venda.to_numpy()
    for j in np.flatnonzero(sinais):
        if pos[j] < fim:
            continue
        usados.append(j)
        fim = pos[j] + (dc[j] if sinais[j] == 1 else dv[j])
    usados = np.asarray(usados, dtype=int)
    direcao = sinais[usados]
    rs = np.where(direcao == 1, meta.retorno_liquido_compra.to_numpy()[usados], meta.retorno_liquido_venda.to_numpy()[usados])
    if not np.isfinite(rs).all():
        raise ValueError("Retorno indisponível.")
    patrimonio = np.r_[1., np.cumprod(1+rs)]
    ganhos, perdas = rs[rs > 1e-12].sum(), -rs[rs < -1e-12].sum()
    r = {"operacoes": len(rs), "taxa_lucro_total": float(np.mean(rs > 1e-12)) if len(rs) else None,
         "retorno_acumulado": float(patrimonio[-1]-1), "lucro_medio": float(rs.mean()) if len(rs) else None,
         "drawdown_saidas": float(np.max(1-patrimonio/np.maximum.accumulate(patrimonio))),
         "profit_factor": float(ganhos/perdas) if perdas > 0 else None,
         "cobertura_sinais": float(np.mean(sinais != 0)) if len(sinais) else 0.,
         "fracao_casos_ambiguos": float(np.mean(np.where(direcao == 1, meta.ambiguo_compra.to_numpy()[usados], meta.ambiguo_venda.to_numpy()[usados]))) if len(rs) else 0.}
    for classe, nome in [(1, "compra"), (2, "venda")]:
        x = rs[direcao == classe]
        r[f"operacoes_{nome}"], r[f"lucrativas_{nome}"] = len(x), int((x > 1e-12).sum())
        r[f"taxa_lucro_{nome}"] = float(np.mean(x > 1e-12)) if len(x) else None
        r[f"lucro_medio_{nome}"] = float(x.mean()) if len(x) else None
        r[f"ganho_medio_{nome}"] = float(x[x > 1e-12].mean()) if (x > 1e-12).any() else None
        r[f"perda_media_{nome}"] = float(x[x <= 1e-12].mean()) if (x <= 1e-12).any() else None
    r["suporte_suficiente"] = all(r[f"operacoes_{n}"] >= minimo for n in ["compra", "venda"])
    r["meta_70_lucro_ambas"] = bool(r["suporte_suficiente"] and all(r[f"taxa_lucro_{n}"] >= meta_acerto for n in ["compra", "venda"]))
    r["expectativa_positiva_ambas"] = bool(r["suporte_suficiente"] and all(r[f"lucro_medio_{n}"] > 0 for n in ["compra", "venda"]))
    r["aprovado"] = bool(r["meta_70_lucro_ambas"] and r["expectativa_positiva_ambas"] and r["retorno_acumulado"] > 0)
    return r, usados


def pontuar_barreiras(m):
    if not m["suporte_suficiente"]:
        return (-1, -1, -1., -1., m["operacoes"])
    return (int(m["aprovado"]), int(m["expectativa_positiva_ambas"]),
            min(m["taxa_lucro_compra"], m["taxa_lucro_venda"]), m["retorno_acumulado"], m["operacoes"])


def selecionar_politica_barreiras(meta, p, masks, minimo=20, meta_acerto=.7):
    quantis = [0, .25, .5, .65, .75, .85, .9, .95]
    cortes = [np.unique(np.quantile(p[mask, k], quantis)).tolist() if mask.any() else [None] for k, mask in zip([1, 2], masks)]
    melhor = None
    for c in cortes[0]:
        for v in cortes[1]:
            pol = {"limiar_compra": c, "limiar_venda": v}
            m, _ = avaliar_barreiras(meta, aplicar_politica(p, pol, *masks), minimo, meta_acerto)
            if melhor is None or pontuar_barreiras(m) > pontuar_barreiras(melhor[1]):
                melhor = pol, m
    return melhor


def reservar_confirmacao(csv, saida, plano, pasta, escolha):
    """Uma avaliação por snapshot/período nesta família de buscas.

    O marcador é reservado antes de prever a confirmação. Reexecutar a busca
    não pode liberar o mesmo teste para escolher outra estratégia.
    """
    origem = hashlib.sha256(Path(csv).read_bytes()).hexdigest()
    chave = hashlib.sha256(f"{origem}|{plano.fim_selecao}|{plano.fim_teste}".encode()).hexdigest()
    raiz = Path(saida)/"confirmacoes_utilizadas"
    raiz.mkdir(parents=True, exist_ok=True)
    try:
        with (raiz/f"{chave}.json").open("x", encoding="utf-8") as f:
            json.dump({"csv_sha256": origem, "inicio": plano.fim_selecao, "fim": plano.fim_teste,
                "execucao": str(pasta), "escolha": escolha,
                "reservado_em": datetime.now(timezone.utc).isoformat()}, f, ensure_ascii=False, indent=2)
        return True
    except FileExistsError:
        return False


def executar_busca_estrategias(csv, saida, plano, regras, eventos=EVENTOS_ESTRATEGIA, objetivo="taxa", confirmar=True):
    plano.validar()
    if objetivo not in {"taxa", "valor"} or (objetivo == "valor" and plano.familias != ("hist_binario",)):
        raise ValueError("Objetivo valor usa exclusivamente hist_binario (regressores de retorno).")
    if not regras or not eventos or not set(eventos) <= set(EVENTOS_ESTRATEGIA):
        raise ValueError("Informe regras e eventos válidos.")
    for regra in regras:
        regra.validar()
    csv = Path(csv)
    bruto, qualidade = limpar_candles(pd.read_csv(csv))
    bruto = bruto.loc[bruto.time < pd.Timestamp(plano.fim_teste)].reset_index(drop=True)
    ctx = criar_contexto(bruto, ConfigPrevisao())
    matriz = ctx[list(FEATURES_EVENTOS)].to_numpy(dtype=np.float32)
    valido = np.isfinite(matriz).all(axis=1)
    fit = (ctx.time < pd.Timestamp(plano.fim_treino)).to_numpy() & valido
    scaler = StandardScaler().fit(matriz[fit])
    normalizada = scaler.transform(matriz).astype(np.float32)
    janelas_ok = pd.Series(valido).rolling(12).sum().eq(12).to_numpy()
    pasta = Path(saida)/datetime.now(timezone.utc).strftime("estrategias_%Y%m%dT%H%M%S_%fZ")
    pasta.mkdir(parents=True, exist_ok=False)
    (pasta/"codigo_snapshot").mkdir()
    for f in Path(__file__).parent.glob("*.py"):
        shutil.copy2(f, pasta/"codigo_snapshot"/f.name)
    salvar_json(pasta/"plano_congelado.json", {"plano": asdict(plano), "regras": [asdict(x) for x in regras],
        "eventos": eventos, "objetivo": objetivo, "csv": str(csv), "sha256_csv": hashlib.sha256(csv.read_bytes()).hexdigest(),
        "qualidade": qualidade, "criterio": "70% e >=20 operações por direção, expectativa positiva em ambas e retorno acumulado positivo; só então abrir confirmação uma vez.",
        "intrabar": "stop primeiro quando ambos os limites tocam; gap de stop no open; gap de alvo no limite"})
    linhas, melhor, estado = [], None, None
    for numero, regra in enumerate(regras):
        meta = criar_operacoes_barreiras(ctx, regra, plano.custo)
        meta = meta.loc[janelas_ok[meta.posicao_bruta]].reset_index(drop=True)
        partes = _partes(meta, plano)
        mp = {k: meta.iloc[v].reset_index(drop=True) for k,v in partes.items()}
        y = (meta[["retorno_liquido_compra", "retorno_liquido_venda"]].to_numpy() > 1e-12).astype(np.float32)
        for familia in plano.familias:
            janela = 12 if familia == "mlp_binario" else 1
            x = {k: _entradas(normalizada, m.posicao_bruta.to_numpy(), janela) for k,m in mp.items() if k != "teste"}
            if objetivo == "taxa":
                modelos, hist = ajustar_modelo(familia, x["treino"], y[partes["treino"]], x["early"], y[partes["early"]], plano)
            else:
                modelos = ajustar_valor(x["treino"], meta[["retorno_liquido_compra", "retorno_liquido_venda"]].to_numpy()[partes["treino"]])
                hist = []
            salvar_json(pasta/f"historico_{numero}_{familia}.json", hist)
            pa, ps = prever_scores(familia, modelos, x["ajuste"]), prever_scores(familia, modelos, x["selecao"])
            for evento in eventos:
                ma = permissoes_estrategia(ctx.iloc[mp["ajuste"].posicao_bruta], evento)
                ms = permissoes_estrategia(ctx.iloc[mp["selecao"].posicao_bruta], evento)
                pol, am = selecionar_politica_barreiras(mp["ajuste"], pa, ma, plano.minimo, plano.meta)
                m, _ = avaliar_barreiras(mp["selecao"], aplicar_politica(ps, pol, *ms), plano.minimo, plano.meta)
                reg = {"regra": numero, **asdict(regra), "familia": familia, "objetivo": objetivo, "evento": evento, **pol, **m}
                linhas.append(reg)
                if melhor is None or pontuar_barreiras(m) > pontuar_barreiras(melhor):
                    melhor = reg
                    estado = {"modelos": list(modelos), "janela": janela, "regra": regra,
                              "mp": mp, "politica": pol, "ajuste": am}
            pd.DataFrame(linhas).to_csv(pasta/"comparacao_validacao.csv", index=False)
            logging.info("Estratégia %d/%d %s %s h=%d alvo=%g stop=%gx; melhor compra=%.1f%% venda=%.1f%%", numero+1,
                len(regras), familia, regra.tipo, regra.horizonte, regra.alvo, regra.razao_stop,
                100*(melhor["taxa_lucro_compra"] or 0), 100*(melhor["taxa_lucro_venda"] or 0))
    salvar_json(pasta/"escolha_antes_da_confirmacao.json", melhor)
    status = {"candidatos_meta_70": sum(int(x["meta_70_lucro_ambas"]) for x in linhas),
              "candidatos_aprovados": sum(int(x["aprovado"]) for x in linhas),
              "meta_na_selecao": bool(melhor["meta_70_lucro_ambas"]), "aprovado_na_selecao": bool(melhor["aprovado"]),
              "confirmacao_avaliada": False, "meta_confirmada": False, "aprovado_na_confirmacao": False,
              "confirmacao_bloqueada_por_reuso": False}
    abrir = melhor["aprovado"] and confirmar
    if abrir and not reservar_confirmacao(csv, saida, plano, pasta, melhor):
        status["confirmacao_bloqueada_por_reuso"] = True
        abrir = False
    if abrir:
        teste = estado["mp"]["teste"].copy()
        x = _entradas(normalizada, teste.posicao_bruta.to_numpy(), estado["janela"])
        p = prever_scores(melhor["familia"], estado["modelos"], x)
        mask = permissoes_estrategia(ctx.iloc[teste.posicao_bruta], melhor["evento"])
        sinais = aplicar_politica(p, estado["politica"], *mask)
        m, usados = avaliar_barreiras(teste, sinais, plano.minimo, plano.meta)
        teste["sinal"] = sinais
        teste["operacao_executada"] = False
        teste.loc[usados, "operacao_executada"] = True
        teste["score_compra"], teste["score_venda"] = p[:,1], p[:,2]
        teste.to_csv(pasta/"operacoes_confirmacao.csv", index=False)
        salvar_json(pasta/"resultado_confirmacao.json", m)
        status.update({"confirmacao_avaliada": True, "meta_confirmada": bool(m["meta_70_lucro_ambas"]),
                       "aprovado_na_confirmacao": bool(m["aprovado"])})
    # Preserva também os candidatos reprovados, marcados como experimentais.
    salvar_binario(pasta/"modelo_experimental", estado["modelos"], scaler,
        {"familia": melhor["familia"], "janela": estado["janela"], "horizonte": melhor["horizonte"],
         "evento": "todos", "banda": 0., "objetivo": objetivo, "politica": estado["politica"], "meta_confirmada": status["meta_confirmada"]}, plano)
    salvar_json(pasta/"modelo_experimental/modelo_estrategia.json", {"regra": asdict(estado["regra"]),
        "evento": melhor["evento"], "selecao": melhor, **status})
    salvar_json(pasta/"status.json", status)
    return pasta, pd.DataFrame(linhas), status


def carregar_prever_estrategia(dados, pasta):
    from src.binary_events import carregar_prever_binario
    pasta = Path(pasta)
    info = json.loads((pasta/"modelo_estrategia.json").read_text(encoding="utf-8"))
    base = json.loads((pasta/"modelo_binario.json").read_text(encoding="utf-8"))
    p = carregar_prever_binario(dados, pasta)
    bruto, _ = limpar_candles(dados)
    ctx = criar_contexto(bruto, ConfigPrevisao())
    regra = SaidaBarreiras(**info["regra"])
    masks = permissoes_estrategia(ctx.iloc[-1:], info["evento"])
    scores = np.array([[0., p["score_compra"], p["score_venda"]]])
    sinal = int(aplicar_politica(scores, base["politica"], *masks)[0])
    alvo, stop = distancias_barreiras(ctx, regra, base["plano"]["custo"])
    return {**p, "tipo_modelo": "lucro_com_barreiras", "sinal": ["manter", "compra", "venda"][sinal],
        "classe_predita": sinal, "evento": info["evento"], "alvo_fracao": float(alvo[-1]), "stop_fracao": float(stop[-1]),
        "referencia_barreiras": "Abertura do próximo candle, conhecida apenas na execução.",
        "objetivo_treino": base.get("objetivo", "taxa"),
        "natureza_scores": "Ranking de retorno esperado, não probabilidade de lucro." if base.get("objetivo") == "valor" else "Escores de classificação de lucro, sem garantia de acerto.",
        "aprovado_na_confirmacao": info["aprovado_na_confirmacao"]}
