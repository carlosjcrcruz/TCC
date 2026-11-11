"""Rótulos binários por direção, movimentos claros e eventos causais."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from src.forecast_data import FEATURES, criar_atributos, limpar_candles
from src.forecast_models import tensorflow
from src.trade_policy import aplicar_politica


FEATURES_EVENTOS = tuple(FEATURES) + (
    "posicao_close", "gap_abertura", "retorno_lag1", "retorno_lag2", "retorno_lag3",
    "vol_anterior", "razao_vol5_20", "distancia_ema5", "retorno_sessao",
    "desvio_media_ticks", "candles_sessao", "distancia_maxima20", "distancia_minima20",
)
EVENTOS = ("todos", "choque_reversao", "rsi_extremo", "rompimento")


@dataclass
class PlanoBinario:
    fim_treino: str = "2023-03-01T00:00:00Z"
    fim_early: str = "2023-05-01T00:00:00Z"
    fim_ajuste: str = "2023-09-01T00:00:00Z"
    fim_selecao: str = "2024-01-01T00:00:00Z"
    fim_teste: str = "2024-07-01T00:00:00Z"
    horizontes: tuple = (1, 3, 5, 10, 20, 40, 60)
    bandas: tuple = (0., .5, 1.)
    familias: tuple = ("hist_binario", "mlp_binario")
    eventos: tuple = EVENTOS
    custo: float = .0002
    meta: float = .70
    minimo: int = 20
    epocas: int = 30
    paciencia: int = 5
    sementes_finais: tuple = (42, 123, 2026)

    def validar(self):
        datas = [pd.Timestamp(getattr(self, k)) for k in ["fim_treino", "fim_early", "fim_ajuste", "fim_selecao", "fim_teste"]]
        if not all(a < b for a, b in zip(datas, datas[1:])) or any(d.tzinfo is None for d in datas):
            raise ValueError("Fronteiras devem ser crescentes e incluir fuso horário.")
        if not self.horizontes or min(self.horizontes) < 1 or not self.bandas or min(self.bandas) < 0:
            raise ValueError("Horizontes ou bandas inválidos.")
        if not self.eventos or not self.familias or not set(self.eventos) <= set(EVENTOS) or not set(self.familias) <= {"hist_binario", "mlp_binario"}:
            raise ValueError("Eventos ou famílias inválidos.")
        if not 0 <= self.custo < 1 or not 0 < self.meta <= 1 or self.minimo < 1 or self.epocas < 1:
            raise ValueError("Parâmetros econômicos ou de treinamento inválidos.")
        if not self.sementes_finais:
            raise ValueError("Informe sementes para a confirmação do modelo final.")


def criar_contexto(bruto, config):
    d = criar_atributos(bruto, config)
    amplitude = (d.high - d.low).replace(0, np.nan)
    d["posicao_close"] = ((d.close - d.low) / amplitude).fillna(.5)
    d["gap_abertura"] = d.open / d.close.shift(1) - 1
    for lag in [1, 2, 3]:
        d[f"retorno_lag{lag}"] = d.retorno_1.shift(lag)
    d["vol_anterior"] = d.retorno_1.rolling(20).std().shift(1)
    d["razao_vol5_20"] = d.retorno_1.rolling(5).std() / (d.retorno_1.rolling(20).std() + 1e-8)
    d["distancia_ema5"] = d.close / d.close.ewm(span=5, adjust=False).mean() - 1
    abertura_dia = d.groupby("sessao", sort=False).open.transform("first")
    d["retorno_sessao"] = d.close / abertura_dia - 1
    pesos = d.tick_volume + 1
    media_ticks = (d.close * pesos).groupby(d.sessao).cumsum() / pesos.groupby(d.sessao).cumsum()
    d["desvio_media_ticks"] = d.close / media_ticks - 1
    d["candles_sessao"] = d.groupby("sessao", sort=False).cumcount().astype(float)
    max_anterior = d.high.rolling(20).max().shift(1)
    min_anterior = d.low.rolling(20).min().shift(1)
    d["distancia_maxima20"] = d.close / max_anterior - 1
    d["distancia_minima20"] = d.close / min_anterior - 1
    return d.replace([np.inf, -np.inf], np.nan)


def permissoes(dados, nome):
    if nome == "todos":
        return np.ones(len(dados), dtype=bool), np.ones(len(dados), dtype=bool)
    if nome == "choque_reversao":
        limite = 1.5 * dados.vol_anterior
        return (dados.retorno_1 < -limite).to_numpy(), (dados.retorno_1 > limite).to_numpy()
    if nome == "rsi_extremo":
        return (dados.rsi14 <= .30).to_numpy(), (dados.rsi14 >= .70).to_numpy()
    if nome == "rompimento":
        return (dados.distancia_maxima20 > 0).to_numpy(), (dados.distancia_minima20 < 0).to_numpy()
    raise ValueError("Evento desconhecido.")


def rotulos_binarios(meta, volatilidade_passada, horizonte, banda):
    r = meta[["retorno_liquido_compra", "retorno_liquido_venda"]].to_numpy(dtype=float)
    if banda == 0:
        return np.where(np.isfinite(r), (r > 1e-12).astype(float), np.nan).astype(np.float32)
    b = banda * np.asarray(volatilidade_passada)[:, None] * np.sqrt(horizonte)
    return np.where(r > b, 1., np.where(r < -b, 0., np.nan)).astype(np.float32)


def mascarar_eventos(p, politica, permitidas):
    return aplicar_politica(p, politica, *permitidas)


def perda_binaria_mascarada(y_true, y_pred):
    tf = tensorflow()
    conhecido = tf.math.is_finite(y_true)
    y = tf.where(conhecido, y_true, tf.zeros_like(y_true))
    p = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
    erro = -(y * tf.math.log(p) + (1-y) * tf.math.log(1-p))
    mask = tf.cast(conhecido, erro.dtype)
    return tf.reduce_sum(erro * mask, axis=-1) / tf.maximum(tf.reduce_sum(mask, axis=-1), 1.)


def ajustar_modelo(familia, xt, yt, xe, ye, plano, sementes=(42,)):
    if familia == "hist_binario":
        modelos = []
        for lado in range(2):
            valido = np.isfinite(yt[:, lado])
            y = yt[valido, lado].astype(int)
            if not len(y):
                raise ValueError("Nenhum rótulo conhecido para uma direção.")
            modelo = (DummyClassifier(strategy="prior") if len(np.unique(y)) == 1 else
                HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=15, min_samples_leaf=50,
                    l2_regularization=5., early_stopping=False, random_state=42))
            modelo.fit(xt[valido], y)
            modelos.append(modelo)
        return modelos, []
    tf = tensorflow()
    modelos, historicos = [], []
    valido = np.isfinite(yt).any(axis=1)
    valido_e = np.isfinite(ye).any(axis=1)
    if not valido.any() or not valido_e.any():
        raise ValueError("Rótulos insuficientes para treino/validação.")
    opcoes = tf.data.Options()
    opcoes.threading.private_threadpool_size = 1
    for seed in sementes:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        modelo = tf.keras.Sequential([
            tf.keras.layers.Input(shape=xt.shape[1:]), tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"), tf.keras.layers.Dropout(.25),
            tf.keras.layers.Dense(32, activation="relu"), tf.keras.layers.Dense(2, activation="sigmoid")])
        modelo.compile(optimizer=tf.keras.optimizers.Adam(.0003, clipnorm=1.), loss=perda_binaria_mascarada)
        treino = tf.data.Dataset.from_tensor_slices((xt[valido], yt[valido])).shuffle(int(valido.sum()), seed=seed).batch(128).with_options(opcoes)
        early = tf.data.Dataset.from_tensor_slices((xe[valido_e], ye[valido_e])).batch(128).with_options(opcoes)
        cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=plano.paciencia, restore_best_weights=True)
        h = modelo.fit(treino, validation_data=early, epochs=plano.epocas, callbacks=[cb], verbose=0)
        modelos.append(modelo)
        historicos.append({"seed": seed, "history": h.history})
    return modelos, historicos


def prever_scores(familia, modelos, x):
    if familia == "hist_binario":
        pp = []
        for m in modelos:
            p = m.predict_proba(x)
            col = np.flatnonzero(np.asarray(m.classes_) == 1)
            pp.append(p[:, col[0]] if len(col) else np.zeros(len(x)))
        scores = np.column_stack(pp)
    else:
        scores = np.mean([m(x, training=False).numpy() for m in modelos], axis=0)
    # São dois escores binários independentes, não uma distribuição multiclasse.
    return np.column_stack([np.zeros(len(x)), scores])


def salvar_binario(pasta, modelos, scaler, info, plano):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, pasta / "scaler.joblib")
    arquivos = []
    for i, modelo in enumerate(modelos):
        nome = f"estimador_{i}.joblib" if info["familia"] == "hist_binario" else f"rede_{i}.keras"
        if nome.endswith("joblib"):
            joblib.dump(modelo, pasta / nome)
        else:
            modelo.save(pasta / nome)
        arquivos.append(nome)
    meta = {"tipo_modelo": "binario_eventos", "plano": asdict(plano), "features": list(FEATURES_EVENTOS),
            "arquivos": arquivos, **info}
    (pasta / "modelo_binario.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def carregar_prever_binario(dados, pasta):
    pasta = Path(pasta)
    info = json.loads((pasta / "modelo_binario.json").read_text(encoding="utf-8"))
    if info["features"] != list(FEATURES_EVENTOS):
        raise ValueError("A representação dos eventos mudou.")
    from src.forecast_config import ConfigPrevisao
    c = ConfigPrevisao(horizonte=info["horizonte"])
    bruto, _ = limpar_candles(dados)
    contexto = criar_contexto(bruto, c)
    janela = info["janela"]
    valores = contexto[list(FEATURES_EVENTOS)].iloc[-janela:].to_numpy(dtype=np.float32)
    if len(valores) != janela or not np.isfinite(valores).all():
        raise ValueError("Histórico insuficiente para os atributos do modelo.")
    scaler = joblib.load(pasta / "scaler.joblib")
    x = scaler.transform(valores).astype(np.float32)
    if info["familia"] == "mlp_binario":
        x = x[None, :, :]
    else:
        x = x[-1:]
    modelos = [joblib.load(pasta / f) if f.endswith("joblib") else tensorflow().keras.models.load_model(pasta / f, compile=False) for f in info["arquivos"]]
    scores = prever_scores(info["familia"], modelos, x)
    masks = permissoes(contexto.iloc[-1:], info["evento"])
    sinal = int(mascarar_eventos(scores, info["politica"], masks)[0])
    return {"tipo_modelo": "binario_eventos", "sinal": ["manter", "compra", "venda"][sinal],
            "classe_predita": sinal, "horizonte_candles": info["horizonte"],
            "horizonte_minutos": info["horizonte"] * c.minutos_candle,
            "score_compra": float(scores[0, 1]), "score_venda": float(scores[0, 2]),
            "evento": info["evento"], "custo_por_lado": info["plano"]["custo"],
            "instante_emissao": str(contexto.time.iloc[-1] + pd.Timedelta(minutes=c.minutos_candle)),
            "meta_confirmada": info.get("meta_confirmada", False),
            "status": "Previsão experimental; escores não garantem lucro. Horizonte condicionado à continuidade do pregão."}
