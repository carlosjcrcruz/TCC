"""Referências simples, modelos tabulares e redes para a mesma tarefa futura."""

from dataclasses import dataclass, field
import logging
import os

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.utils.class_weight import compute_class_weight

from src.forecast_data import classes_do_retorno
from src.forecast_evaluation import pontuacao_validacao

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")


def tensorflow():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    if not getattr(tensorflow, "configurado", False):
        try:
            tf.config.threading.set_intra_op_parallelism_threads(4)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            # O pipeline legado ou outro consumidor pode ter inicializado o runtime.
            logging.info("TensorFlow já inicializado; mantendo sua configuração de threads.")
        tf.config.experimental.enable_op_determinism()
        tensorflow.configurado = True
    return tf


@dataclass
class ModeloPrevisao:
    nome: str
    config: object
    estimadores: list = field(default_factory=list)
    parametros: dict = field(default_factory=dict)
    historicos: list = field(default_factory=list)
    regressao: bool = False
    sequencial: bool = False
    media_alvo: float = 0.0
    escala_alvo: float = 1.0
    ajuste_decisao: object = None
    politica_sinais: object = None
    tipo_alvo: str = "retorno_limiar"
    custo_por_lado: object = None
    rotulagem_treinamento: object = None

    def prever(self, x, retorno_passado):
        individuais = []
        if self.nome == "momentum":
            r = np.asarray(retorno_passado)
            return {"classes": classes_do_retorno(r, self.config.limiar), "retornos": r, "probabilidades": None, "individuais": []}
        for modelo in self.estimadores:
            if self.sequencial:
                valor = modelo(x, training=False).numpy()
            elif self.regressao:
                valor = modelo.predict(x)
            else:
                parcial = modelo.predict_proba(x)
                valor = np.zeros((len(x), 3))
                valor[:, np.asarray(modelo.classes_, dtype=int)] = parcial
            if self.regressao:
                valor = np.asarray(valor).reshape(-1) * self.escala_alvo + self.media_alvo
            else:
                valor = np.asarray(valor, dtype=np.float64)
                valor /= valor.sum(axis=1, keepdims=True)
            individuais.append(valor)
        media = np.mean(individuais, axis=0)
        if self.regressao:
            return {"classes": classes_do_retorno(media, self.config.limiar), "retornos": media, "probabilidades": None, "individuais": individuais}
        media /= media.sum(axis=1, keepdims=True)
        if self.ajuste_decisao is not None:
            media *= np.asarray(self.ajuste_decisao)
            media /= media.sum(axis=1, keepdims=True)
        classes = np.argmax(media, axis=1)
        if self.politica_sinais is not None:
            from src.trade_policy import aplicar_politica
            classes = aplicar_politica(media, self.politica_sinais)
        return {"classes": classes, "retornos": None, "probabilidades": media, "individuais": individuais}


def treinar(nome, x_treino, y_treino, r_treino, x_val, y_val, r_val, config):
    modelo = ModeloPrevisao(nome, config)
    if nome == "momentum":
        modelo.regressao = True
        modelo.parametros = {"regra": "retorno passado de h candles como estimativa do retorno futuro"}
        return modelo
    if nome == "majoritaria" or (nome in {"logistica", "hist_gradient"} and len(np.unique(y_treino)) < 2):
        estimador = DummyClassifier(strategy="prior").fit(x_treino, y_treino)
        modelo.estimadores = [estimador]
        modelo.parametros = {"strategy": "prior", "fallback_classe_unica": nome != "majoritaria"}
        return modelo
    if nome in {"lstm", "lstm_retorno"}:
        return _treinar_redes(nome, x_treino, y_treino, r_treino, x_val, y_val, r_val, config)

    candidatos = []
    if nome == "logistica":
        candidatos = [(LogisticRegression(C=c, class_weight=peso, max_iter=1500), {"C": c, "class_weight": peso})
                      for c in [0.1, 1.0] for peso in [None, "balanced"]]
    elif nome == "hist_gradient":
        candidatos = [(HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, l2_regularization=1.0,
                        early_stopping=False, class_weight=peso, random_state=config.sementes[0]), {"class_weight": peso, "max_iter": 100, "max_leaf_nodes": 7})
                      for peso in [None, "balanced"]]
    elif nome == "ridge_retorno":
        modelo.regressao = True
        candidatos = [(Ridge(alpha=a), {"alpha": a}) for a in [1.0, 10.0, 100.0]]
    else:
        raise ValueError(f"Modelo desconhecido: {nome}")

    melhor = None
    for estimador, parametros in candidatos:
        estimador.fit(x_treino, r_treino if modelo.regressao else y_treino)
        if modelo.regressao:
            retorno = estimador.predict(x_val)
            # Regressão é selecionada pelo erro de retorno na validação.
            score = -float(np.mean((r_val - retorno) ** 2))
        else:
            score = pontuacao_validacao(y_val, estimador.predict(x_val))
        if melhor is None or score > melhor[0]:
            melhor = score, estimador, parametros
    modelo.estimadores = [melhor[1]]
    modelo.parametros = {**melhor[2], "score_validacao": melhor[0], "criterio_validacao": "menos_mse" if modelo.regressao else "f1_macro"}
    return modelo


def _treinar_redes(nome, xt, yt, rt, xv, yv, rv, config):
    tf = tensorflow()
    modelo = ModeloPrevisao(nome, config, regressao=nome.endswith("retorno"), sequencial=True)
    if modelo.regressao:
        modelo.media_alvo = float(np.mean(rt))
        modelo.escala_alvo = max(float(np.std(rt)), 1e-6)
        alvo_t = ((rt - modelo.media_alvo) / modelo.escala_alvo).astype(np.float32)
        alvo_v = ((rv - modelo.media_alvo) / modelo.escala_alvo).astype(np.float32)
        pesos = None
    else:
        alvo_t, alvo_v = yt, yv
        classes = np.unique(yt)
        valores = compute_class_weight("balanced", classes=classes, y=yt)
        pesos = {int(c): float(p) for c, p in zip(classes, valores)}

    # Sem embaralhamento, inclusive na entrada do TensorFlow.
    opcoes = tf.data.Options()
    opcoes.threading.private_threadpool_size = 1
    opcoes.experimental_deterministic = True
    treino = tf.data.Dataset.from_tensor_slices((xt, alvo_t)).batch(config.batch_size).with_options(opcoes)
    validacao = tf.data.Dataset.from_tensor_slices((xv, alvo_v)).batch(config.batch_size).with_options(opcoes)

    class F1Validacao(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            prob = self.model(xv, training=False).numpy()
            logs["val_f1_macro"] = pontuacao_validacao(yv, np.argmax(prob, axis=1))

    for seed in config.sementes:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        rede = tf.keras.Sequential([
            tf.keras.layers.Input(shape=xt.shape[1:]),
            tf.keras.layers.LSTM(config.unidades),
            tf.keras.layers.Dropout(config.dropout),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1 if modelo.regressao else 3, activation="linear" if modelo.regressao else "softmax"),
        ])
        rede.compile(optimizer=tf.keras.optimizers.Adam(config.taxa_aprendizado, clipnorm=1.0),
                     loss="mse" if modelo.regressao else "sparse_categorical_crossentropy")
        callbacks = [] if modelo.regressao else [F1Validacao()]
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if modelo.regressao else "val_f1_macro",
            mode="min" if modelo.regressao else "max", patience=config.paciencia, restore_best_weights=True,
        ))
        logging.info("Treinando %s, semente %d, até %d épocas...", nome, seed, config.epocas)
        historico = rede.fit(treino, validation_data=validacao, epochs=config.epocas,
                             callbacks=callbacks, class_weight=pesos, verbose=0)
        modelo.estimadores.append(rede)
        modelo.historicos.append({"seed": seed, "history": historico.history})
        logging.info("%s semente %d concluída em %d épocas.", nome, seed, len(historico.history["loss"]))
    modelo.parametros = {"sementes": list(config.sementes), "unidades": config.unidades,
                         "agregacao": "media_das_previsoes_sem_escolher_a_melhor_semente",
                         "pesos_classes": pesos, "monitor": "val_loss" if modelo.regressao else "val_f1_macro"}
    return modelo
