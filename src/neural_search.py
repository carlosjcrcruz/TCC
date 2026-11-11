"""Busca de redes em desenvolvimento temporal, sem selecionar pelo teste final."""

from dataclasses import asdict, dataclass
import logging

import numpy as np
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from src.forecast_evaluation import pontuacao_validacao
from src.forecast_models import ModeloPrevisao, tensorflow


@dataclass(frozen=True)
class EspecificacaoRede:
    nome: str
    arquitetura: str
    janela: int
    unidades: int = 32
    dropout: float = .2
    taxa: float = .0003
    potencia_peso: float = .5
    focal: bool = False
    ajustar_decisao: bool = False
    criterio: str = "f1_macro"


def candidatos():
    return [
        EspecificacaoRede("lstm12_sem_pesos", "lstm", 12, 16, .1, .001, 0),
        EspecificacaoRede("lstm24_sem_pesos", "lstm", 24, potencia_peso=0),
        EspecificacaoRede("lstm24_pesos_suaves", "lstm", 24),
        EspecificacaoRede("lstm60_balanceada", "lstm", 60, potencia_peso=1),
        EspecificacaoRede("gru24_pesos_suaves", "gru", 24),
        EspecificacaoRede("gru12_balanceada", "gru", 12, 64, .2, .001, 1),
        EspecificacaoRede("cnn24_pesos_suaves", "cnn", 24, taxa=.001),
        EspecificacaoRede("cnn60_balanceada", "cnn", 60, potencia_peso=1),
        EspecificacaoRede("cnn_gru24", "cnn_gru", 24),
        EspecificacaoRede("lstm24_focal", "lstm", 24, focal=True),
        EspecificacaoRede("gru24_decisao", "gru", 24, potencia_peso=1, ajustar_decisao=True),
        EspecificacaoRede("cnn_gru24_decisao", "cnn_gru", 24, potencia_peso=1, ajustar_decisao=True),
        EspecificacaoRede("lstm24_decisao", "lstm", 24, potencia_peso=1, ajustar_decisao=True),
        EspecificacaoRede("mlp12_decisao", "mlp", 12, potencia_peso=1, ajustar_decisao=True),
        EspecificacaoRede("mlp24_suave", "mlp", 24, ajustar_decisao=True),
        EspecificacaoRede("cnn_dilatada24", "cnn_dilatada", 24, potencia_peso=1, ajustar_decisao=True),
        EspecificacaoRede("gru24_diagonal", "gru", 24, potencia_peso=1, ajustar_decisao=True, criterio="diagonal"),
        EspecificacaoRede("cnn_gru24_diagonal", "cnn_gru", 24, potencia_peso=1, ajustar_decisao=True, criterio="diagonal"),
        EspecificacaoRede("lstm24_diagonal", "lstm", 24, potencia_peso=1, ajustar_decisao=True, criterio="diagonal"),
    ]


def treinar_candidato(especificacao, xt, yt, xv, yv, config):
    tf = tensorflow()
    modelo = ModeloPrevisao(especificacao.nome, config, sequencial=True)
    classes = np.unique(yt)
    w = compute_class_weight("balanced", classes=classes, y=yt) ** especificacao.potencia_peso
    pesos = {int(c): float(p) for c, p in zip(classes, w)} if especificacao.potencia_peso else None
    opcoes = tf.data.Options()
    opcoes.threading.private_threadpool_size = 1
    opcoes.experimental_deterministic = True

    def score_validacao(real, previsto):
        if especificacao.criterio == "diagonal":
            cm = confusion_matrix(real, previsto, labels=[0, 1, 2])
            rec = np.divide(cm.diagonal(), cm.sum(axis=1), out=np.zeros(3), where=cm.sum(axis=1) > 0)
            return float(.5 * rec.min() + .5 * rec.mean())
        return pontuacao_validacao(real, previsto)

    class F1Validacao(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            p = self.model(xv, training=False).numpy()
            logs["val_score_selecao"] = score_validacao(yv, p.argmax(axis=1))

    def focal_esparsa(y_real, y_prob):
        y_real = tf.cast(tf.reshape(y_real, [-1]), tf.int32)
        pt = tf.reduce_sum(tf.one_hot(y_real, 3) * y_prob, axis=1)
        pt = tf.clip_by_value(pt, 1e-7, 1 - 1e-7)
        return -tf.square(1 - pt) * tf.math.log(pt)

    for seed in config.sementes:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        camadas = [tf.keras.layers.Input(shape=xt.shape[1:])]
        if especificacao.arquitetura.startswith("cnn"):
            camadas += [tf.keras.layers.Conv1D(especificacao.unidades, 5, padding="causal", activation="relu"),
                        tf.keras.layers.Conv1D(especificacao.unidades, 3, dilation_rate=2 if especificacao.arquitetura == "cnn_dilatada" else 1, padding="causal", activation="relu")]
        if especificacao.arquitetura in {"cnn", "cnn_dilatada"}:
            camadas.append(tf.keras.layers.GlobalAveragePooling1D())
        elif especificacao.arquitetura == "mlp":
            camadas += [tf.keras.layers.Flatten(), tf.keras.layers.Dense(especificacao.unidades, activation="relu")]
        else:
            recorrente = tf.keras.layers.LSTM if especificacao.arquitetura == "lstm" else tf.keras.layers.GRU
            camadas.append(recorrente(especificacao.unidades))
        camadas += [tf.keras.layers.Dropout(especificacao.dropout), tf.keras.layers.Dense(16, activation="relu"),
                    tf.keras.layers.Dense(3, activation="softmax")]
        rede = tf.keras.Sequential(camadas)
        rede.compile(optimizer=tf.keras.optimizers.Adam(especificacao.taxa, clipnorm=1),
                     loss=focal_esparsa if especificacao.focal else "sparse_categorical_crossentropy")
        # As partições são cronológicas. Apenas minibatches do treino são misturados;
        # as redes não são stateful e cada janela continua temporalmente ordenada.
        treino = tf.data.Dataset.from_tensor_slices((xt, yt)).shuffle(len(yt), seed=seed).batch(config.batch_size).with_options(opcoes)
        validacao = tf.data.Dataset.from_tensor_slices((xv, yv)).batch(config.batch_size).with_options(opcoes)
        cb = [] if especificacao.criterio == "log_loss" else [F1Validacao()]
        cb.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if especificacao.criterio == "log_loss" else "val_score_selecao",
            mode="min" if especificacao.criterio == "log_loss" else "max",
            patience=config.paciencia, restore_best_weights=True, min_delta=.0001))
        logging.info("Ajuste %s / semente %d", especificacao.nome, seed)
        h = rede.fit(treino, validation_data=validacao, epochs=config.epocas, class_weight=pesos, callbacks=cb, verbose=0)
        modelo.estimadores.append(rede)
        modelo.historicos.append({"seed": seed, "history": h.history})
    modelo.parametros = {"especificacao": asdict(especificacao), "pesos_classes": pesos,
                         "shuffle_apenas_treino": True, "monitor": especificacao.criterio}
    if especificacao.ajustar_decisao:
        p = modelo.prever(xv, np.zeros(len(yv)))["probabilidades"]
        melhor = (score_validacao(yv, p.argmax(axis=1)), [1., 1., 1.])
        for compra in [.25, .5, 1., 2., 4., 8.]:
            for venda in [.25, .5, 1., 2., 4., 8.]:
                fatores = [1., compra, venda]
                score = score_validacao(yv, (p * fatores).argmax(axis=1))
                if score > melhor[0]:
                    melhor = score, fatores
        modelo.ajuste_decisao = melhor[1]
        modelo.parametros["ajuste_decisao"] = {"multiplicadores": melhor[1], "score_validacao": melhor[0],
            "observacao": "Ajuste de decisão supervisionado na validação; não é calibração de probabilidades."}
    return modelo
