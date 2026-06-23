"""Construção e treinamento dos modelos LSTM/GRU."""

from __future__ import annotations

import random

import numpy as np


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow não está instalado. Instale as dependências de requirements.txt."
        ) from exc
    return tf


def configurar_reprodutibilidade(seed: int) -> None:
    """Fixa as sementes disponíveis para reduzir variações entre execuções."""
    random.seed(seed)
    np.random.seed(seed)
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(seed)


def criar_modelo(forma_entrada, config):
    """Cria a arquitetura mínima solicitada, alternável entre LSTM e GRU."""
    tf = _tensorflow()
    camada_recorrente = (
        tf.keras.layers.LSTM
        if config.tipo_modelo.upper() == "LSTM"
        else tf.keras.layers.GRU
    )
    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=forma_entrada),
            camada_recorrente(config.unidades_recorrentes),
            tf.keras.layers.Dropout(config.dropout),
            tf.keras.layers.Dense(config.unidades_dense, activation="relu"),
            tf.keras.layers.Dropout(config.dropout),
            tf.keras.layers.Dense(3, activation="softmax"),
        ],
        name=f"classificador_{config.tipo_modelo.lower()}",
    )
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.taxa_aprendizado),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def treinar_modelo(modelo, x_treino, y_treino, x_validacao, y_validacao, config):
    """Treina sem embaralhar e restaura os melhores pesos de validação."""
    tf = _tensorflow()
    y_treino_cat = tf.keras.utils.to_categorical(y_treino, num_classes=3)
    y_validacao_cat = tf.keras.utils.to_categorical(y_validacao, num_classes=3)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.paciencia_early_stopping,
        restore_best_weights=True,
    )
    return modelo.fit(
        x_treino,
        y_treino_cat,
        validation_data=(x_validacao, y_validacao_cat),
        epochs=config.epocas,
        batch_size=config.batch_size,
        callbacks=[early_stopping],
        shuffle=False,
        verbose=config.verbose_treinamento,
    )


def limpar_sessao() -> None:
    """Libera o grafo anterior antes de criar o modelo do próximo cenário."""
    _tensorflow().keras.backend.clear_session()

