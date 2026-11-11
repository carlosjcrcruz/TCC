"""Configuração independente do experimento de previsão sem ruído."""

from dataclasses import dataclass


@dataclass
class ConfigPrevisao:
    # Evento preservado do TCC, sem depender da intensidade de ruído artificial.
    horizonte: int = 10
    minutos_candle: int = 5
    limiar: float = 0.008
    janela: int = 60
    timezone: str = "America/Sao_Paulo"
    # As entradas podem conter pregões anteriores; o alvo exige continuidade.
    folds: int = 3
    fracao_teste_final: float = 0.15
    fracao_bloco_desenvolvimento: float = 0.15
    modelos: tuple = (
        "majoritaria", "momentum", "logistica", "hist_gradient",
        "ridge_retorno", "lstm", "lstm_retorno",
    )
    sementes: tuple = (42, 123, 2026)
    epocas: int = 25
    paciencia: int = 5
    batch_size: int = 128
    unidades: int = 32
    dropout: float = 0.2
    taxa_aprendizado: float = 0.001
    meta: float = 0.70
    minimo_sinais: int = 20
    bootstrap_sessoes: int = 300
    custos_por_lado: tuple = (0.0, 0.0002, 0.0005)

    def validar(self):
        if self.horizonte < 1 or self.minutos_candle < 1 or self.janela < 2:
            raise ValueError("Horizonte, intervalo e janela inválidos.")
        if self.limiar <= 0 or self.folds < 1:
            raise ValueError("Limiar e número de folds devem ser positivos.")
        if not 0 < self.fracao_teste_final < 0.4:
            raise ValueError("A fração de teste final deve estar entre zero e 0,4.")
        if not 0 < self.fracao_bloco_desenvolvimento < 0.3:
            raise ValueError("Fração de bloco de desenvolvimento inválida.")
        permitidos = {"majoritaria", "momentum", "logistica", "hist_gradient", "ridge_retorno", "lstm", "lstm_retorno"}
        if not self.modelos or not set(self.modelos) <= permitidos:
            raise ValueError("Lista de modelos inválida.")
        if "majoritaria" not in self.modelos:
            raise ValueError("A referência majoritária é obrigatória.")
        if len(set(self.modelos)) != len(self.modelos):
            raise ValueError("Modelos duplicados.")
        if not self.sementes or len(set(self.sementes)) != len(self.sementes):
            raise ValueError("Informe sementes distintas.")
        if self.epocas < 1 or self.paciencia < 1 or self.batch_size < 1 or self.unidades < 1:
            raise ValueError("Parâmetros de treinamento inválidos.")
        if not 0 <= self.dropout < 1 or self.taxa_aprendizado <= 0:
            raise ValueError("Dropout ou taxa de aprendizado inválidos.")
        if not 0 < self.meta <= 1 or self.minimo_sinais < 1 or self.bootstrap_sessoes < 0:
            raise ValueError("Critério de avaliação inválido.")
        if not self.custos_por_lado or min(self.custos_por_lado) < 0:
            raise ValueError("Custos hipotéticos por lado devem ser não negativos.")
