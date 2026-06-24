"""Configurações centralizadas do experimento do TCC.

Edite os valores desta classe ou use os argumentos de linha de comando de
``main.py``. Nenhuma configuração deste projeto envia ordens ao mercado.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


RAIZ_PROJETO = Path(__file__).resolve().parent


@dataclass
class ConfigProjeto:
    """Parâmetros de coleta, processamento, modelo e saída."""

    # Fonte dos dados — exemplo solicitado para PETR4 no M5.
    usar_metatrader: bool = True
    simbolo: str = "PETR4"
    timeframe: str = "M5"
    quantidade_candles: int = 10_000
    ignorar_candle_em_formacao: bool = True
    caminho_terminal_mt5: Optional[str] = None
    csv_entrada: Optional[Path] = None

    # Diretórios. Todos os artefatos ficam dentro de pasta_saida.
    pasta_saida: Path = RAIZ_PROJETO
    nome_pasta_resultados: str = "resultados"
    nome_pasta_dados_processados: str = "dados_processados"

    # Engenharia de atributos.
    periodo_sma: int = 20
    periodo_ema_feature: int = 20
    periodo_volatilidade: int = 20
    periodo_rsi: int = 14
    tipo_scaler: str = "standard"  # "standard" ou "minmax"

    # Cenários experimentais.
    intensidades_ruido: Tuple[float, ...] = (0.0005, 0.001, 0.002, 0.005)
    # Intensidade usada para comparar a série ruidosa com EMA/Wavelet. Os
    # filtros são aplicados SOBRE esta série, nunca diretamente sobre o close
    # original, pois o ruído injetado é a verdade de referência da análise.
    intensidade_ruido_filtros: float = 0.002
    ema_span_cenario: int = 20
    wavelet: str = "db4"
    wavelet_nivel: int = 3
    wavelet_janela_causal: int = 256
    wavelet_threshold_scale: float = 1.0
    wavelet_threshold_mode: str = "soft"

    # Busca de hiperparâmetros dos filtros. A escolha minimiza o RMSE entre a
    # série filtrada e a original somente no trecho de treino.
    executar_busca_filtros: bool = True
    ema_spans_busca: Tuple[int, ...] = (2, 3, 5, 8, 13, 20)
    wavelets_busca: Tuple[str, ...] = ("db2", "db4", "sym4")
    wavelet_niveis_busca: Tuple[int, ...] = (1, 2, 3)
    wavelet_janelas_busca: Tuple[int, ...] = (64, 128, 256)
    wavelet_threshold_scales_busca: Tuple[float, ...] = (0.5, 1.0)
    wavelet_threshold_modes_busca: Tuple[str, ...] = ("soft",)

    # Rótulos e sequências. O limiar de 0,8% foi definido no treino acima do
    # limite de 95% do ruído relativo medido (1,96 * sqrt(2) * sigma) e acima
    # do desvio-padrão dos retornos no horizonte de 10 candles.
    horizonte_futuro: int = 10
    limiar_retorno: float = 0.008
    tamanho_janela: int = 60
    proporcao_treino: float = 0.70
    proporcao_validacao: float = 0.15
    proporcao_teste: float = 0.15

    # Rede neural.
    tipo_modelo: str = "LSTM"  # também aceita "GRU"
    unidades_recorrentes: int = 64
    dropout: float = 0.30
    unidades_dense: int = 32
    taxa_aprendizado: float = 0.001
    epocas: int = 50
    batch_size: int = 64
    paciencia_early_stopping: int = 8
    usar_pesos_classes: bool = True
    seed: int = 42
    verbose_treinamento: int = 1

    # Features usadas pelo modelo. A ordem é preservada.
    features: Tuple[str, ...] = field(
        default_factory=lambda: (
            "open",
            "high",
            "low",
            "close_modelo",
            "tick_volume",
            "retorno_percentual",
            "sma",
            "ema",
            "volatilidade",
            "rsi",
        )
    )

    @property
    def pasta_resultados(self) -> Path:
        return Path(self.pasta_saida) / self.nome_pasta_resultados

    @property
    def pasta_dados_processados(self) -> Path:
        return Path(self.pasta_saida) / self.nome_pasta_dados_processados

    def validar(self) -> None:
        """Falha cedo quando uma configuração é inconsistente."""
        if self.quantidade_candles <= 0:
            raise ValueError("quantidade_candles deve ser positiva.")
        if self.horizonte_futuro <= 0 or self.tamanho_janela <= 1:
            raise ValueError("Horizonte e janela temporal devem ser positivos.")
        if self.limiar_retorno < 0:
            raise ValueError("limiar_retorno não pode ser negativo.")
        if self.tipo_scaler.lower() not in {"standard", "minmax"}:
            raise ValueError("tipo_scaler deve ser 'standard' ou 'minmax'.")
        if self.tipo_modelo.upper() not in {"LSTM", "GRU"}:
            raise ValueError("tipo_modelo deve ser 'LSTM' ou 'GRU'.")
        if self.wavelet_threshold_mode not in {"soft", "hard"}:
            raise ValueError("wavelet_threshold_mode deve ser 'soft' ou 'hard'.")
        if self.wavelet_janela_causal <= 1:
            raise ValueError("wavelet_janela_causal deve ser maior que 1.")
        if self.intensidade_ruido_filtros not in self.intensidades_ruido:
            raise ValueError(
                "intensidade_ruido_filtros deve pertencer a intensidades_ruido."
            )
        if not self.ema_spans_busca or min(self.ema_spans_busca) <= 0:
            raise ValueError("ema_spans_busca deve conter apenas valores positivos.")
        if not self.wavelets_busca or not self.wavelet_niveis_busca:
            raise ValueError("A grade de busca Wavelet não pode ser vazia.")
        if min(self.wavelet_niveis_busca) <= 0:
            raise ValueError("Os níveis Wavelet devem ser positivos.")
        if min(self.wavelet_janelas_busca) <= 1:
            raise ValueError("As janelas Wavelet devem ser maiores que 1.")
        if min(self.wavelet_threshold_scales_busca) < 0:
            raise ValueError("As escalas de limiar Wavelet não podem ser negativas.")
        if not set(self.wavelet_threshold_modes_busca) <= {"soft", "hard"}:
            raise ValueError("Modos de limiar da busca devem ser 'soft' ou 'hard'.")
        soma = self.proporcao_treino + self.proporcao_validacao + self.proporcao_teste
        if abs(soma - 1.0) > 1e-9 or min(
            self.proporcao_treino,
            self.proporcao_validacao,
            self.proporcao_teste,
        ) <= 0:
            raise ValueError("As proporções devem ser positivas e somar 1.")
