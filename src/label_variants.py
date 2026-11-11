"""Rótulos auxiliares alternativos; o resultado operacional continua separado."""

import numpy as np
import pandas as pd


ROTULAGENS = ("lucro_liquido", "direcao_bruta", "volatilidade_05", "volatilidade_10",
              "trajetoria_media", "barreiras_002", "barreiras_volatilidade")


def rotulos_alternativos(dados, config, nome, custo_por_lado=.0002):
    if nome not in ROTULAGENS:
        raise ValueError(f"Rotulagem desconhecida: {nome}")
    if nome == "lucro_liquido":
        return dados.rotulo.copy()
    entrada = dados.open.shift(-1)
    retorno = dados.close.shift(-config.horizonte) / entrada - 1
    minimo = (1 + custo_por_lado) / (1 - custo_por_lado) - 1
    vol = dados.close.pct_change(fill_method=None).rolling(20).std() * np.sqrt(config.horizonte)
    if nome == "direcao_bruta":
        y = np.select([retorno > 1e-12, retorno < -1e-12], [1., 2.], default=0.)
    elif nome.startswith("volatilidade_"):
        fator = .5 if nome == "volatilidade_05" else 1.
        barreira = (fator * vol).clip(lower=minimo)
        y = np.select([(retorno > 1e-12) & (retorno >= barreira),
                       (retorno < -1e-12) & (retorno <= -barreira)], [1., 2.], default=0.)
        y[barreira.isna().to_numpy()] = np.nan
    elif nome == "trajetoria_media":
        media_futura = sum(dados.close.shift(-k) for k in range(1, config.horizonte+1)) / config.horizonte
        razao = media_futura / entrada
        long = razao - 1 - custo_por_lado * (1 + razao)
        short = 1 - razao - custo_por_lado * (1 + razao)
        y = np.select([long > 1e-12, short > 1e-12], [1., 2.], default=0.)
    else:
        barreira = pd.Series(.002, index=dados.index) if nome == "barreiras_002" else vol.clip(lower=max(.001, minimo))
        y = np.zeros(len(dados), dtype=float)
        for i in np.flatnonzero(dados.alvo_elegivel.to_numpy()):
            if not np.isfinite(barreira.iloc[i]):
                y[i] = np.nan
                continue
            preco = entrada.iloc[i]
            superior, inferior = preco * (1+barreira.iloc[i]), preco * (1-barreira.iloc[i])
            for j in range(i+1, i+config.horizonte+1):
                # O open antecede os extremos dentro do candle e resolve gaps.
                if dados.open.iloc[j] >= superior:
                    y[i] = 1
                    break
                if dados.open.iloc[j] <= inferior:
                    y[i] = 2
                    break
                acima, abaixo = dados.high.iloc[j] >= superior, dados.low.iloc[j] <= inferior
                if acima and abaixo:
                    y[i] = np.nan  # OHLC não informa a ordem intrabar.
                    break
                if acima or abaixo:
                    y[i] = 1 if acima else 2
                    break
    y = pd.Series(y, index=dados.index, name="rotulo_auxiliar")
    y.loc[~dados.alvo_elegivel] = np.nan
    return y
