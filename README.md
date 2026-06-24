# Ruído e falsos sinais em séries financeiras

Projeto experimental em Python para investigar como ruído, suavização por EMA e
Wavelet Denoising afetam falsos positivos em sinais de compra e venda. O código
usa somente candles históricos e **não envia ordens ao MetaTrader 5**.

## Metodologia resumida

O pipeline coleta ou lê candles, cria indicadores, define classes futuras,
normaliza os atributos e treina um LSTM ou GRU independente para cada cenário:

1. Original;
2. Ruído gaussiano proporcional de 0.0005;
3. Ruído gaussiano proporcional de 0.001;
4. Ruído gaussiano proporcional de 0.002;
5. Ruído gaussiano proporcional de 0.005;
6. série com ruído de referência tratada por EMA;
7. série com ruído de referência tratada por Wavelet Denoising causal.

EMA e Wavelet recebem como entrada a série contaminada com ruído gaussiano de
intensidade `0.002`. A série original não é filtrada no cenário experimental;
ela é mantida exclusivamente como referência para medir o ruído injetado, o
ruído removido e a distorção causada pelo tratamento.

Os rótulos dos sete cenários são calculados pelo `close` original. Essa verdade
de referência comum permite atribuir diferenças de desempenho às entradas
experimentais, em vez de mudar simultaneamente os dados e a definição de acerto.

Para evitar vazamento temporal:

- os dados nunca são embaralhados;
- o scaler é ajustado apenas nas linhas de treino;
- validação e teste recebem somente `transform`;
- a posição do alvo define a partição de cada sequência;
- os últimos rótulos de treino/validação são expurgados quando seu horizonte
  futuro atravessaria a fronteira da partição seguinte;
- cada janela contém apenas o candle-alvo e candles anteriores;
- o filtro Wavelet usa uma janela que termina no candle atual.

## Requisitos

- Windows 64 bits (necessário para a integração oficial com MetaTrader 5);
- Python 3.10, 3.11 ou 3.12;
- terminal MetaTrader 5 instalado e autenticado, ou um CSV histórico.

No PowerShell, dentro desta pasta:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se a política do PowerShell bloquear a ativação, também é possível chamar
diretamente `.venv\Scripts\python.exe` nos comandos seguintes.

## Configuração de exemplo: PETR4 M5

O arquivo `config.py` já contém este exemplo:

```python
usar_metatrader = True
simbolo = "PETR4"
timeframe = "M5"
quantidade_candles = 10_000
```

O nome do ativo depende da corretora e pode ter um sufixo. Confirme no
"Observação do Mercado" do terminal se o símbolo é `PETR4`, `PETR4.SA` ou outro.

Com o terminal aberto e conectado:

```powershell
python main.py --simbolo PETR4 --timeframe M5 --candles 10000
```

Os mesmos valores podem ser alterados diretamente na classe `ConfigProjeto`, em
`config.py`. Parâmetros como horizonte, limiar de retorno, janela LSTM, EMA,
Wavelet, épocas e proporções também ficam centralizados nesse arquivo.

### Escolha do limiar de retorno (theta)

O rótulo compara o fechamento atual com o fechamento após 10 candles. O valor
de `limiar_retorno` não deve ser menor que a incerteza típica dessa diferença,
pois pequenas oscilações dominadas pelo ruído seriam tratadas como movimentos
direcionais. O diagnóstico usa somente os 70% iniciais destinados ao treino.

Para ruído multiplicativo independente com desvio `sigma` nos dois preços, o
desvio aproximado do erro do retorno é `sqrt(2) * sigma`. Nos dados atuais,
`sigma = 0.002013` e o limite de 95% é `1.96 * sqrt(2) * sigma = 0.005579`.
A volatilidade robusta (1,4826 vezes o MAD) do retorno de 10 candles é
`0.004192`, enquanto o desvio-padrão clássico é `0.007600`. Pelo critério mais
conservador, adotou-se `theta = 0.008` (0,8%), que supera o limite do ruído e a
volatilidade clássica. Esse valor mantém 15,94% das amostras de treino nas
classes compra/venda. Para compensar o desbalanceamento resultante, o treino
usa pesos inversamente proporcionais à frequência de cada classe.

Os valores usados nessa decisão são salvos em
`resultados/diagnostico_filtros/comparacao_limiares_theta.csv` e
`resumo_limiar_theta.json`.

## Execução por CSV

O CSV deve conter estas colunas (nomes em minúsculas ou maiúsculas são aceitos):

```text
time,open,high,low,close,tick_volume
```

`time` pode ser uma data interpretável ou epoch em segundos. Para ignorar
completamente o MetaTrader 5:

```powershell
python main.py --sem-mt5 --csv .\meus_dados\PETR4_M5.csv
```

Para tentar o MT5 e usar o CSV apenas se a conexão falhar:

```powershell
python main.py --csv .\meus_dados\PETR4_M5.csv
```

Para trocar a rede e a pasta de saída:

```powershell
python main.py --sem-mt5 --csv .\PETR4_M5.csv --modelo GRU --saida .\execucao_gru
```

## Arquivos gerados

`dados_processados/` recebe os dados originais e um CSV por cenário. Cada CSV
inclui a partição temporal (`treino`, `validacao`, `teste`), as features antes da
normalização, as versões normalizadas, o retorno futuro e o rótulo.

`resultados/` recebe:

- `comparacao_cenarios.csv`: tabela principal solicitada;
- `comparacao_cenarios_detalhada.csv`: inclui também o FPR geral;
- `graficos/`: séries, matrizes de confusão, FPR e F1;
- `metricas/`: JSON completo por cenário, com TP, FP, FN e TN por classe;
- `predicoes/`: classe real, classe predita e probabilidades do teste;
- `modelos/`: redes `.keras` e scalers de cada cenário;
- `historicos_treinamento/`: loss e accuracy por época.

## Como interpretar

- **FPR de compra/venda**: entre os casos que não pertenciam à classe avaliada,
  proporção classificada incorretamente como compra ou venda. Quanto menor,
  menos alarmes falsos.
- **FDR de compra/venda**: entre os sinais emitidos daquela classe, proporção que
  estava errada. É uma leitura direta da falta de confiabilidade do sinal.
- **F1 macro**: equilíbrio entre precisão e recall, dando o mesmo peso a manter,
  comprar e vender, mesmo quando as classes têm frequências diferentes.
- **Matriz de confusão**: mostra quais classes são confundidas entre si.

Compare cada cenário ruidoso com `Original`. Crescimento de FPR/FDR junto com
queda de F1 sustenta a hipótese de que o ruído gera mais falsos sinais. Compare
EMA e Wavelet com os cenários ruidosos para verificar se o tratamento reduz esse
efeito sem eliminar sinais verdadeiros. Accuracy isolada pode enganar quando a
classe `manter` domina, por isso a análise deve priorizar também F1 macro, FPR,
FDR e a matriz de confusão.

Para uma conclusão acadêmica mais robusta, repita o experimento com diferentes
períodos históricos, ativos e sementes, e reporte média e desvio padrão. O
projeto fixa a semente por padrão para tornar uma execução reproduzível.

Após instalar as dependências, os testes determinísticos podem ser executados
sem MetaTrader 5 e sem treinar redes neurais:

```powershell
python -m unittest discover -s tests -v
```

## Estrutura

```text
main.py
config.py
requirements.txt
src/
  data_collection.py
  preprocessing.py
  noise.py
  labeling.py
  sequences.py
  model.py
  evaluation.py
  plots.py
  experiment.py
```
