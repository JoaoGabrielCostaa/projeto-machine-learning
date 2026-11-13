# Sinais de Compra e Venda de Criptomoedas com Machine Learning

**C318 – Fundamentos de Machine Learning · Inatel** — Projeto Computacional (NP2)

> ⚠️ Projeto acadêmico. Nada aqui é recomendação de investimento.

## Resumo
O modelo de classificação estima, a cada dia, a probabilidade de uma criptomoeda estar **mais cara daqui a 7 dias**. Essa probabilidade vira um sinal **COMPRAR / MANTER / VENDER**. Para medir o valor de negócio, a estratégia é simulada (backtest) com custos de transação e comparada ao *Buy & Hold*.

| Etapa (escopo C318) | Onde está |
|---|---|
| Contexto e objetivos | Notebook, seção 1 |
| Pesquisa e enquadramento (perguntas de negócio, tipo de aprendizagem, métricas) | Notebook, seção 2 |
| Coleta de dados | `src/coleta.py` + notebook, seção 3 |
| Desenvolvimento: análise (EDA) e modelagem | Notebook, seções 4–7 |
| Geração de análises e resultados | Notebook, seções 8–11 + `reports/figures/` |

## Enquadramento
- **Tipo:** aprendizagem supervisionada, classificação binária.
- **Alvo:** `1` se `close[t+7] > close[t]`.
- **Atributos:** 26 indicadores técnicos (momentum, volatilidade, médias móveis, RSI, MACD, Bollinger, volume, pressão compradora e regime do BTC).
- **Modelos:** baseline de classe majoritária, Regressão Logística, Random Forest e XGBoost.
- **Validação:** divisão **temporal** com purga (treino até 2023 · validação 2024 · teste 2025→hoje), mais `TimeSeriesSplit` com 5 *folds*.
- **Métricas técnicas:** AUC, acurácia, precisão, recall e F1.
- **Métricas de negócio:** retorno, Sharpe, *max drawdown* e nº de operações (taxa de 0,1%).

## Dados
- **Fonte:** API pública da Binance `GET /api/v3/klines` (velas diárias), sem chave de API.
- **Moedas:** BTC, ETH, BNB, SOL, XRP, ADA, DOGE e LTC (pares USDT), de 2017 até hoje. São ~24 mil linhas (moeda × dia).
- **Por que não usamos o dataset do Kaggle "Binance Crypto Pairs 24h Ticker":** ele é um **snapshot único**, sem série temporal, e por isso não permite modelar *quando* comprar ou vender.

## Resultados (execução de 07/11/2026; teste de 01/2025 a 10/2026)
| | Estratégia ML (XGBoost) | Buy & Hold | Regra SMA30 |
|---|---|---|---|
| Retorno total (carteira de 8 moedas) | **+16%** | −47% | −22% |
| Sharpe | **0,40** | −0,24 | −0,27 |
| Queda máxima | −44% | −67% | −41% |

- AUC no teste: **0,548**. O sinal é fraco, mas consistente com a literatura. A estratégia supera 500 estratégias aleatórias com a mesma exposição ao mercado.
- **Ressalvas:** o período de teste foi um mercado de baixa, e a AUC fica abaixo de 0,5 nos *folds* de 2019–2021. Os detalhes estão na seção 11 do notebook.

## Estrutura
```
├── projeto_cripto_ml.ipynb   # notebook principal (documentação + análise + modelos)
├── src/
│   ├── coleta.py             # download do histórico na API da Binance
│   ├── features.py           # indicadores técnicos e variável-alvo
│   └── estrategia.py         # sinais COMPRAR/MANTER/VENDER e backtest
├── tools/gerar_notebook.py   # gera o .ipynb a partir do código-fonte das células
├── data/raw/                 # CSVs baixados (um por moeda + consolidado)
├── reports/figures/          # gráficos exportados para a apresentação
└── requirements.txt
```

## Como executar
```bash
pip install -r requirements.txt
```
```bash
python -m src.coleta
```
```bash
python -m nbconvert --to notebook --execute --inplace projeto_cripto_ml.ipynb
```
Também dá para abrir `projeto_cripto_ml.ipynb` no VS Code ou no Jupyter e rodar célula por célula. Para baixar dados mais recentes, use `ATUALIZAR = True` na seção 3 ou rode `python -m src.coleta` de novo.

Opções de coleta: `python -m src.coleta --symbols BTCUSDT ETHUSDT --interval 4h --start 2020-01-01`.

## Próximos passos
- Velas de 4h/1h e outros horizontes de previsão (3 e 14 dias).
- Atributos de sentimento (Fear & Greed Index), *funding rate* de futuros e dados on-chain.
- Validação *walk-forward* com re-treino periódico.
- Formulação como regressão do retorno ou como problema de 3 classes.
