"""
Engenharia de atributos (indicadores técnicos) e criação da variável-alvo.

Regra de ouro contra vazamento de dados (data leakage):
  - Todo atributo do dia t usa SOMENTE informação disponível até o fechamento do dia t.
  - O alvo do dia t olha para o futuro (t+1 ... t+H) e NUNCA entra como atributo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONTE_PADRAO = 7  # dias à frente que o modelo tenta prever


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Índice de Força Relativa (Wilder). 0-100; >70 'sobrecomprado', <30 'sobrevendido'."""
    delta = close.diff()
    ganho = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    perda = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = ganho / perda.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atributos_moeda(g: pd.DataFrame) -> pd.DataFrame:
    """Calcula os indicadores para UMA moeda (série ordenada por data)."""
    g = g.sort_values("date").copy()
    c = g["close"]
    logret = np.log(c).diff()

    # Retornos passados (momentum) em várias janelas
    for n in (1, 3, 7, 14, 30):
        g[f"ret_{n}d"] = np.log(c / c.shift(n))

    # Volatilidade
    g["vol_7d"] = logret.rolling(7).std()
    g["vol_30d"] = logret.rolling(30).std()
    g["vol_ratio"] = g["vol_7d"] / g["vol_30d"]
    g["amplitude_dia"] = (g["high"] - g["low"]) / c

    # Tendência: distância do preço para médias móveis
    for n in (7, 30, 90):
        g[f"dist_sma{n}"] = c / c.rolling(n).mean() - 1
    g["sma7_sobre_sma30"] = c.rolling(7).mean() / c.rolling(30).mean() - 1

    # Osciladores
    g["rsi_14"] = rsi(c, 14)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    g["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c  # normalizado pelo preço
    sma20, std20 = c.rolling(20).mean(), c.rolling(20).std()
    g["bollinger_pct"] = (c - (sma20 - 2 * std20)) / (4 * std20)  # 0 = banda inferior, 1 = superior

    # Quedas a partir do topo recente
    g["drawdown_90d"] = c / c.rolling(90).max() - 1
    g["dist_min_30d"] = c / c.rolling(30).min() - 1

    # Volume e pressão compradora
    g["volume_rel_30d"] = np.log(g["quote_volume"] / g["quote_volume"].rolling(30).mean())
    g["taker_buy_ratio"] = g["taker_buy_base"] / g["volume"].replace(0, np.nan)
    g["taker_buy_ratio_7d"] = g["taker_buy_ratio"].rolling(7).mean()
    g["trades_rel_30d"] = np.log(g["trades"] / g["trades"].rolling(30).mean().replace(0, np.nan))
    return g


def criar_alvo(df: pd.DataFrame, horizonte: int = HORIZONTE_PADRAO) -> pd.DataFrame:
    """
    retorno_futuro = log(close[t+H] / close[t])
    alvo = 1 se o preço estiver MAIS ALTO daqui a H dias (bom momento para comprar), senão 0.
    """
    df = df.copy()
    futuro = df.groupby("symbol")["close"].shift(-horizonte)
    df["retorno_futuro"] = np.log(futuro / df["close"])
    df["alvo"] = (df["retorno_futuro"] > 0).astype(float)
    df.loc[df["retorno_futuro"].isna(), "alvo"] = np.nan
    return df


def montar_dataset(klines: pd.DataFrame, horizonte: int = HORIZONTE_PADRAO, mercado: str = "BTCUSDT") -> pd.DataFrame:
    """Gera atributos de cada moeda + atributos do 'mercado' (BTC) + alvo."""
    df = pd.concat([_atributos_moeda(g) for _, g in klines.groupby("symbol")], ignore_index=True)

    # Regime de mercado: o BTC costuma puxar as demais moedas
    btc = df[df["symbol"] == mercado][["date", "ret_7d", "ret_30d", "dist_sma30", "vol_30d"]]
    btc = btc.rename(columns=lambda c: c if c == "date" else f"btc_{c}")
    df = df.merge(btc, on="date", how="left")

    df = criar_alvo(df, horizonte)
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


COLUNAS_FEATURES = [
    "ret_1d", "ret_3d", "ret_7d", "ret_14d", "ret_30d",
    "vol_7d", "vol_30d", "vol_ratio", "amplitude_dia",
    "dist_sma7", "dist_sma30", "dist_sma90", "sma7_sobre_sma30",
    "rsi_14", "macd_hist", "bollinger_pct",
    "drawdown_90d", "dist_min_30d",
    "volume_rel_30d", "taker_buy_ratio", "taker_buy_ratio_7d", "trades_rel_30d",
    "btc_ret_7d", "btc_ret_30d", "btc_dist_sma30", "btc_vol_30d",
]
