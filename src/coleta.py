"""
Coleta de dados históricos (velas / candles OHLCV) da API pública da Binance.

Endpoint usado: GET /api/v3/klines  (dados de mercado públicos, não exige chave de API)
Documentação:   https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints

Uso pela linha de comando:
    python -m src.coleta                       # moedas e período padrão
    python -m src.coleta --symbols BTCUSDT ETHUSDT --interval 1d --start 2018-01-01
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

# data-api.binance.vision é o espelho oficial só de dados de mercado; api.binance.com é o principal.
BASE_URLS = ["https://api.binance.com", "https://data-api.binance.vision"]
KLINES_PATH = "/api/v3/klines"
LIMIT = 1000  # máximo de velas por requisição

SIMBOLOS_PADRAO = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LTCUSDT"]

COLUNAS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _get(params: dict) -> list:
    """Faz a requisição tentando os hosts da Binance em ordem (com retentativas simples)."""
    ultimo_erro = None
    for tentativa in range(3):
        for base in BASE_URLS:
            try:
                r = requests.get(base + KLINES_PATH, params=params, timeout=20)
                if r.status_code == 429:  # limite de requisições: espera e tenta de novo
                    time.sleep(int(r.headers.get("Retry-After", 5)))
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                ultimo_erro = e
        time.sleep(2 * (tentativa + 1))
    raise RuntimeError(f"Falha ao baixar dados da Binance: {ultimo_erro}")


def baixar_klines(symbol: str, interval: str = "1d", start: str = "2017-01-01", end: str | None = None) -> pd.DataFrame:
    """Baixa todas as velas de `symbol` entre `start` e `end` (paginando de 1000 em 1000)."""
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else None

    linhas: list = []
    while True:
        params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": LIMIT}
        if end_ms:
            params["endTime"] = end_ms
        lote = _get(params)
        if not lote:
            break
        linhas.extend(lote)
        start_ms = lote[-1][0] + 1  # próxima página começa após a última vela recebida
        if len(lote) < LIMIT:
            break
        time.sleep(0.2)  # gentileza com o limite de requisições da API

    df = pd.DataFrame(linhas, columns=COLUNAS).drop(columns="ignore")
    num = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]
    df[num] = df[num].astype(float)
    df["trades"] = df["trades"].astype(int)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.drop(columns="open_time").drop_duplicates("date").sort_values("date").reset_index(drop=True)
    df.insert(0, "symbol", symbol)

    # Remove a vela ainda em formação (o dia atual não terminou -> dado incompleto)
    agora = pd.Timestamp.now(tz="UTC").tz_localize(None)
    df = df[df["close_time"] < agora].reset_index(drop=True)
    return df


def coletar(symbols=None, interval="1d", start="2017-01-01", end=None, pasta: Path = DATA_DIR) -> pd.DataFrame:
    """Baixa várias moedas, salva um CSV por moeda e um CSV consolidado. Retorna o consolidado."""
    symbols = symbols or SIMBOLOS_PADRAO
    pasta.mkdir(parents=True, exist_ok=True)
    frames = []
    for s in symbols:
        df = baixar_klines(s, interval, start, end)
        df.to_csv(pasta / f"{s}_{interval}.csv", index=False)
        print(f"{s:10s} {len(df):5d} velas  {df['date'].min().date()} -> {df['date'].max().date()}")
        frames.append(df)
    todos = pd.concat(frames, ignore_index=True)
    todos.to_csv(pasta / f"klines_{interval}.csv", index=False)
    return todos


def carregar(interval: str = "1d", pasta: Path = DATA_DIR) -> pd.DataFrame:
    """Lê o CSV consolidado salvo por `coletar`."""
    return pd.read_csv(pasta / f"klines_{interval}.csv", parse_dates=["date", "close_time"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Baixa histórico de velas da Binance.")
    p.add_argument("--symbols", nargs="+", default=SIMBOLOS_PADRAO)
    p.add_argument("--interval", default="1d", help="1h, 4h, 1d, 1w ...")
    p.add_argument("--start", default="2017-01-01")
    p.add_argument("--end", default=None)
    a = p.parse_args()
    coletar(a.symbols, a.interval, a.start, a.end)
