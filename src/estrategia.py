"""
Transforma as probabilidades do modelo em sinais (COMPRAR / MANTER / VENDER) e simula
(backtest) a estratégia contra o "Buy & Hold" (comprar no início e nunca vender).

Hipóteses da simulação (mercado à vista / spot, sem venda a descoberto):
  - Decisão tomada no fechamento do dia t; posição vale para o retorno de t -> t+1.
  - COMPRAR  -> fica 100% na moeda;  VENDER -> fica 100% em caixa (USDT);
    MANTER   -> mantém a posição do dia anterior (evita trocar de posição toda hora).
  - Custo de 0,1% por operação (taxa padrão da Binance spot).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TAXA = 0.001


def gerar_sinais(prob: pd.Series, limiar_compra: float = 0.55, limiar_venda: float = 0.45) -> pd.Series:
    sinal = pd.Series("MANTER", index=prob.index)
    sinal[prob >= limiar_compra] = "COMPRAR"
    sinal[prob <= limiar_venda] = "VENDER"
    return sinal


def _posicao(sinal: pd.Series) -> pd.Series:
    pos = sinal.map({"COMPRAR": 1.0, "VENDER": 0.0, "MANTER": np.nan})
    return pos.ffill().fillna(0.0)  # começa em caixa até o primeiro sinal de compra


def simular(df: pd.DataFrame, col_prob: str = "prob", limiar_compra: float = 0.55,
            limiar_venda: float = 0.45, taxa: float = TAXA) -> pd.DataFrame:
    """
    df precisa de: date, symbol, close, <col_prob>.
    Retorna por (symbol, date): sinal, posição, retorno da estratégia e do buy & hold.
    """
    saidas = []
    for s, g in df.sort_values("date").groupby("symbol"):
        g = g[["date", "symbol", "close", col_prob]].copy()
        g["sinal"] = gerar_sinais(g[col_prob], limiar_compra, limiar_venda).values
        g["posicao"] = _posicao(g["sinal"]).values
        g["ret_ativo"] = g["close"].pct_change().shift(-1).fillna(0.0)  # retorno de t -> t+1
        trocas = g["posicao"].diff().abs().fillna(g["posicao"])
        g["ret_estrategia"] = g["posicao"] * g["ret_ativo"] - trocas * taxa
        g["ret_buyhold"] = g["ret_ativo"]
        g["n_trocas"] = trocas
        saidas.append(g)
    return pd.concat(saidas, ignore_index=True)


def _max_drawdown(curva: pd.Series) -> float:
    return float((curva / curva.cummax() - 1).min())


def metricas(ret: pd.Series, dias_ano: int = 365) -> dict:
    curva = (1 + ret).cumprod()
    anos = len(ret) / dias_ano
    total = curva.iloc[-1] - 1
    return {
        "retorno_total": total,
        "retorno_anual": (1 + total) ** (1 / anos) - 1 if anos > 0 else np.nan,
        "volatilidade_anual": ret.std() * np.sqrt(dias_ano),
        "sharpe": ret.mean() / ret.std() * np.sqrt(dias_ano) if ret.std() > 0 else np.nan,
        "max_drawdown": _max_drawdown(curva),
    }


def resumo(sim: pd.DataFrame) -> pd.DataFrame:
    """Tabela comparando Estratégia x Buy & Hold por moeda e na carteira igualmente ponderada."""
    linhas = []
    for s, g in sim.groupby("symbol"):
        e, b = metricas(g["ret_estrategia"]), metricas(g["ret_buyhold"])
        linhas.append({
            "moeda": s,
            "estrat_retorno": e["retorno_total"], "bh_retorno": b["retorno_total"],
            "estrat_sharpe": e["sharpe"], "bh_sharpe": b["sharpe"],
            "estrat_max_dd": e["max_drawdown"], "bh_max_dd": b["max_drawdown"],
            "tempo_posicionado": g["posicao"].mean(), "n_operacoes": int(g["n_trocas"].sum()),
        })
    cart = carteira(sim)
    e, b = metricas(cart["ret_estrategia"]), metricas(cart["ret_buyhold"])
    linhas.append({
        "moeda": "CARTEIRA (média)",
        "estrat_retorno": e["retorno_total"], "bh_retorno": b["retorno_total"],
        "estrat_sharpe": e["sharpe"], "bh_sharpe": b["sharpe"],
        "estrat_max_dd": e["max_drawdown"], "bh_max_dd": b["max_drawdown"],
        "tempo_posicionado": sim["posicao"].mean(), "n_operacoes": int(sim["n_trocas"].sum()),
    })
    return pd.DataFrame(linhas).set_index("moeda")


def carteira(sim: pd.DataFrame) -> pd.DataFrame:
    """Carteira com peso igual entre as moedas (rebalanceada diariamente)."""
    return sim.groupby("date")[["ret_estrategia", "ret_buyhold"]].mean()
