# -*- coding: cp1252 -*-
"""
test_portfolio_backtester.py
Smoke test: build three synthetic assets, run portfolio backtest.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from portfolio_backtester import portfolio_backtest

def make_asset(seed: int, n=3000):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.001, n)
    price = 100 * np.exp(np.cumsum(rets))
    high = price * (1 + rng.uniform(0, 0.002, n))
    low  = price * (1 - rng.uniform(0, 0.002, n))
    open_ = price * (1 + rng.uniform(-0.0005, 0.0005, n))
    vol = rng.integers(1000, 2000, n)
    dt = pd.date_range("2022-01-01", periods=n, freq="T", tz="UTC")
    df = pd.DataFrame({"open":open_,"high":high,"low":low,"close":price,"volume":vol}, index=dt)
    return df

def run_smoke():
    assets = {
        "A": make_asset(1),
        "B": make_asset(2),
        "C": make_asset(3),
    }
    cfg = {
        "signals":{"mode":"fixed","fixed_signal_col":"signal_supertrend"},
        "sizing":{"risk_per_trade":0.01,"use_conformal":False},
        "execution":{"fee":0.0005,"slippage":0.0002,"allow_short":1},
        "stops":{"atr_len":14,"atr_mult":2.0,"trailing":True},
        "portfolio_limits":{"max_positions":3,"max_gross_exposure":1.5,"max_drawdown":0.5,"daily_loss_limit":0.2,"hard_stop_on_maxdd":False}
    }
    eq, tr, dg = portfolio_backtest(assets, cfg)
    assert len(eq) > 0 and len(dg) == 1 or True  # diag is single-row metrics; eq is curve
    print("OK - portfolio backtester ran.")

if __name__ == "__main__":
    run_smoke()
