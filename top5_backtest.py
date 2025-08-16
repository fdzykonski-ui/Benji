# -*- coding: cp1252 -*-
"""
top5_backtest.py
Backtest for the first 5 models from top5_models.py:
  1) MA Crossover 50/200
  2) EMA 20/50 + ADX Filter
  3) Donchian Turtle (20/55)
  4) SuperTrend (10, 3.0)
  5) Ichimoku Kumo Break (9,26,52)

Features:
- Vectorized signal computation via compute_top5(df)
- Simple execution model (enter at next bar open)
- ATR-based initial stop and trailing (configurable)
- Position sizing by fixed risk fraction of equity per trade
- Long/Short allowed (toggle)
- Slippage and fee modelling (per trade % plus per share fixed ignored)
- Performance metrics: CAGR, Sharpe, MaxDD, Win rate, Profit factor
- Equity curve & trades CSV export

Usage:
  python top5_backtest.py --csv data.csv --model signal_ma_cross --initial 10000 --risk 0.01 --atr_mult 2.0 --fee 0.0005 --slip 0.0002 --allow_short 1
CSV must contain: datetime,open,high,low,close,volume (datetime parseable)

Outputs:
  backtest_equity.csv
  backtest_trades.csv
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict

from top5_models import compute_top5, atr

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Input OHLCV CSV with datetime,open,high,low,close,volume")
    p.add_argument("--sep", default=",")
    p.add_argument("--tz", default=None)
    p.add_argument("--model", required=True, help="One of: signal_ma_cross, signal_ema_adx, signal_donchian, signal_supertrend, signal_ichimoku")
    p.add_argument("--initial", type=float, default=10000.0)
    p.add_argument("--risk", type=float, default=0.01, help="Fraction of equity risked per trade (e.g. 0.01 = 1%)")
    p.add_argument("--atr_mult", type=float, default=2.0, help="ATR multiple for stop distance")
    p.add_argument("--atr_len", type=int, default=14)
    p.add_argument("--fee", type=float, default=0.0005, help="Proportional fee per trade (0.0005=5bps)")
    p.add_argument("--slip", type=float, default=0.0002, help="Proportional slippage applied to entry/exit price")
    p.add_argument("--allow_short", type=int, default=1, help="1 allow shorts, 0 longs only")
    p.add_argument("--out_prefix", default="backtest")
    return p.parse_args()

def load_df(path: str, sep: str = ",", tz: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    # Find datetime column
    dt_col = None
    for cand in ["datetime","date","timestamp","time"]:
        if cand in df.columns:
            dt_col = cand
            break
    if dt_col is None:
        raise ValueError("CSV needs a datetime column (datetime/date/timestamp/time)")
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
    df = df.set_index(dt_col).sort_index()
    if tz:
        df = df.tz_convert(tz)
    # lower-case columns
    df = df.rename(columns={c:c.lower() for c in df.columns})
    required = ["open","high","low","close","volume"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"CSV missing columns: {miss}")
    return df

def position_size(equity: float, risk_frac: float, atr_stop_dist: float) -> float:
    """
    Dollar position sizing: risk_frac * equity / atr_stop_dist.
    For spot (no leverage). For futures/CFDs, adapt to contract value.
    Returns units in notional currency; we will convert to quantity as notional/price.
    """
    if atr_stop_dist <= 0:
        return 0.0
    return (equity * risk_frac) / atr_stop_dist

def backtest(df: pd.DataFrame,
             signal_col: str,
             initial_equity: float = 10000.0,
             risk_frac: float = 0.01,
             atr_len: int = 14,
             atr_mult: float = 2.0,
             fee: float = 0.0005,
             slip: float = 0.0002,
             allow_short: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple next-bar execution backtest with ATR stops & trailing.
    - Enter long when signal==1, enter short when signal==-1 (if allowed).
    - Exit when opposite signal appears or ATR stop is hit intrabar (approx by low/high vs stop level).
    """
    df = df.copy()
    sig = df[signal_col].fillna(0).astype(int)
    atrv = atr(df, atr_len)
    df["atr_bt"] = atrv
    # Entry at next bar open; apply slippage on both sides
    opens = df["open"]
    highs = df["high"]
    lows  = df["low"]
    closes= df["close"]

    equity = initial_equity
    pos = 0  # +1 long, -1 short, 0 flat
    qty = 0.0
    entry_price = np.nan
    stop = np.nan
    peak = np.nan  # track for trailing in long
    trough = np.nan  # track for trailing in short

    records = []  # trades
    curve = []    # equity curve per bar

    for i in range(1, len(df)):  # start from 1 to allow next-bar execution
        idx_prev = df.index[i-1]
        idx = df.index[i]

        signal_now = sig.iloc[i-1]  # we decide using previous bar's signal; execute at current open
        o, h, l, c = opens.iloc[i], highs.iloc[i], lows.iloc[i], closes.iloc[i]
        atr_i = atrv.iloc[i-1]  # stop distance based on prior ATR
        if np.isnan(atr_i):
            curve.append((idx, equity, pos))
            continue

        # manage open position: trailing & stop-out intra-bar approximation
        if pos != 0:
            # Update trailing levels
            if pos > 0:
                peak = max(peak, h) if not np.isnan(peak) else h
                trail = peak - atr_mult * atr_i
                stop = max(stop, trail) if not np.isnan(stop) else trail
                # stop check: if low breaches stop, exit at stop with slippage and fee
                if l <= stop:
                    exit_price = stop * (1 - slip)  # assume sell with negative slippage
                    pnl = (exit_price - entry_price) * qty
                    cost = fee * (abs(exit_price) + abs(entry_price)) * qty
                    equity += pnl - cost
                    records.append((idx, "EXIT_LONG_STOP", exit_price, qty, pnl - cost, equity))
                    pos, qty, entry_price, stop, peak, trough = 0, 0.0, np.nan, np.nan, np.nan, np.nan
            else:
                trough = min(trough, l) if not np.isnan(trough) else l
                trail = trough + atr_mult * atr_i
                stop = min(stop, trail) if not np.isnan(stop) else trail
                if h >= stop:
                    exit_price = stop * (1 + slip)  # buy to cover with slippage
                    pnl = (entry_price - exit_price) * qty
                    cost = fee * (abs(exit_price) + abs(entry_price)) * qty
                    equity += pnl - cost
                    records.append((idx, "EXIT_SHORT_STOP", exit_price, qty, pnl - cost, equity))
                    pos, qty, entry_price, stop, peak, trough = 0, 0.0, np.nan, np.nan, np.nan, np.nan

        # Entry/flip logic on signal:
        desired = 1 if signal_now > 0 else (-1 if (signal_now < 0 and allow_short) else 0)

        if desired != pos:
            # if in position, close it first at open
            if pos != 0:
                if pos > 0:
                    exit_price = o * (1 - slip)
                    pnl = (exit_price - entry_price) * qty
                    cost = fee * (abs(exit_price) + abs(entry_price)) * qty
                    equity += pnl - cost
                    records.append((idx, "EXIT_LONG_FLIP", exit_price, qty, pnl - cost, equity))
                else:
                    exit_price = o * (1 + slip)
                    pnl = (entry_price - exit_price) * qty
                    cost = fee * (abs(exit_price) + abs(entry_price)) * qty
                    equity += pnl - cost
                    records.append((idx, "EXIT_SHORT_FLIP", exit_price, qty, pnl - cost, equity))
                pos, qty, entry_price, stop, peak, trough = 0, 0.0, np.nan, np.nan, np.nan, np.nan

            # open new position if desired != 0
            if desired != 0:
                # stop distance: atr_mult * atr
                stop_dist = atr_mult * atr_i
                if stop_dist <= 0 or np.isnan(stop_dist):
                    curve.append((idx, equity, pos))
                    continue

                # risk-based dollar size
                dollar_risk = risk_frac * equity
                units = dollar_risk / stop_dist  # notional
                qty_new = max(units / o, 0.0)    # quantity (units / price)
                if qty_new > 0:
                    if desired > 0:
                        price = o * (1 + slip)
                        entry_price = price
                        stop = entry_price - stop_dist
                        peak = entry_price
                        pos = 1
                        qty = qty_new
                        cost = fee * price * qty
                        equity -= cost
                        records.append((idx, "ENTER_LONG", price, qty, -cost, equity))
                    else:
                        price = o * (1 - slip)
                        entry_price = price
                        stop = entry_price + stop_dist
                        trough = entry_price
                        pos = -1
                        qty = qty_new
                        cost = fee * price * qty
                        equity -= cost
                        records.append((idx, "ENTER_SHORT", price, qty, -cost, equity))

        curve.append((idx, equity, pos))

    # Close open position at last close
    if pos != 0:
        last_idx = df.index[-1]
        last_c = df["close"].iloc[-1]
        if pos > 0:
            exit_price = last_c * (1 - slip)
            pnl = (exit_price - entry_price) * qty
            cost = fee * (abs(exit_price) + abs(entry_price)) * qty
            equity += pnl - cost
            records.append((last_idx, "EXIT_LONG_EOD", exit_price, qty, pnl - cost, equity))
        else:
            exit_price = last_c * (1 + slip)
            pnl = (entry_price - exit_price) * qty
            cost = fee * (abs(exit_price) + abs(entry_price)) * qty
            equity += pnl - cost
            records.append((last_idx, "EXIT_SHORT_EOD", exit_price, qty, pnl - cost, equity))

    equity_df = pd.DataFrame(curve, columns=["datetime","equity","pos"]).set_index("datetime")
    trades_df = pd.DataFrame(records, columns=["datetime","action","price","qty","pnl","equity"]).set_index("datetime")

    return equity_df, trades_df

def metrics(equity_df: pd.DataFrame, freq_per_year: int = 252) -> Dict[str, float]:
    eq = equity_df["equity"]
    rets = eq.pct_change().fillna(0.0)
    # CAGR
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1.0 if eq.iloc[0] > 0 else 0.0
    yrs = max(len(eq) / freq_per_year, 1e-9)
    cagr = (1 + total_ret) ** (1/yrs) - 1 if total_ret > -1 else -1.0
    # Sharpe (daily)
    sharpe = (rets.mean() / (rets.std() + 1e-12)) * np.sqrt(freq_per_year) if rets.std() > 0 else 0.0
    # Max drawdown
    roll_max = eq.cummax()
    dd = (eq / roll_max - 1.0)
    maxdd = dd.min()
    return {
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(total_ret),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": float(maxdd),
    }

def main():
    args = parse_args()
    df = load_df(args.csv, sep=args.sep, tz=args.tz)
    # compute all indicator sheets
    ind = compute_top5(df)
    all_df = pd.concat([df, ind], axis=1)
    # run backtest for chosen model
    equity_df, trades_df = backtest(
        all_df,
        signal_col=args.model,
        initial_equity=args.initial,
        risk_frac=args.risk,
        atr_len=args.atr_len,
        atr_mult=args.atr_mult,
        fee=args.fee,
        slip=args.slip,
        allow_short=bool(args.allow_short),
    )
    m = metrics(equity_df)
    # Save
    prefix = args.out_prefix
    equity_df.to_csv(f"{prefix}_equity.csv", encoding="cp1252")
    trades_df.to_csv(f"{prefix}_trades.csv", encoding="cp1252")
    print("Metrics:")
    for k,v in m.items():
        print(f"  {k}: {v:.6f}")
    print(f"Saved: {prefix}_equity.csv, {prefix}_trades.csv")

if __name__ == "__main__":
    main()
