# -*- coding: cp1252 -*-
"""
top5_models.py
Computes indicators and trading signals for the first 5 models:
  1) MA Crossover 50/200
  2) EMA 20/50 + ADX Filter
  3) Donchian Turtle (20/55)
  4) SuperTrend Rider (10, 3.0)
  5) Ichimoku Kumo Break (9,26,52)

Input:  DataFrame with columns ['open','high','low','close','volume'] and DatetimeIndex
Output: DataFrame with indicator columns and the following signal columns (1 long, -1 short, 0 flat):

  signal_ma_cross
  signal_ema_adx
  signal_donchian
  signal_supertrend
  signal_ichimoku

All calculations are vectorized; minimal look-ahead. ATR/ADX use Wilder smoothing.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

# ------------------------------
# Helpers: Wilder smoothing, ATR, ADX, MAs
# ------------------------------

def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    ema = series.ewm(alpha=1/period, adjust=False).mean()
    return ema

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    tr.name = "tr"
    return tr

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return _wilder_smooth(tr, period).rename(f"atr_{period}")

def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(df)

    atr_w = _wilder_smooth(tr, period)
    plus_di = 100 * _wilder_smooth(pd.Series(plus_dm, index=df.index), period) / atr_w
    minus_di = 100 * _wilder_smooth(pd.Series(minus_dm, index=df.index), period) / atr_w
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_ = _wilder_smooth(dx, period)

    out = pd.DataFrame({
        f"+di_{period}": plus_di,
        f"-di_{period}": minus_di,
        f"adx_{period}": adx_
    })
    return out

# ------------------------------
# Model 1: MA Crossover 50/200
# ------------------------------

def model_ma_crossover(df: pd.DataFrame, fast: int = 50, slow: int = 200, atr_len: int = 14) -> pd.DataFrame:
    close = df["close"]
    sma_fast = sma(close, fast).rename(f"sma_{fast}")
    sma_slow = sma(close, slow).rename(f"sma_{slow}")
    atrv = atr(df, atr_len)

    cross_up = (sma_fast > sma_slow) & (sma_fast.shift() <= sma_slow.shift())
    cross_dn = (sma_fast < sma_slow) & (sma_fast.shift() >= sma_slow.shift())

    signal = pd.Series(0, index=df.index, dtype=int)
    signal = signal.where(~cross_up, 1)
    signal = signal.where(~cross_dn, -1)

    res = pd.DataFrame({
        f"sma_{fast}": sma_fast,
        f"sma_{slow}": sma_slow,
        f"atr_{atr_len}": atrv,
        "signal_ma_cross": signal
    })
    return res

# ------------------------------
# Model 2: EMA 20/50 + ADX filter
# ------------------------------

def model_ema_adx(df: pd.DataFrame, f: int = 20, s: int = 50, adx_len: int = 14, adx_th: float = 20.0) -> pd.DataFrame:
    c = df["close"]
    ema_f = ema(c, f).rename(f"ema_{f}")
    ema_s_ = ema(c, s).rename(f"ema_{s}")
    adxd = adx(df, adx_len)
    adx_col = adxd[f"adx_{adx_len}"]

    long_sig = (ema_f > ema_s_) & (adx_col > adx_th)
    short_sig = (ema_f < ema_s_) & (adx_col > adx_th)

    signal = pd.Series(0, index=df.index, dtype=int)
    signal = signal.where(~long_sig, 1)
    signal = signal.where(~short_sig, -1)

    res = pd.concat([ema_f, ema_s_, adxd], axis=1)
    res["signal_ema_adx"] = signal
    return res

# ------------------------------
# Model 3: Donchian Turtle (20/55)
# ------------------------------

def donchian(df: pd.DataFrame, period: int) -> pd.DataFrame:
    upper = df["high"].rolling(period, min_periods=period).max()
    lower = df["low"].rolling(period, min_periods=period).min()
    mid = (upper + lower) / 2.0
    return pd.DataFrame({f"donchian_u_{period}": upper, f"donchian_l_{period}": lower, f"donchian_m_{period}": mid})

def model_donchian_turtle(df: pd.DataFrame, fast: int = 20, slow: int = 55) -> pd.DataFrame:
    dc_fast = donchian(df, fast)
    dc_slow = donchian(df, slow)
    c = df["close"]
    # Entry: breakout fast channel; optional confirm with slow
    long_sig = c > dc_fast[f"donchian_u_{fast}"]
    short_sig = c < dc_fast[f"donchian_l_{fast}"]
    signal = pd.Series(0, index=df.index, dtype=int)
    signal = signal.where(~long_sig, 1)
    signal = signal.where(~short_sig, -1)

    res = pd.concat([dc_fast, dc_slow], axis=1)
    res["signal_donchian"] = signal
    return res

# ------------------------------
# Model 4: SuperTrend Rider (10, 3.0)
# ------------------------------

def supertrend(df: pd.DataFrame, atr_period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    tr = true_range(df)
    atrv = tr.rolling(atr_period, min_periods=atr_period).mean()
    hl2 = (df["high"] + df["low"]) / 2.0
    upper = hl2 + multiplier * atrv
    lower = hl2 - multiplier * atrv

    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    if len(df) == 0:
        return pd.DataFrame({"supertrend": st, "supertrend_dir": direction.fillna(0)})
    st.iloc[0] = hl2.iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(df)):
        prev_st = st.iloc[i-1]
        prev_dir = direction.iloc[i-1]
        c = df["close"].iloc[i]
        up = upper.iloc[i]
        lo = lower.iloc[i]

        if prev_dir >= 0:
            st_i = min(up, prev_st) if c > prev_st else up
            dir_i = 1 if c > st_i else -1
        else:
            st_i = max(lo, prev_st) if c < prev_st else lo
            dir_i = -1 if c < st_i else 1

        st.iloc[i] = st_i
        direction.iloc[i] = dir_i

    return pd.DataFrame({"supertrend": st, "supertrend_dir": direction.fillna(0).astype(int)})

def model_supertrend(df: pd.DataFrame, atr_period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    st = supertrend(df, atr_period=atr_period, multiplier=multiplier)
    signal = st["supertrend_dir"].copy()
    signal.name = "signal_supertrend"
    return pd.concat([st, signal], axis=1)

# ------------------------------
# Model 5: Ichimoku Kumo Break (9,26,52)
# ------------------------------

def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]

    conv = (high.rolling(tenkan, min_periods=tenkan).max() + low.rolling(tenkan, min_periods=tenkan).min()) / 2.0
    base = (high.rolling(kijun, min_periods=kijun).max() + low.rolling(kijun, min_periods=kijun).min()) / 2.0
    span_a = ((conv + base) / 2.0).shift(kijun)
    span_b = ((high.rolling(senkou, min_periods=senkou).max() + low.rolling(senkou, min_periods=senkou).min()) / 2.0).shift(kijun)
    lag = close.shift(-kijun)

    df_out = pd.DataFrame({
        "tenkan": conv,
        "kijun": base,
        "senkou_a": span_a,
        "senkou_b": span_b,
        "chikou": lag
    })
    return df_out

def model_ichimoku_break(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> pd.DataFrame:
    ich = ichimoku(df, tenkan, kijun, senkou)
    c = df["close"]
    cloud_top = ich[["senkou_a", "senkou_b"]].max(axis=1)
    cloud_bot = ich[["senkou_a", "senkou_b"]].min(axis=1)
    # Signal: close above cloud and tenkan>kijun -> long; below cloud and tenkan<kijun -> short
    long_sig = (c > cloud_top) & (ich["tenkan"] > ich["kijun"])
    short_sig = (c < cloud_bot) & (ich["tenkan"] < ich["kijun"])
    signal = pd.Series(0, index=df.index, dtype=int)
    signal = signal.where(~long_sig, 1)
    signal = signal.where(~short_sig, -1)

    res = ich.copy()
    res["cloud_top"] = cloud_top
    res["cloud_bot"] = cloud_bot
    res["signal_ichimoku"] = signal
    return res

# ------------------------------
# Orchestrator
# ------------------------------

def compute_top5(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Compute indicators and signals for the five models and return a joined DataFrame.\"\"\"
    if not set(["open","high","low","close","volume"]).issubset(df.columns):
        raise ValueError("DataFrame must have columns: open, high, low, close, volume")

    m1 = model_ma_crossover(df)
    m2 = model_ema_adx(df)
    m3 = model_donchian_turtle(df)
    m4 = model_supertrend(df)
    m5 = model_ichimoku_break(df)

    out = pd.concat([m1, m2, m3, m4, m5], axis=1)
    return out

# ------------------------------
# Optional CLI
# ------------------------------

def _demo_cli():
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Compute Top-5 trading models on OHLCV CSV.")
    parser.add_argument("--csv", required=True, help="Path to CSV with columns: datetime,open,high,low,close,volume")
    parser.add_argument("--sep", default=",")
    parser.add_argument("--tz", default=None, help="Optional timezone name for index")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, sep=args.sep)
    # Allow various datetime column names
    dt_col = None
    for cand in ["datetime","date","timestamp","time"]:
        if cand in df.columns:
            dt_col = cand
            break
    if dt_col is None:
        raise SystemExit("No datetime column found (expected one of: datetime,date,timestamp,time)")

    df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
    df = df.set_index(dt_col).sort_index()
    if args.tz:
        df = df.tz_convert(args.tz)

    # Basic column normalization
    rename_map = {c:c.lower() for c in df.columns}
    df = df.rename(columns=rename_map)

    required = ["open","high","low","close","volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}")

    out = compute_top5(df)
    out.to_csv("top5_output.csv", encoding="cp1252")
    print("Saved: top5_output.csv")

if __name__ == "__main__":
    try:
        _demo_cli()
    except Exception as e:
        print(f"Error: {e}")
