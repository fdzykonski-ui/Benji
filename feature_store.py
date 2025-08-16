# -*- coding: cp1252 -*-
"""
feature_store.py
Step 2: Feature Store + Leakage-safe Lags + Triple-Barrier Labeling + Caching (simple).

- Loads OHLCV CSV
- Computes a configurable feature set (basic + TA)
- Applies leak-guard shift to all features (default 1 bar)
- Computes triple-barrier labels using ATR or fractional volatility
- Saves features, labels, and a merged dataset (aligned index)

Usage:
  python feature_store.py --csv data.csv --config feature_config.yaml --out_prefix step2 --force 0

Outputs (configurable in YAML):
  features.parquet, labels.parquet, dataset.parquet, dataset.csv

Dependencies: pandas, numpy; reuses atr/adx from top5_models.py
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Local import (Step 1 / earlier)
from top5_models import atr, adx

# ------------------------
# Config loader
# ------------------------
def load_config(path: str | Path) -> dict:
    p = Path(path)
    txt = p.read_text(encoding="cp1252")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(txt)
    except Exception:
        return json.loads(txt)

def hash_config(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]

# ------------------------
# IO helpers
# ------------------------
def load_df(path: str, sep: str = ",", tz: str | None = "Europe/Berlin") -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    # datetime col
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
    df = df.rename(columns={c:c.lower() for c in df.columns})
    required = ["open","high","low","close","volume"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"CSV missing columns: {miss}")
    return df

# ------------------------
# Basic features
# ------------------------
def simple_returns(c: pd.Series, n: int) -> pd.Series:
    return c.pct_change(n).rename(f"ret_{n}")

def log_returns(c: pd.Series, n: int) -> pd.Series:
    return (np.log(c) - np.log(c.shift(n))).rename(f"logret_{n}")

def rolling_mean(c: pd.Series, n: int) -> pd.Series:
    return c.rolling(n, min_periods=n).mean().rename(f"roll_mean_{n}")

def rolling_std(c: pd.Series, n: int) -> pd.Series:
    return c.rolling(n, min_periods=n).std().rename(f"roll_std_{n}")

def rolling_minmax_range(h: pd.Series, l: pd.Series, n: int) -> pd.Series:
    hi = h.rolling(n, min_periods=n).max()
    lo = l.rolling(n, min_periods=n).min()
    return (hi - lo).rename(f"roll_range_{n}")

# ------------------------
# TA features (stand-alone implementations)
# ------------------------
def rsi(c: pd.Series, period: int = 14) -> pd.Series:
    delta = c.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-12)
    out = 100 - (100 / (1 + rs))
    out.name = f"rsi_{period}"
    return out

def macd(c: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.DataFrame:
    ema_fast = c.ewm(span=fast, adjust=False).mean()
    ema_slow = c.ewm(span=slow, adjust=False).mean()
    macd_line = (ema_fast - ema_slow).rename(f"macd_{fast}_{slow}")
    signal = macd_line.ewm(span=sig, adjust=False).mean().rename(f"macd_sig_{sig}")
    hist = (macd_line - signal).rename("macd_hist")
    return pd.DataFrame({macd_line.name: macd_line, signal.name: signal, hist.name: hist})

def stochastic(h: pd.Series, l: pd.Series, c: pd.Series, k: int = 14, d: int = 3) -> pd.DataFrame:
    lowest = l.rolling(k, min_periods=k).min()
    highest = h.rolling(k, min_periods=k).max()
    percent_k = ((c - lowest) / (highest - lowest + 1e-12) * 100).rename(f"stoch_k_{k}")
    percent_d = percent_k.rolling(d, min_periods=d).mean().rename(f"stoch_d_{d}")
    return pd.DataFrame({percent_k.name: percent_k, percent_d.name: percent_d})

def bollinger(c: pd.Series, period: int = 20, k: float = 2.0) -> pd.DataFrame:
    ma = c.rolling(period, min_periods=period).mean()
    sd = c.rolling(period, min_periods=period).std()
    upper = (ma + k * sd).rename(f"bb_up_{period}")
    lower = (ma - k * sd).rename(f"bb_lo_{period}")
    width = (upper - lower).rename(f"bb_width_{period}")
    pb = ((c - lower) / (upper - lower + 1e-12)).rename(f"bb_pctb_{period}")
    return pd.DataFrame({upper.name: upper, lower.name: lower, width.name: width, pb.name: pb})

# ------------------------
# Feature store builder
# ------------------------
def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = df["close"]
    h, l, v = df["high"], df["low"], df["volume"]

    feats = []

    basic = cfg.get("features", {}).get("basic", {})
    for n in basic.get("returns", []) or []:
        feats.append(simple_returns(c, int(n)))
    for n in basic.get("log_returns", []) or []:
        feats.append(log_returns(c, int(n)))
    for n in basic.get("rolling_mean", []) or []:
        feats.append(rolling_mean(c, int(n)))
    for n in basic.get("rolling_std", []) or []:
        feats.append(rolling_std(c, int(n)))
    for n in basic.get("rolling_minmax", []) or []:
        feats.append(rolling_minmax_range(h, l, int(n)))

    ta = cfg.get("features", {}).get("ta", {})
    rsi_cfg = ta.get("rsi", {})
    if rsi_cfg.get("enabled", False):
        feats.append(rsi(c, int(rsi_cfg.get("period", 14))))

    macd_cfg = ta.get("macd", {})
    if macd_cfg.get("enabled", False):
        feats.append(macd(c, int(macd_cfg.get("fast", 12)), int(macd_cfg.get("slow", 26)), int(macd_cfg.get("signal", 9))))

    stoch_cfg = ta.get("stochastic", {})
    if stoch_cfg.get("enabled", False):
        feats.append(stochastic(h, l, c, int(stoch_cfg.get("k", 14)), int(stoch_cfg.get("d", 3))))

    bb_cfg = ta.get("bollinger", {})
    if bb_cfg.get("enabled", False):
        feats.append(bollinger(c, int(bb_cfg.get("period", 20)), float(bb_cfg.get("k", 2.0))))

    atr_adx_cfg = ta.get("atr_adx", {})
    if atr_adx_cfg.get("enabled", False):
        feats.append(atr(df, int(atr_adx_cfg.get("atr_len", 14))).rename(f"atr_{int(atr_adx_cfg.get('atr_len',14))}"))
        feats.append(adx(df, int(atr_adx_cfg.get("adx_len", 14))))

    X = pd.concat(feats, axis=1)

    # leak guard: shift features by N bars to remove lookahead
    shift_n = int(cfg.get("leak_guard", {}).get("shift", 1))
    if shift_n > 0:
        X = X.shift(shift_n)

    return X

# ------------------------
# Triple-Barrier Labeling
# ------------------------
def triple_barrier_labels(df: pd.DataFrame,
                          horizon_bars: int = 20,
                          tp_mult: float = 2.0,
                          sl_mult: float = 1.0,
                          use_atr: bool = True,
                          atr_len: int = 14,
                          zero_threshold: float = 0.0,
                          side: str = "both") -> pd.DataFrame:
    """
    For each time t, set upper/lower barriers relative to close[t]:
      upper = close[t] * (1 + tp_mult * vol[t])
      lower = close[t] * (1 - sl_mult * vol[t])
    where vol[t] = ATR[t]/close[t] if use_atr else rolling_vol (not implemented here).
    We then look forward up to horizon_bars to see which barrier is hit first.
    Labels:
      y =  1  if upper first (or final return >= zero_threshold if neither)
      y = -1  if lower first (or final return <= -zero_threshold if neither)
      y =  0  if neither and |final return| < zero_threshold
    Returns columns: ['y','ret','exit_index','exit_type'].
    """
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)

    if use_atr:
        atrv = atr(df, atr_len)
        vol_frac = (atrv / c).fillna(0.0)
    else:
        # fallback: use rolling std of returns
        vol_frac = c.pct_change().rolling(atr_len, min_periods=atr_len).std().fillna(0.0)

    n = len(df)
    y = np.zeros(n, dtype=int)
    ret = np.zeros(n, dtype=float)
    exit_idx = np.full(n, -1, dtype=int)
    exit_type = np.array([""] * n, dtype=object)

    for i in range(n - horizon_bars - 1):
        price = c.iloc[i]
        vf = vol_frac.iloc[i]
        if not np.isfinite(price) or price <= 0 or vf <= 0:
            continue
        up = price * (1.0 + tp_mult * vf)
        dn = price * (1.0 - sl_mult * vf)

        # scan forward
        hit = None
        hit_j = None
        for j in range(1, horizon_bars + 1):
            hi = h.iloc[i + j]
            lo = l.iloc[i + j]
            if np.isnan(hi) or np.isnan(lo):
                continue
            if hi >= up:
                hit = "tp"
                hit_j = j
                break
            if lo <= dn:
                hit = "sl"
                hit_j = j
                break

        if hit is None:
            # time limit
            j = horizon_bars
            r = (c.iloc[i + j] / price) - 1.0
            if abs(r) < zero_threshold:
                y[i] = 0
            else:
                y[i] = 1 if r > 0 else -1
            ret[i] = r
            exit_idx[i] = i + j
            exit_type[i] = "t1"
        else:
            if hit == "tp":
                y[i] = 1
                r = (up / price) - 1.0
            else:
                y[i] = -1
                r = (dn / price) - 1.0
            ret[i] = r
            exit_idx[i] = i + hit_j
            exit_type[i] = hit

    # side filtering
    if side == "long_only":
        y = np.where(y < 0, 0, y)
    elif side == "short_only":
        y = np.where(y > 0, 0, y)

    out = pd.DataFrame({
        "y": y,
        "ret": ret,
        "exit_index": exit_idx,
        "exit_type": exit_type
    }, index=df.index)
    return out

# ------------------------
# Orchestrator
# ------------------------
def run_pipeline(csv_path: str, cfg_path: str, out_prefix: str = "step2", sep: str = ",", tz: str | None = "Europe/Berlin", force: bool = False):
    cfg = load_config(cfg_path)
    tz = cfg.get("meta", {}).get("timezone", tz)
    df = load_df(csv_path, sep=sep, tz=tz)

    # Build features
    features_path = cfg.get("exports", {}).get("features_parquet", "features.parquet")
    labels_path = cfg.get("exports", {}).get("labels_parquet", "labels.parquet")
    dataset_parquet = cfg.get("exports", {}).get("dataset_parquet", "dataset.parquet")
    dataset_csv = cfg.get("exports", {}).get("dataset_csv", "dataset.csv")

    # Simple cache: skip if files exist and not forcing
    if (Path(features_path).exists() and Path(labels_path).exists() and Path(dataset_parquet).exists() and not force):
        print("Using cached outputs (set --force 1 to recompute).")
        return features_path, labels_path, dataset_parquet, dataset_csv

    X = build_features(df, cfg)
    # Labels
    lab_cfg = cfg.get("labeling", {})
    Y = triple_barrier_labels(
        df,
        horizon_bars=int(lab_cfg.get("horizon_bars", 20)),
        tp_mult=float(lab_cfg.get("tp_mult", 2.0)),
        sl_mult=float(lab_cfg.get("sl_mult", 1.0)),
        use_atr=bool(lab_cfg.get("use_atr", True)),
        atr_len=int(lab_cfg.get("atr_len", 14)),
        zero_threshold=float(lab_cfg.get("zero_threshold", 0.0)),
        side=str(lab_cfg.get("side", "both")),
    )

    # Align & clean
    dataset = pd.concat([X, Y], axis=1).dropna()
    # Persist
    X.to_parquet(features_path)
    Y.to_parquet(labels_path)
    dataset.to_parquet(dataset_parquet)
    dataset.to_csv(dataset_csv, encoding="cp1252")

    return features_path, labels_path, dataset_parquet, dataset_csv

# ------------------------
# CLI
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input OHLCV CSV with datetime,open,high,low,close,volume")
    ap.add_argument("--config", default="feature_config.yaml")
    ap.add_argument("--out_prefix", default="step2")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--tz", default="Europe/Berlin")
    ap.add_argument("--force", type=int, default=0)
    args = ap.parse_args()

    paths = run_pipeline(args.csv, args.config, args.out_prefix, sep=args.sep, tz=args.tz, force=bool(args.force))
    print("Saved files:")
    for p in paths:
        print(" -", p)

if __name__ == "__main__":
    main()
