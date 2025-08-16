# -*- coding: cp1252 -*-
"""
signal_router.py
Step 1: Model Router + unified SignalFrame + YAML config.

- Loads OHLCV CSV, computes Top-5 indicators/signals via top5_models.compute_top5()
- Detects market regime via ADX with hysteresis (configurable)
- Routes to a chosen model set per regime and produces:
    * active_model (str), active_signal (int: -1/0/1), regime (str)
    * plus all underlying signal_* columns for transparency

Usage:
    python signal_router.py --csv data.csv --config router_config.yaml --model_debug 0
Outputs:
    routed_signals.csv  (SignalFrame with regime + active model/signal)
    router_debug.csv    (ADX and regime internals)

Dependencies:
    - top5_models.py in the same folder or on PYTHONPATH
    - pyyaml (pip install pyyaml)  [falls nicht vorhanden: JSON-Fallback]
"""
from __future__ import annotations
import argparse
from typing import Dict, List
import pandas as pd
import numpy as np
from pathlib import Path

# Local imports
from top5_models import compute_top5, adx

# -----------------------
# Config loader (YAML or JSON fallback)
# -----------------------
def load_config(path: str | Path) -> dict:
    p = Path(path)
    txt = p.read_text(encoding="cp1252")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(txt)
    except Exception:
        import json
        return json.loads(txt)

# -----------------------
# IO helpers
# -----------------------
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

# -----------------------
# Regime detection (ADX + hysteresis)
# -----------------------
def detect_regime(df: pd.DataFrame, adx_len: int, th_trend_on: float, th_range_on: float, keep_mid: float) -> pd.Series:
    """
    Returns a Series of regime labels: 'trend' or 'range' (and 'breakout' placeholder later).
    Hysteresis: when ADX between th_range_on..th_trend_on, keep previous regime,
                unless no previous -> default 'range'.
    """
    adxd = adx(df, adx_len)
    adx_col = adxd[f"adx_{adx_len}"]
    regime = pd.Series(index=df.index, dtype='object')
    prev = "range"
    for i in range(len(adx_col)):
        a = adx_col.iloc[i]
        if np.isnan(a):
            regime.iloc[i] = prev
            continue
        if a >= th_trend_on:
            r = "trend"
        elif a <= th_range_on:
            r = "range"
        else:
            # hysteresis region: hold prev unless crosses keep_mid significantly
            if prev == "trend" and a > keep_mid:
                r = "trend"
            elif prev == "range" and a < keep_mid:
                r = "range"
            else:
                r = prev
        regime.iloc[i] = r
        prev = r
    regime.name = "regime"
    return regime, adxd

# -----------------------
# Routing
# -----------------------
def choose_signal_row(row: pd.Series, candidates: List[str], mode: str) -> int:
    vals = [int(np.sign(row.get(c, 0))) if not pd.isna(row.get(c, np.nan)) else 0 for c in candidates]
    if mode == "first_nonzero":
        for c, v in zip(candidates, vals):
            if v != 0:
                return v
        return 0
    elif mode == "majority_vote":
        s = np.sum(vals)
        return int(np.sign(s))
    else:
        return 0

def route_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # Compute base signals
    ind = compute_top5(df)
    base = pd.concat([df, ind], axis=1)

    # Config
    rd = cfg.get("regime_detection", {})
    adx_len = int(rd.get("adx_len", 14))
    th = rd.get("thresholds", {})
    th_trend_on = float(th.get("trend_on", 22))
    th_range_on = float(th.get("range_on", 18))
    hyster = rd.get("hysteresis", {})
    keep_mid = float(hyster.get("keep_below", 20))

    # Regime detection
    regime, adxd = detect_regime(base, adx_len, th_trend_on, th_range_on, keep_mid)

    routing = cfg.get("routing", {})
    mode = routing.get("mode", "first_nonzero")
    regimes = routing.get("regimes", {})

    # Prepare output
    out = base.copy()
    out["regime"] = regime

    active_model = pd.Series(index=out.index, dtype='object')
    active_signal = pd.Series(0, index=out.index, dtype=int)

    for ridx, r in enumerate(out["regime"]):
        rconf = regimes.get(r, {"enabled": False, "candidates": []})
        enabled = bool(rconf.get("enabled", False))
        cands = list(rconf.get("candidates", []))
        if not enabled or len(cands) == 0:
            active_model.iloc[ridx] = "none"
            active_signal.iloc[ridx] = 0
            continue
        row = out.iloc[ridx]
        sig = choose_signal_row(row, cands, mode)
        # record first candidate used (for transparency); if all 0, mark none
        picked = "none"
        if mode == "first_nonzero":
            for c in cands:
                v = int(np.sign(row.get(c, 0))) if not pd.isna(row.get(c, np.nan)) else 0
                if v != 0:
                    picked = c
                    break
        else:
            picked = "vote(" + ",".join(cands) + ")"
        active_model.iloc[ridx] = picked
        active_signal.iloc[ridx] = int(sig)

    out["active_model"] = active_model
    out["active_signal"] = active_signal

    # Debug sheet
    debug = pd.concat([adxd, regime.rename("regime")], axis=1)

    return out, debug

# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input OHLCV CSV with datetime,open,high,low,close,volume")
    ap.add_argument("--config", default="router_config.yaml")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--tz", default="Europe/Berlin")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tz = cfg.get("meta", {}).get("timezone", args.tz)

    df = load_df(args.csv, sep=args.sep, tz=tz)
    out, dbg = route_signals(df, cfg)

    # Exports
    exp = cfg.get("exports", {})
    routed_path = exp.get("routed_signals_csv", "routed_signals.csv")
    debug_path = exp.get("debug_sheet_csv", "router_debug.csv")

    out.to_csv(routed_path, encoding="cp1252")
    dbg.to_csv(debug_path, encoding="cp1252")
    print(f"Saved: {routed_path}, {debug_path}")
    print("Columns (head):")
    print(out.filter(items=['regime','active_model','active_signal']).head())

if __name__ == "__main__":
    main()
