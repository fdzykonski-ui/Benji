# -*- coding: cp1252 -*-
"""
portfolio_backtester.py
Step 4: Multi-asset portfolio backtester with policy layer.

Assumptions (Step 4 scope):
- All assets share the same timestamp grid (same timeframe, aligned rows).
- OHLCV columns: open, high, low, close, volume; DatetimeIndex (tz-aware preferred).
- Signals:
    * mode "fixed": pick one of top5 signal columns (computed internally)
    * mode "router": ADX-based regime and routing (internal, Step 1 logic)

Policy:
- Risk-based sizing: allocate total risk_per_trade * equity per bar across all new entries,
  proportionally to 'acceptance weight' (1.0 by default; conformal size if enabled).
- ATR-based initial & trailing stop per asset.
- Exposure guard: max positions & max gross exposure.
- Risk guards: max drawdown cap, daily loss limit; optional hard flatten.

Outputs:
- portfolio_equity.csv (equity MTM & exposure)
- portfolio_trades.csv (trade blotter)
- portfolio_diag.csv (per-bar diagnostics)

Usage:
  python portfolio_backtester.py --config portfolio_config.yaml --assets_json assets.json

where assets.json is optional; if provided, overrides `assets:` in YAML with a list of
{"symbol": "...", "csv": "...", "sep": ","}.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from top5_models import compute_top5, atr, adx

# -----------------------
# Router (minimal, inline)
# -----------------------
def detect_regime(df: pd.DataFrame, adx_len: int, th_trend_on: float, th_range_on: float, keep_mid: float) -> pd.Series:
    adxd = adx(df, adx_len)
    a = adxd[f"adx_{adx_len}"]
    regime = pd.Series(index=df.index, dtype='object')
    prev = "range"
    for i in range(len(a)):
        ai = a.iloc[i]
        if np.isnan(ai):
            regime.iloc[i] = prev
            continue
        if ai >= th_trend_on:
            r = "trend"
        elif ai <= th_range_on:
            r = "range"
        else:
            if prev == "trend" and ai > keep_mid:
                r = "trend"
            elif prev == "range" and ai < keep_mid:
                r = "range"
            else:
                r = prev
        regime.iloc[i] = r
        prev = r
    regime.name = "regime"
    return regime

def load_router_cfg(path: str) -> dict:
    txt = Path(path).read_text(encoding="cp1252")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(txt)
    except Exception:
        return json.loads(txt)

def build_signals(df: pd.DataFrame, mode: str, fixed_signal_col: str, router_cfg_path: str) -> pd.DataFrame:
    ind = compute_top5(df)
    base = pd.concat([df, ind], axis=1)

    if mode == "fixed":
        if fixed_signal_col not in base.columns:
            raise ValueError(f"Fixed signal column {fixed_signal_col} not found in computed signals.")
        base["active_signal"] = base[fixed_signal_col].fillna(0).astype(int)
        base["active_model"] = fixed_signal_col
        base["regime"] = "fixed"
        return base

    cfg = load_router_cfg(router_cfg_path)
    rd = cfg.get("regime_detection", {})
    adx_len = int(rd.get("adx_len", 14))
    th = rd.get("thresholds", {})
    tr_on = float(th.get("trend_on", 22))
    rg_on = float(th.get("range_on", 18))
    keep = float(rd.get("hysteresis", {}).get("keep_below", 20))

    routing = cfg.get("routing", {})
    mode_r = routing.get("mode", "first_nonzero")
    regimes = routing.get("regimes", {})

    regime = detect_regime(base, adx_len, tr_on, rg_on, keep)

    active_model = pd.Series(index=base.index, dtype='object')
    active_signal = pd.Series(0, index=base.index, dtype=int)

    for i, r in enumerate(regime):
        rconf = regimes.get(r, {"enabled": False, "candidates": []})
        enabled = bool(rconf.get("enabled", False))
        cands = list(rconf.get("candidates", []))
        if not enabled or not cands:
            active_model.iloc[i] = "none"
            active_signal.iloc[i] = 0
            continue
        row = base.iloc[i]
        if mode_r == "first_nonzero":
            picked = "none"; sig = 0
            for c in cands:
                v = int(np.sign(row.get(c, 0))) if not pd.isna(row.get(c, np.nan)) else 0
                if v != 0:
                    picked = c; sig = v; break
        else:  # majority
            vals = [int(np.sign(row.get(c, 0))) if not pd.isna(row.get(c, np.nan)) else 0 for c in cands]
            s = int(np.sign(np.sum(vals))); sig = s
            picked = "vote(" + ",".join(cands) + ")"
        active_model.iloc[i] = picked
        active_signal.iloc[i] = int(sig)

    base["regime"] = regime
    base["active_model"] = active_model
    base["active_signal"] = active_signal
    return base

# -----------------------
# Conformal threshold (optional sizing)
# -----------------------
def load_theta(artifacts_dir: str) -> float:
    p = Path(artifacts_dir) / "conformal_threshold.json"
    if not p.exists():
        return 0.5
    try:
        data = json.loads(p.read_text(encoding="cp1252"))
        return float(data.get("theta", 0.5))
    except Exception:
        return 0.5

# -----------------------
# Core portfolio backtest
# -----------------------
@dataclass
class Position:
    side: int      # +1 long, -1 short
    qty: float
    entry: float
    stop: float
    peak: float
    trough: float

def portfolio_backtest(assets: Dict[str, pd.DataFrame], cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # --- settings ---
    sz = cfg.get("sizing", {})
    risk_per_trade = float(sz.get("risk_per_trade", 0.01))
    use_conf = bool(sz.get("use_conformal", False))
    size_floor = float(sz.get("size_floor", 0.0))
    size_cap = float(sz.get("size_cap", 1.0))
    theta = load_theta(sz.get("conformal_artifacts_dir", "step3_artifacts")) if use_conf else 0.0

    ex = cfg.get("execution", {})
    fee = float(ex.get("fee", 0.0005))
    slip = float(ex.get("slippage", 0.0002))
    allow_short = bool(ex.get("allow_short", True))

    stp = cfg.get("stops", {})
    atr_len = int(stp.get("atr_len", 14))
    atr_mult = float(stp.get("atr_mult", 2.0))
    trailing = bool(stp.get("trailing", True))

    lim = cfg.get("portfolio_limits", {})
    max_positions = int(lim.get("max_positions", 5))
    max_gross = float(lim.get("max_gross_exposure", 1.5))
    max_dd = float(lim.get("max_drawdown", 0.2))
    daily_loss = float(lim.get("daily_loss_limit", 0.05))
    hard_stop = bool(lim.get("hard_stop_on_maxdd", False))

    # Align indices (intersection)
    indices = [df.index for df in assets.values()]
    common_index = indices[0]
    for idx in indices[1:]:
        common_index = common_index.intersection(idx)
    if len(common_index) < 10:
        raise ValueError("Not enough overlapping timestamps across assets. Ensure same timeframe and alignment.")
    # Reindex
    for k in assets:
        assets[k] = assets[k].loc[common_index]

    symbols = list(assets.keys())

    # Precompute ATR and signals for each asset
    atrs = {}
    signals = {}
    act_models = {}
    regimes = {}

    sig_cfg = cfg.get("signals", {})
    mode = sig_cfg.get("mode", "router")
    fixed_col = sig_cfg.get("fixed_signal_col", "signal_supertrend")
    router_cfg = sig_cfg.get("router_config", "router_config.yaml")

    for sym in symbols:
        df = assets[sym]
        atrs[sym] = atr(df, atr_len)
        sig_df = build_signals(df, mode, fixed_col, router_cfg)
        signals[sym] = sig_df["active_signal"].fillna(0).astype(int)
        act_models[sym] = sig_df["active_model"]
        regimes[sym] = sig_df["regime"]

    # State
    equity_realized = 10000.0  # start equity
    cash = equity_realized
    positions: Dict[str, Position] = {}
    last_day_equity_close = equity_realized

    trades = []
    diagnostics = []

    # Iterate over bars (next-bar execution based on previous bar's signal)
    for i in range(1, len(common_index)):
        ts_prev = common_index[i-1]
        ts = common_index[i]

        # --- MTM equity before decisions ---
        mtm_pnl = 0.0
        for sym, pos in positions.items():
            c = assets[sym]["close"].iloc[i-1]
            if pos.side > 0:
                mtm_pnl += (c - pos.entry) * pos.qty
            else:
                mtm_pnl += (pos.entry - c) * pos.qty
        equity_mtm = equity_realized + mtm_pnl

        # Risk guards: daily loss / max drawdown
        # We define "day" by calendar day on ts
        day_change = 0.0
        if ts_prev.date() != ts.date():
            last_day_equity_close = equity_mtm  # carry fwd previous day's close for next day's ref
        else:
            if last_day_equity_close != 0:
                day_change = (equity_mtm - last_day_equity_close) / last_day_equity_close

        # For drawdown, we can track peak on the fly
        if i == 1:
            eq_peak = equity_mtm
        else:
            eq_peak = max(eq_peak, equity_mtm)
        dd = (equity_mtm / eq_peak - 1.0) if eq_peak > 0 else 0.0
        can_open_new = True
        breach_reason = ""
        if dd <= -max_dd:
            can_open_new = False
            breach_reason = "max_dd"
            if hard_stop and positions:
                # flatten all at next open
                for sym in list(positions.keys()):
                    o = assets[sym]["open"].iloc[i]  # current open
                    pos = positions[sym]
                    if pos.side > 0:
                        exit_price = o * (1 - slip)
                        pnl = (exit_price - pos.entry) * pos.qty
                    else:
                        exit_price = o * (1 + slip)
                        pnl = (pos.entry - exit_price) * pos.qty
                    cost = fee * exit_price * pos.qty
                    equity_realized += pnl - cost
                    trades.append((ts, sym, "FLATTEN_DD", exit_price, pos.qty, pnl - cost, equity_realized))
                    del positions[sym]
        if abs(day_change) >= daily_loss and day_change < 0:
            can_open_new = False
            breach_reason = "daily_loss"

        # --- Manage existing positions: trailing stops intrabar ---
        for sym, pos in list(positions.items()):
            df = assets[sym]
            h = df["high"].iloc[i]
            l = df["low"].iloc[i]
            c_prev = df["close"].iloc[i-1]
            atr_i = atrs[sym].iloc[i-1]
            if pos.side > 0:
                pos.peak = max(pos.peak, h)
                if trailing:
                    trail = pos.peak - atr_mult * atr_i
                    pos.stop = max(pos.stop, trail)
                if l <= pos.stop:  # stop hit
                    exit_price = pos.stop * (1 - slip)
                    pnl = (exit_price - pos.entry) * pos.qty
                    cost = fee * exit_price * pos.qty
                    equity_realized += pnl - cost
                    trades.append((ts, sym, "STOP_LONG", exit_price, pos.qty, pnl - cost, equity_realized))
                    del positions[sym]
            else:
                pos.trough = min(pos.trough, l) if np.isfinite(pos.trough) else l
                if trailing:
                    trail = pos.trough + atr_mult * atr_i
                    pos.stop = min(pos.stop, trail)
                if h >= pos.stop:
                    exit_price = pos.stop * (1 + slip)
                    pnl = (pos.entry - exit_price) * pos.qty
                    cost = fee * exit_price * pos.qty
                    equity_realized += pnl - cost
                    trades.append((ts, sym, "STOP_SHORT", exit_price, pos.qty, pnl - cost, equity_realized))
                    del positions[sym]

        # --- Desired positions from signals (based on ts_prev) ---
        desired = {}
        for sym in symbols:
            s_prev = signals[sym].iloc[i-1]
            if s_prev > 0:
                desired[sym] = 1
            elif s_prev < 0 and allow_short:
                desired[sym] = -1
            else:
                desired[sym] = 0

        # --- Close positions where desired!=current ---
        for sym in list(positions.keys()):
            want = desired.get(sym, 0)
            cur = positions[sym].side
            if want == cur:
                continue
            # Close at open
            o = assets[sym]["open"].iloc[i]
            pos = positions[sym]
            if cur > 0:
                exit_price = o * (1 - slip)
                pnl = (exit_price - pos.entry) * pos.qty
                action = "EXIT_LONG"
            else:
                exit_price = o * (1 + slip)
                pnl = (pos.entry - exit_price) * pos.qty
                action = "EXIT_SHORT"
            cost = fee * exit_price * pos.qty
            equity_realized += pnl - cost
            trades.append((ts, sym, action, exit_price, pos.qty, pnl - cost, equity_realized))
            del positions[sym]

        # --- Open new positions where desired!=0 and not already in ---
        entering = [sym for sym in symbols if desired[sym] != 0 and sym not in positions]
        # Risk guard gates
        if not can_open_new:
            entering = []

        # Exposure constraints: max positions, gross exposure
        if len(positions) >= max_positions:
            entering = []
        else:
            entering = entering[:max(0, max_positions - len(positions))]

        # Compute allocation weights (acceptance)
        weights = {}
        sum_w = 0.0
        for sym in entering:
            # conformal acceptance proxy: if enabled, we use theta against a dummy prob=1.0 (always accept)
            # In Step 4 baseline we don't have per-asset live probs; size weight defaults to 1.0.
            w = 1.0
            weights[sym] = w
            sum_w += w

        total_risk_dollars = equity_mtm * risk_per_trade
        for sym in entering:
            # Exposure cap check (approx using MTM post-entry)
            price = assets[sym]["open"].iloc[i]
            atr_i = atrs[sym].iloc[i-1]
            if not np.isfinite(atr_i) or atr_i <= 0:
                continue
            stop_dist = atr_mult * atr_i
            # allocate risk
            share = weights[sym] / max(sum_w, 1e-9)
            dollar_risk = total_risk_dollars * share
            qty = max(dollar_risk / stop_dist / max(price, 1e-12), 0.0)
            if qty <= 0:
                continue
            # Check gross exposure
            notional_after = sum(abs(positions[s].qty) * assets[s]["close"].iloc[i-1] for s in positions) + qty * price
            exposure = notional_after / max(equity_mtm, 1e-9)
            if exposure > max_gross:
                continue

            # Enter
            side = desired[sym]
            if side > 0:
                entry_price = price * (1 + slip)
                stop = entry_price - stop_dist
                peak = entry_price
                trough = np.nan
                cost = fee * entry_price * qty
                equity_realized -= cost
                trades.append((ts, sym, "ENTER_LONG", entry_price, qty, -cost, equity_realized))
            else:
                entry_price = price * (1 - slip)
                stop = entry_price + stop_dist
                peak = np.nan
                trough = entry_price
                cost = fee * entry_price * qty
                equity_realized -= cost
                trades.append((ts, sym, "ENTER_SHORT", entry_price, qty, -cost, equity_realized))

            positions[sym] = Position(side=side, qty=qty, entry=entry_price, stop=stop, peak=peak, trough=trough)

        # --- Diagnostics & equity curve (MTM using current close) ---
        mtm_pnl = 0.0
        gross_notional = 0.0
        for sym, pos in positions.items():
            c = assets[sym]["close"].iloc[i]
            gross_notional += abs(pos.qty * c)
            if pos.side > 0:
                mtm_pnl += (c - pos.entry) * pos.qty
            else:
                mtm_pnl += (pos.entry - c) * pos.qty
        equity_curve = equity_realized + mtm_pnl
        gross_exposure = gross_notional / max(equity_curve, 1e-9)
        diagnostics.append((ts, equity_curve, equity_realized, gross_exposure, len(positions), breach_reason))

    # Build outputs
    equity_df = pd.DataFrame(diagnostics, columns=["datetime","equity_mtm","equity_realized","gross_exposure","open_positions","breach"]).set_index("datetime")
    trades_df = pd.DataFrame(trades, columns=["datetime","symbol","action","price","qty","pnl","equity_realized"]).set_index("datetime")

    # Metrics per portfolio
    eq = equity_df["equity_mtm"]
    rets = eq.pct_change().fillna(0.0)
    freq = 252  # assume daily-equivalent bars; adapt as needed
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (freq / max(len(eq),1)) - 1 if eq.iloc[0] > 0 else 0.0
    sharpe = (rets.mean() / (rets.std() + 1e-12)) * np.sqrt(freq) if rets.std() > 0 else 0.0
    roll_max = eq.cummax()
    maxdd = (eq / roll_max - 1.0).min()
    diag = pd.DataFrame({"cagr":[cagr], "sharpe":[sharpe], "max_dd":[maxdd], "final_equity":[eq.iloc[-1]]}, index=["portfolio"])

    return equity_df, trades_df, diag

# -----------------------
# CLI
# -----------------------
def load_config(path: str) -> dict:
    txt = Path(path).read_text(encoding="cp1252")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(txt)
    except Exception:
        return json.loads(txt)

def load_assets(specs: List[dict], tz: str = "Europe/Berlin") -> Dict[str, pd.DataFrame]:
    out = {}
    for spec in specs:
        sym = spec["symbol"]
        csv = spec["csv"]
        sep = spec.get("sep", ",")
        df = pd.read_csv(csv, sep=sep)
        # datetime
        dt_col = None
        for cand in ["datetime","date","timestamp","time"]:
            if cand in df.columns:
                dt_col = cand; break
        if dt_col is None:
            raise ValueError(f"{csv} missing datetime column.")
        df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
        df = df.set_index(dt_col).sort_index()
        if tz:
            df = df.tz_convert(tz)
        df = df.rename(columns={c:c.lower() for c in df.columns})
        req = ["open","high","low","close","volume"]
        miss = [c for c in req if c not in df.columns]
        if miss:
            raise ValueError(f"{csv} missing columns: {miss}")
        out[sym] = df
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="portfolio_config.yaml")
    ap.add_argument("--assets_json", default=None, help="Optional JSON file to override assets list")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tz = cfg.get("meta", {}).get("timezone", "Europe/Berlin")
    assets_spec = cfg.get("assets", [])
    if args.assets_json:
        assets_spec = json.loads(Path(args.assets_json).read_text(encoding="cp1252"))

    if not assets_spec:
        raise SystemExit("No assets provided. Edit portfolio_config.yaml or pass --assets_json.")

    assets = load_assets(assets_spec, tz=tz)

    equity_df, trades_df, diag = portfolio_backtest(assets, cfg)

    exp = cfg.get("exports", {})
    eq_path = exp.get("equity_csv", "portfolio_equity.csv")
    tr_path = exp.get("trades_csv", "portfolio_trades.csv")
    dg_path = exp.get("diagnostics_csv", "portfolio_diag.csv")

    equity_df.to_csv(eq_path, encoding="cp1252")
    trades_df.to_csv(tr_path, encoding="cp1252")
    diag.to_csv(dg_path, encoding="cp1252")
    print("Saved:", eq_path, tr_path, dg_path)

if __name__ == "__main__":
    main()
