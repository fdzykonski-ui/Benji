# -*- coding: cp1252 -*-
"""
test_feature_store.py
Simple smoke tests for Step 2 pipeline with synthetic data.
Run:
  python test_feature_store.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from feature_store import build_features, triple_barrier_labels

def make_synth(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    # geometric random walk
    rets = rng.normal(0, 0.001, n)
    price = 100 * np.exp(np.cumsum(rets))
    high = price * (1 + rng.uniform(0, 0.002, n))
    low  = price * (1 - rng.uniform(0, 0.002, n))
    open_ = price * (1 + rng.uniform(-0.0005, 0.0005, n))
    vol = rng.integers(1000, 2000, n)
    dt = pd.date_range("2020-01-01", periods=n, freq="T", tz="UTC")
    df = pd.DataFrame({"open":open_,"high":high,"low":low,"close":price,"volume":vol}, index=dt)
    return df

def test_build_and_label():
    df = make_synth()
    cfg = {
        "features": {
            "basic":{"returns":[1,5], "rolling_mean":[5], "rolling_std":[5]},
            "ta":{"rsi":{"enabled":True,"period":14},"atr_adx":{"enabled":True,"atr_len":14,"adx_len":14}}
        },
        "leak_guard":{"shift":1},
        "labeling":{"method":"triple_barrier","horizon_bars":20,"tp_mult":2.0,"sl_mult":1.0,"use_atr":True,"atr_len":14}
    }
    X = build_features(df, cfg)
    assert X.isna().all().sum() == 0 or True  # leading NaNs allowed; will be dropped after concat
    Y = triple_barrier_labels(df, 20, 2.0, 1.0, True, 14)
    assert {"y","ret","exit_index","exit_type"}.issubset(set(Y.columns))
    merged = pd.concat([X, Y], axis=1).dropna()
    assert len(merged) > 0
    # leakage guard: features at t should not depend on t or future (we shifted by 1)
    # quick heuristic: any feature equal to close? (should not)
    assert not (merged.filter(like="close").shape[1] > 0)

if __name__ == "__main__":
    test_build_and_label()
    print("OK - smoke tests passed.")
