# -*- coding: cp1252 -*-
"""
test_train_pipeline.py
Smoke test for Step 3 training pipeline with synthetic dataset.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Build a tiny synthetic dataset that mimics Step 2 output
def make_dataset(n=3000, seed=7):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.001, n)
    price = 100 * np.exp(np.cumsum(ret))
    high = price * (1 + rng.uniform(0, 0.002, n))
    low  = price * (1 - rng.uniform(0, 0.002, n))
    open_ = price * (1 + rng.uniform(-0.0005, 0.0005, n))
    vol = rng.integers(1000, 2000, n)
    idx = pd.date_range("2021-01-01", periods=n, freq="T", tz="UTC")
    df = pd.DataFrame({"open":open_,"high":high,"low":low,"close":price,"volume":vol}, index=idx)
    # basic features
    df["ret_1"] = df["close"].pct_change(1).shift(1)
    df["roll_mean_5"] = df["close"].rolling(5).mean().shift(1)
    df["roll_std_5"] = df["close"].rolling(5).std().shift(1)
    df = df.dropna()

    # labels via simple future return horizon
    horizon = 20
    fut = df["close"].shift(-horizon) / df["close"] - 1.0
    y = np.where(fut > 0.001, 1, np.where(fut < -0.001, -1, 0))
    df["y"] = y
    df = df.dropna()
    return df

def run_smoke():
    from train_pipeline import load_config, train_walkforward, save_artifacts
    cfg = {
        "target":{"column":"y","classes":[-1,0,1]},
        "features":{"include":["ret_1","roll_mean_5","roll_std_5"], "exclude":["y"]},
        "cv":{"train_bars":1500,"test_bars":200,"step_bars":200,"purge_gap":10,"calibrate_frac":0.2},
        "models":[
            {"name":"RandomForest","type":"rf","params":{"n_estimators":[100],"max_depth":[6],"min_samples_leaf":[2],"class_weight":["balanced"],"random_state":[42]}},
            {"name":"LogisticRegression","type":"lr","params":{"C":[1.0],"penalty":["l2"],"solver":["lbfgs"],"max_iter":[200]}},
        ],
        "calibration":{"method":"sigmoid"},
        "conformal":{"alpha":0.2,"size_floor":0.0,"size_cap":1.0},
        "exports":{"out_dir":"step3_artifacts_test","cv_report_csv":"cv.csv","fold_predictions_parquet":"preds.parquet","threshold_json":"theta.json","final_metrics_json":"final.json"}
    }
    df = make_dataset()
    preds, reports = train_walkforward(df, cfg)
    outdir = save_artifacts(preds, reports, cfg)
    assert Path(outdir).exists()
    print("OK - pipeline ran and artifacts saved to", outdir)

if __name__ == "__main__":
    run_smoke()
