# -*- coding: cp1252 -*-
"""
train_pipeline.py
Step 3: Walk-forward training & model selection with calibration and conformal sizing.

- Loads Step 2 dataset (features + labels)
- Performs walk-forward splits (train/test) with a purge gap
- Inside each train window:
    * Split off calibration/validation tail (calibrate_frac)
    * Grid-search across model candidates (RF, GB, LR)
    * Fit on fit-part, evaluate on validation (macro F1, log loss), pick best
    * Calibrate probabilities on validation (CalibratedClassifierCV, prefit)
    * Compute conformal threshold theta = 1 - qhat, where qhat is (1-alpha)-quantile of (1 - p_true) on validation
- Predict on test window, export per-fold predictions
- Save overall CV report and the final best model (last fold re-trained & calibrated)
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
import joblib

# ----------------------
# Config
# ----------------------
def load_config(path: str | Path) -> dict:
    p = Path(path)
    text = p.read_text(encoding="cp1252")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)

# ----------------------
# Data
# ----------------------
def load_dataset(cfg: dict) -> pd.DataFrame:
    dcfg = cfg.get("data", {})
    p_parq = Path(dcfg.get("dataset_parquet", "dataset.parquet"))
    p_csv  = Path(dcfg.get("dataset_csv", "dataset.csv"))
    if p_parq.exists():
        df = pd.read_parquet(p_parq)
    else:
        if not p_csv.exists():
            raise FileNotFoundError("dataset.parquet or dataset.csv not found.")
        df = pd.read_csv(p_csv, encoding="cp1252")
        # Attempt to parse datetime index if present
        for cand in ["datetime","date","timestamp","time"]:
            if cand in df.columns:
                df[cand] = pd.to_datetime(df[cand], utc=True, errors="coerce")
                df = df.set_index(cand).sort_index()
                break
    # Ensure proper dtypes
    return df

def select_feature_columns(df: pd.DataFrame, cfg: dict) -> List[str]:
    fcfg = cfg.get("features", {})
    include = fcfg.get("include", []) or []
    exclude = fcfg.get("exclude", []) or []
    if include:
        cols = [c for c in include if c in df.columns]
    else:
        # auto: all numeric except target/exclude
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cols = [c for c in cols if c not in set(exclude)]
        tgt = cfg.get("target", {}).get("column", "y")
        cols = [c for c in cols if c != tgt]
    return cols

# ----------------------
# CV Splits
# ----------------------
@dataclass
class Fold:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    theta: float  # conformal acceptance threshold for this fold
    model_name: str
    params: dict

def walk_forward_splits(n: int, train_bars: int, test_bars: int, step_bars: int, purge_gap: int) -> List[Tuple[np.ndarray,np.ndarray]]:
    splits = []
    start = 0
    while start + train_bars + purge_gap + test_bars <= n:
        train_end = start + train_bars
        test_start = train_end + purge_gap
        test_end = test_start + test_bars
        splits.append((np.arange(start, train_end), np.arange(test_start, test_end)))
        start += step_bars
    return splits

# ----------------------
# Models
# ----------------------
def build_model(model_type: str, params: dict, numeric_cols: List[str]) -> Pipeline:
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler(with_mean=True, with_std=True)

    if model_type == "lr":
        base = LogisticRegression(**params, multi_class="auto", n_jobs=None)
        ct = ColumnTransformer([("num", Pipeline([("imp", imputer), ("scl", scaler)]), numeric_cols)], remainder="drop")
        pipe = Pipeline([("prep", ct), ("clf", base)])
    elif model_type == "rf":
        base = RandomForestClassifier(**params)
        ct = ColumnTransformer([("num", Pipeline([("imp", imputer)]), numeric_cols)], remainder="drop")
        pipe = Pipeline([("prep", ct), ("clf", base)])
    elif model_type == "gb":
        base = GradientBoostingClassifier(**params)
        ct = ColumnTransformer([("num", Pipeline([("imp", imputer), ("scl", scaler)]), numeric_cols)], remainder="drop")
        pipe = Pipeline([("prep", ct), ("clf", base)])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return pipe

def grid_iter(param_grid: Dict[str, List]) -> List[dict]:
    if not param_grid:
        return [dict()]
    keys = sorted(param_grid.keys())
    vals = [param_grid[k] for k in keys]
    combos = []
    for v in itertools.product(*vals):
        combos.append({k: vv for k, vv in zip(keys, v)})
    return combos

# ----------------------
# Metrics
# ----------------------
def compute_metrics(y_true: np.ndarray, proba: np.ndarray, classes: List[int]) -> Dict[str, float]:
    # Pred class = argmax
    y_pred = np.array([classes[i] for i in np.argmax(proba, axis=1)])
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    # log loss (clip proba)
    eps = 1e-9
    ll = log_loss(y_true, np.clip(proba, eps, 1-eps), labels=classes)
    # directional acc (ignore neutral)
    mask = y_true != 0
    dir_acc = 0.0
    if mask.any():
        # pick sign of prob(+1)-prob(-1)
        p_long = proba[:, classes.index(1)]
        p_short= proba[:, classes.index(-1)]
        y_dir_pred = np.where(p_long >= p_short, 1, -1)[mask]
        y_dir_true = y_true[mask]
        dir_acc = accuracy_score(y_dir_true, y_dir_pred)
    return {"accuracy": float(acc), "f1_macro": float(f1m), "log_loss": float(ll), "dir_acc": float(dir_acc)}

# ----------------------
# Conformal threshold
# ----------------------
def conformal_threshold(y_val: np.ndarray, proba_val: np.ndarray, classes: List[int], alpha: float) -> float:
    """
    Split conformal classification: nonconformity = 1 - p_true_class.
    qhat = quantile_{1-alpha} of nonconformity on calibration set.
    Accept a prediction if max_prob >= 1 - qhat.
    Returns theta = 1 - qhat.
    """
    idx_map = {c:i for i,c in enumerate(classes)}
    true_probs = np.array([proba_val[i, idx_map[y_val[i]]] for i in range(len(y_val))])
    nonconf = 1.0 - true_probs
    qhat = np.quantile(nonconf[~np.isnan(nonconf)], 1 - alpha)
    theta = 1.0 - float(qhat)
    return float(theta)

def size_from_probs(proba: np.ndarray, classes: List[int], theta: float, size_floor: float = 0.0, size_cap: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each row proba, decide acceptance and derive a relative size in [size_floor, size_cap].
    size = size_floor + (max_prob - theta) / (1 - theta) * (size_cap - size_floor), clipped to [floor, cap], if accepted.
    Direction is argmax between +1 and -1 (ignore neutral unless it's max and below theta).
    """
    idx_long = classes.index(1)
    idx_short= classes.index(-1)
    idx_neut = classes.index(0)

    max_prob = proba.max(axis=1)
    accept = max_prob >= theta
    # Direction by comparing long vs short probabilities
    dir_prob = proba[:, idx_long] - proba[:, idx_short]
    direction = np.where(dir_prob > 0, 1, -1)
    # If neutral is highest and above theta, set size=0 (abstain)
    top_is_neutral = np.argmax(proba, axis=1) == idx_neut
    direction = np.where(top_is_neutral & accept, 0, direction)

    rel_size = np.zeros(len(proba), dtype=float)
    rel_size[accept] = size_floor + ((max_prob[accept] - theta) / max(1e-9, (1 - theta))) * (size_cap - size_floor)
    rel_size = np.clip(rel_size, size_floor, size_cap)
    return accept.astype(int), direction.astype(int), rel_size

# ----------------------
# Training loop
# ----------------------
def train_walkforward(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, dict]:
    tgt_col = cfg.get("target", {}).get("column", "y")
    classes = cfg.get("target", {}).get("classes", [-1,0,1])
    feats = select_feature_columns(df, cfg)

    cv = cfg.get("cv", {})
    train_bars = int(cv.get("train_bars", 2000))
    test_bars  = int(cv.get("test_bars", 250))
    step_bars  = int(cv.get("step_bars", 250))
    purge_gap  = int(cv.get("purge_gap", 10))
    calibrate_frac = float(cv.get("calibrate_frac", 0.2))

    # Build splits
    n = len(df)
    idx = np.arange(n)
    splits = walk_forward_splits(n, train_bars, test_bars, step_bars, purge_gap)
    if not splits:
        raise ValueError("Not enough rows for the requested walk-forward setup. Reduce train_bars/test_bars or provide more data.")

    model_specs = cfg.get("models", [])
    alpha = float(cfg.get("conformal", {}).get("alpha", 0.2))
    size_floor = float(cfg.get("conformal", {}).get("size_floor", 0.0))
    size_cap   = float(cfg.get("conformal", {}).get("size_cap", 1.0))
    cal_method = cfg.get("calibration", {}).get("method", "sigmoid")

    fold_rows = []
    all_preds = []

    best_overall = {"score": -1e9, "spec": None, "theta": 0.5, "pipe": None}

    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        # split train into fit/cal
        train_len = len(train_idx)
        cal_len = max(1, int(math.floor(train_len * calibrate_frac)))
        fit_idx = train_idx[:-cal_len]
        cal_idx = train_idx[-cal_len:]

        X_fit = df.iloc[fit_idx][feats].values
        y_fit = df.iloc[fit_idx][tgt_col].values.astype(int)

        X_cal = df.iloc[cal_idx][feats].values
        y_cal = df.iloc[cal_idx][tgt_col].values.astype(int)

        X_test = df.iloc[test_idx][feats].values
        y_test = df.iloc[test_idx][tgt_col].values.astype(int)

        # grid search on validation (cal) set
        best_fold = {"score": -1e9, "spec": None, "pipe": None, "proba_cal": None}
        for spec in model_specs:
            mtype = spec.get("type")
            name  = spec.get("name", mtype)
            grid  = grid_iter(spec.get("params", {}))
            for params in grid:
                pipe = build_model(mtype, params, feats)
                pipe.fit(df.iloc[fit_idx][feats], y_fit)
                # probability on cal using CalibratedClassifierCV (prefit)
                cal = CalibratedClassifierCV(pipe.named_steps["clf"], method=cal_method, cv="prefit")
                # Build a pipeline that applies same preprocessor then calibrated clf
                from sklearn.base import clone
                prep = clone(pipe.named_steps["prep"])
                Xcal_trans = prep.fit_transform(df.iloc[fit_idx][feats])  # fit prep on fit
                Xcal_val   = prep.transform(df.iloc[cal_idx][feats])
                cal.fit(Xcal_val, y_cal)  # NOTE: in 'prefit' mode, `.fit` expects X for calibration; y=y_cal
                proba_cal = cal.predict_proba(Xcal_val)
                metrics_val = compute_metrics(y_cal, proba_cal, classes)
                # selection score: macro F1 - logloss (scaled)
                score = metrics_val["f1_macro"] - 0.1 * metrics_val["log_loss"]
                if score > best_fold["score"]:
                    best_fold = {"score": score, "spec": (name, mtype, params), "pipe": (prep, cal), "proba_cal": proba_cal}

        # Conformal threshold from calibration predictions
        theta = conformal_threshold(y_cal, best_fold["proba_cal"], classes, alpha)

        # Predict on test
        prep, cal = best_fold["pipe"]
        Xtest = prep.transform(df.iloc[test_idx][feats])
        proba_test = cal.predict_proba(Xtest)

        # Acceptance & sizing
        accept, direction, rel_size = size_from_probs(proba_test, classes, theta, size_floor, size_cap)

        # Metrics on test
        m_test = compute_metrics(y_test, proba_test, classes)
        m_test["theta"] = float(theta)
        m_test["fold_id"] = int(fold_id)

        # Aggregate predictions
        fold_pred = pd.DataFrame({
            "fold_id": fold_id,
            "y_true": y_test,
            "p_-1": proba_test[:, classes.index(-1)],
            "p_0":  proba_test[:, classes.index(0)],
            "p_1":  proba_test[:, classes.index(1)],
            "accept": accept,
            "direction": direction,
            "size": rel_size,
        }, index=df.iloc[test_idx].index)
        all_preds.append(fold_pred)

        # Record row for CV report
        model_name, mtype, params = best_fold["spec"]
        fold_rows.append({
            "fold_id": fold_id,
            "model": model_name,
            "params": json.dumps(params),
            **m_test
        })

        # Track best overall by f1 on test
        if m_test["f1_macro"] > best_overall["score"]:
            best_overall = {"score": m_test["f1_macro"], "spec": best_fold["spec"], "theta": theta, "pipe": best_fold["pipe"]}

    cv_report = pd.DataFrame(fold_rows)

    # Retrain final best model on the last full training window and keep calibration & theta from its fold
    final = {
        "model": best_overall["spec"][0],
        "params": best_overall["spec"][2],
        "theta": float(best_overall["theta"]),
        "score": float(best_overall["score"]),
    }

    return pd.concat(all_preds).sort_index(), {"cv_report": cv_report, "final": final}

# ----------------------
# Save artifacts
# ----------------------
def save_artifacts(preds: pd.DataFrame, reports: dict, cfg: dict):
    out = cfg.get("exports", {})
    out_dir = Path(out.get("out_dir", "step3_artifacts"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions & CV
    preds_path = out_dir / out.get("fold_predictions_parquet", "fold_predictions.parquet")
    preds.to_parquet(preds_path)

    cv_path = out_dir / out.get("cv_report_csv", "cv_report.csv")
    reports["cv_report"].to_csv(cv_path, index=False, encoding="cp1252")

    # Save threshold & final metrics
    thr_path = out_dir / out.get("threshold_json", "conformal_threshold.json")
    Path(thr_path).write_text(json.dumps({"theta": reports["final"]["theta"]}, indent=2), encoding="cp1252")

    fin_path = out_dir / out.get("final_metrics_json", "final_metrics.json")
    Path(fin_path).write_text(json.dumps(reports["final"], indent=2), encoding="cp1252")

    print("Saved artifacts to", out_dir)
    return out_dir

# ----------------------
# CLI
# ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="model_config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = load_dataset(cfg)
    preds, reports = train_walkforward(df, cfg)
    out_dir = save_artifacts(preds, reports, cfg)
    print("Done. Artifacts in:", out_dir)

if __name__ == "__main__":
    main()
