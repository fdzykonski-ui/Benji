# -*- coding: cp1252 -*-
"""
predict_live.py
Utility to load the trained/calibrated model and apply conformal sizing to new feature rows.
Assumes artifacts saved by train_pipeline.py (Step 3).
"""
from __future__ import annotations
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

def load_artifacts(out_dir: str = "step3_artifacts"):
    out = Path(out_dir)
    # In Step 3 we didn't persist the model pipe; here we load threshold only as a placeholder.
    theta = json.loads((out / "conformal_threshold.json").read_text(encoding="cp1252"))["theta"]
    # A real deployment would also persist the preprocessor and calibrated model.
    # For now, we return just the threshold to plug into a sizing rule.
    return {"theta": float(theta)}

def size_from_probs(proba: np.ndarray, theta: float, size_floor: float = 0.0, size_cap: float = 1.0):
    max_prob = proba.max(axis=1)
    accept = max_prob >= theta
    rel_size = np.zeros(len(proba), dtype=float)
    rel_size[accept] = size_floor + (max_prob[accept] - theta) / max(1e-9, (1 - theta)) * (size_cap - size_floor)
    rel_size = np.clip(rel_size, size_floor, size_cap)
    return accept.astype(int), rel_size
