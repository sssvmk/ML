"""Drift monitoring: population stability index (PSI) of incoming data vs the training reference."""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    e, a = np.clip(np.asarray(expected, float), eps, None), np.clip(np.asarray(actual, float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(fe_df: pd.DataFrame, reference: dict, warn: float = 0.1, alert: float = 0.25) -> dict:
    """PSI per feature. Rule of thumb: <0.1 stable, 0.1-0.25 moderate shift, >0.25 major shift."""
    rows = {}
    for col, ref in reference["numeric"].items():
        if col in fe_df:
            v = pd.to_numeric(fe_df[col], errors="coerce").dropna()
            if len(v):
                props = np.histogram(v, bins=np.array(ref["edges"]))[0] / len(v)
                rows[col] = psi(np.array(ref["proportions"]), props)
    for col, ref in reference["categorical"].items():
        if col in fe_df:
            cur = fe_df[col].astype(str).value_counts(normalize=True)
            keys = sorted(set(ref) | set(cur.index))
            rows[col] = psi(np.array([ref.get(k, 0.0) for k in keys]), np.array([cur.get(k, 0.0) for k in keys]))
    status = {k: ("alert" if v > alert else "warn" if v > warn else "ok") for k, v in rows.items()}
    return {"psi": dict(sorted(rows.items(), key=lambda kv: -kv[1])), "status": status,
            "n_alert": sum(s == "alert" for s in status.values()), "n_warn": sum(s == "warn" for s in status.values()),
            "thresholds": {"warn": warn, "alert": alert}}
