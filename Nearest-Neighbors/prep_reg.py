"""
prep_reg.py - Zillow: temporal split, outlier handling and FULL feature engineering (the pool from which each method selects its top-k variables).
  zfeatures.py : zero-fill, missing indicators, derived ratios (age, tax/sqft, structure & land ratio, ...), lat/lon quadratics, log1p of skewed columns, winsorising,
                 one-hot of low/high-cardinality ids (rare levels pooled), standardisation - all fitted on TRAIN only
  + target encoding of the high-cardinality location / zoning codes (smoothed mean of the clipped target; out-of-fold for TRAIN rows)
Fit target = logerror clipped at the train 1st/99th percentile; every metric uses the raw logerror.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import zfeatures as zf

TE_COLS = ["regionidzip", "regionidcity", "regionidneighborhood", "propertyzoningdesc", "propertycountylandusecode", "tract"]


def build_prepared(csv_path, out_dir, cfg, logger=print):
    out = Path(out_dir)
    meta = zf.build_prepared(csv_path, out, cfg, logger=logger)
    names = list(meta["feature_names"])
    df = pd.read_csv(csv_path, low_memory=False)
    df["transactiondate"] = pd.to_datetime(df["transactiondate"], errors="coerce")
    df = df.dropna(subset=["transactiondate", "logerror"]).sort_values("transactiondate", kind="stable").reset_index(drop=True)
    masks = zf.split_masks(df["transactiondate"])
    parts = {k: df[m].reset_index(drop=True) for k, m in masks.items()}
    y_fit = np.load(out / "y_fit_train.npy")
    rng = np.random.RandomState(cfg.seed)
    fold = rng.randint(0, 5, len(y_fit))
    gm, m_smooth = float(y_fit.mean()), 30.0
    cols = {k: [] for k in parts}
    for c in TE_COLS:
        cats = {k: zf._cat(v, c).astype(str).values for k, v in parts.items()}
        tr = pd.Series(cats["train"])
        full = pd.DataFrame({"c": tr, "y": y_fit}).groupby("c")["y"].agg(["sum", "count"])
        te_tr = np.empty(len(y_fit))
        for f in range(5):
            m = fold == f
            g = pd.DataFrame({"c": tr[~m].values, "y": y_fit[~m]}).groupby("c")["y"].agg(["sum", "count"])
            s, n_ = tr[m].map(g["sum"]).fillna(0.0).values, tr[m].map(g["count"]).fillna(0.0).values
            te_tr[m] = (s + m_smooth * gm) / (n_ + m_smooth)
        cols["train"].append(te_tr)
        for k in ("val", "test"):
            s, n_ = pd.Series(cats[k]).map(full["sum"]).fillna(0.0).values, pd.Series(cats[k]).map(full["count"]).fillna(0.0).values
            cols[k].append((s + m_smooth * gm) / (n_ + m_smooth))
    mu = [a.mean() for a in cols["train"]]
    sd = [a.std() or 1.0 for a in cols["train"]]
    for k in parts:
        TE = np.column_stack([(cols[k][j] - mu[j]) / sd[j] for j in range(len(TE_COLS))]).astype(np.float32)
        X = np.load(out / f"X_{k}.npy")
        np.save(out / f"X_{k}.npy", np.hstack([X, TE]))
    names += [f"te_{c}" for c in TE_COLS]
    pm = {"task": "regression", "feature_names": names, "n_features": len(names), "split_counts": meta["split_counts"], "date_ranges": meta["date_ranges"],
          "clip_bounds": meta["clip_bounds"], "target_encoded": TE_COLS}
    (out / "prepared_meta.json").write_text(json.dumps(pm, indent=2))
    cfg.to_json(out / "run_config.json")
    logger(f"candidate variables: {len(names)} (incl. {len(TE_COLS)} target-encoded codes)")
    return pm
