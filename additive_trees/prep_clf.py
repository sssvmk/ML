"""
prep_clf.py - Santander: stratified 80/10/10 split (class ratio + feature distributions checked by KS) and FULL feature engineering = 600 candidate variables:
  200 rank-gaussed (winsorised at train 0.1/99.9 pct -> train ECDF -> normal quantile)   200 leave-one-out frequency features (log(1+count of the same value in train))
  200 weight-of-evidence features (20 train-quantile bins; out-of-fold for TRAIN rows)   -> all standardised on TRAIN. Each method selects its own top-k variables from this pool.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

import sfeatures as sf


def build_prepared(df, out_root, cfg, logger=print):
    out = Path(out_root)
    tmp = out / "_sf"
    meta = sf.build_prepared(df, tmp, cfg, logger=logger)
    names = json.loads((tmp / "rich" / "feature_names.json").read_text())
    for s in ("train", "val", "test"):
        shutil.copy(tmp / "rich" / f"X_{s}.npy", out / f"X_{s}.npy")
        np.save(out / f"y_raw_{s}.npy", np.load(tmp / "rich" / f"y_{s}.npy").astype(np.int8))
    for f in ("split_report.csv", "split_indices.npz"):
        if (tmp / f).exists():
            shutil.copy(tmp / f, out / f)
    shutil.rmtree(tmp, ignore_errors=True)
    y = np.load(out / "y_raw_train.npy")
    pm = {"task": "classification", "feature_names": names, "n_features": len(names), "target": meta["target"], "n": meta["n"], "positive_rate_train": float(y.mean())}
    (out / "prepared_meta.json").write_text(json.dumps(pm, indent=2))
    cfg.to_json(out / "run_config.json")
    logger(f"candidate variables: {len(names)}")
    return pm
