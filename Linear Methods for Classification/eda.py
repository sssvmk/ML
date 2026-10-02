"""
eda.py - exploratory data analysis of santander_train.csv (binary Target, 200 anonymous numeric features).
All statistics that drive decisions are computed on the TRAIN split only (stratified 80 %), so the EDA cannot leak
validation/test information into the feature design.   Output -> <results>/eda/
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, skew, kurtosis

from features import identify_columns, stratified_split


def run_eda(df, results_dir, seed=42, logger=print):
    out = Path(results_dir) / "eda"
    out.mkdir(parents=True, exist_ok=True)
    tcol, idcols, feats = identify_columns(df)
    y_all = df[tcol].astype(int).values
    itr, _, _ = stratified_split(y_all, seed)
    tr = df.iloc[itr]
    y, X = tr[tcol].astype(int).values, tr[feats].values.astype(float)
    summary = {"rows": int(len(df)), "feature_columns": len(feats), "id_columns": idcols, "target": tcol,
               "positive_rate_all": float(y_all.mean()), "positive_rate_train": float(y.mean()),
               "imbalance_ratio_neg_per_pos": float((1 - y.mean()) / y.mean()),
               "missing_cells": int(df[feats].isna().sum().sum()), "duplicate_rows": int(df[feats].duplicated().sum()),
               "constant_columns": [c for c in feats if df[c].nunique() <= 1]}
    logger(f"EDA: {summary}")

    # per-feature statistics + outliers
    rows = []
    for j, c in enumerate(feats):
        v = X[:, j]
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        mad = np.median(np.abs(v - np.median(v))) * 1.4826
        pos, neg = v[y == 1], v[y == 0]
        ks = ks_2samp(pos, neg)
        rows.append({"feature": c, "mean": v.mean(), "std": v.std(), "min": v.min(), "max": v.max(),
                     "skew": skew(v), "excess_kurtosis": kurtosis(v), "n_unique": len(np.unique(v)),
                     "pct_unique": 100 * len(np.unique(v)) / len(v),
                     "pct_outside_1.5IQR": 100 * float(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).mean()),
                     "pct_robust_z_gt_4": 100 * float((np.abs(v - np.median(v)) / max(mad, 1e-12) > 4).mean()),
                     "ks_pos_vs_neg": ks.statistic, "mean_diff_pos_minus_neg": pos.mean() - neg.mean(),
                     "std_ratio_pos_over_neg": pos.std() / max(neg.std(), 1e-12)})
    ft = pd.DataFrame(rows)
    ft.to_csv(out / "feature_stats.csv", index=False)
    summary["n_features_with_skew_abs_gt_0.5"] = int((ft["skew"].abs() > 0.5).sum())
    summary["n_features_with_outliers_gt_1pct_robust_z4"] = int((ft["pct_robust_z_gt_4"] > 1).sum())
    summary["median_pct_unique_values"] = float(ft["pct_unique"].median())
    summary["top10_features_by_ks"] = ft.sort_values("ks_pos_vs_neg", ascending=False)["feature"].head(10).tolist()
    summary["n_features_std_ratio_far_from_1"] = int(((ft["std_ratio_pos_over_neg"] - 1).abs() > 0.1).sum())

    # correlations (independence check)
    sub = X[np.random.RandomState(0).choice(len(X), min(50000, len(X)), replace=False)]
    cm = np.corrcoef(sub, rowvar=False)
    off = np.abs(cm[np.triu_indices_from(cm, k=1)])
    summary["max_abs_pairwise_correlation"] = float(off.max())
    summary["mean_abs_pairwise_correlation"] = float(off.mean())
    ev = np.linalg.eigvalsh(np.cov(sub, rowvar=False))[::-1]
    summary["pca_top1_variance_share"] = float(ev[0] / ev.sum())

    # plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].bar(["0", "1"], np.bincount(y) / len(y)); ax[0].set_title("class balance (train)")
    top = ft.sort_values("ks_pos_vs_neg", ascending=False).head(20)
    ax[1].barh(top["feature"][::-1], top["ks_pos_vs_neg"][::-1]); ax[1].set_title("KS(pos vs neg): strongest single features")
    ax[2].hist(off, bins=60); ax[2].set_title(f"|pairwise correlation| (max {off.max():.3f})")
    fig.tight_layout(); fig.savefig(out / "overview.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for a, c in zip(axes.ravel(), top["feature"].head(6)):
        j = feats.index(c)
        a.hist(X[y == 0, j], bins=80, alpha=0.5, density=True, label="0"); a.hist(X[y == 1, j], bins=80, alpha=0.5, density=True, label="1")
        a.set_title(c); a.legend()
    fig.tight_layout(); fig.savefig(out / "top_features_by_class.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(ft["skew"], bins=40); ax[0].set_title("skewness per feature")
    ax[1].hist(ft["excess_kurtosis"], bins=40); ax[1].set_title("excess kurtosis per feature")
    ax[2].hist(ft["pct_unique"], bins=40); ax[2].set_title("% unique values per feature (low => frequency features informative)")
    fig.tight_layout(); fig.savefig(out / "shape_and_cardinality.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(ft)), ft["pct_robust_z_gt_4"]); ax.set_xlabel("feature index"); ax.set_ylabel("% rows with robust |z|>4"); ax.set_title("Outlier share per feature")
    fig.tight_layout(); fig.savefig(out / "outliers.png", dpi=120); plt.close(fig)

    summary["design_decisions"] = [
        "stratified 80/10/10 split on Target; KS check train-vs-val/test written to split_report.csv",
        "outliers: winsorise at train 0.1/99.9 percentiles, then rank-to-Gaussian (robust to heavy tails and non-normal shapes)",
        "features are near-independent (see correlation summary) -> per-feature nonlinear transforms (rank-gauss, WoE, frequency) are the lever, not interactions",
        "many repeated values per feature -> leave-one-out frequency features in the 'rich' set",
        "class-conditional std ratios differ from 1 for some features -> QDA/RDA can exploit variance differences LDA cannot",
        "heavy class imbalance -> Youden threshold on validation; AUC reported next to error; CV selects on log-loss / AUC, not raw error"]
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger(f"EDA written to {out}")
    return summary
