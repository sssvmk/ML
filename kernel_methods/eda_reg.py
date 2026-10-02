"""
eda.py - exploratory data analysis of zillow.csv (target: logerror) with emphasis on outliers.
All fitted statistics (outlier fences, clip bounds) come from the TRAIN split only; val/test are described for drift.
Outputs -> <results>/eda/ (CSV tables + PNG plots + eda_summary.json)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from prep_reg import CAT_LOW, CAT_HIGH, FLAG_COLS, split_masks


def run_eda(csv_path, results_dir, logger=print):
    out = Path(results_dir) / "eda"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, low_memory=False)
    df["transactiondate"] = pd.to_datetime(df["transactiondate"], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=["transactiondate", "logerror"]).sort_values("transactiondate").reset_index(drop=True)
    masks = split_masks(df["transactiondate"])
    df["split"] = np.select([masks["train"], masks["val"], masks["test"]], ["train", "val", "test"], default="other")
    tr = df[df["split"] == "train"]
    summary = {"rows_read": int(n0), "rows_used": int(len(df)), "columns": int(df.shape[1]),
               "date_min": str(df["transactiondate"].min().date()), "date_max": str(df["transactiondate"].max().date()),
               "split_counts": df["split"].value_counts().to_dict()}
    logger(f"EDA: {summary}")

    # ---- 1. target distribution & outliers (fences fitted on train) ----
    y = tr["logerror"]
    q = y.quantile([0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
    iqr = q[0.75] - q[0.25]
    mad = np.median(np.abs(y - y.median())) * 1.4826
    fence_lo, fence_hi = q[0.25] - 1.5 * iqr, q[0.75] + 1.5 * iqr
    tgt = {"mean": y.mean(), "std": y.std(), "skew": y.skew(), "excess_kurtosis": y.kurt(), "min": y.min(), "max": y.max(),
           **{f"q{k}": v for k, v in q.items()}, "robust_sd_MAD": mad,
           "iqr_fence_lo": fence_lo, "iqr_fence_hi": fence_hi,
           "n_beyond_iqr_fence": int(((y < fence_lo) | (y > fence_hi)).sum()),
           "pct_beyond_iqr_fence": float(((y < fence_lo) | (y > fence_hi)).mean() * 100),
           "n_abs_robust_z_gt_4": int((np.abs(y - y.median()) / mad > 4).sum()),
           "clip_lo_p1": q[0.01], "clip_hi_p99": q[0.99],
           "shapiro_note": "not run (n too large); see QQ plot"}
    pd.Series(tgt).to_csv(out / "logerror_train_stats.csv", header=["value"])
    summary["logerror_train"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in tgt.items()}

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(y, bins=200, color="tab:blue"); ax[0].set_yscale("log"); ax[0].set_title("logerror (train) - log counts")
    for v in (q[0.01], q[0.99]):
        ax[0].axvline(v, color="red", ls="--")
    ax[1].hist(y.clip(q[0.01], q[0.99]), bins=100, color="tab:green"); ax[1].set_title("clipped at train p1/p99 (fit target)")
    stats.probplot(y.sample(min(len(y), 20000), random_state=0), dist="norm", plot=ax[2]); ax[2].set_title("normal QQ-plot")
    fig.tight_layout(); fig.savefig(out / "logerror_distribution.png", dpi=130); plt.close(fig)

    # ---- 2. drift over time / splits ----
    df["ym"] = df["transactiondate"].dt.to_period("M").astype(str)
    mon = df.groupby("ym")["logerror"].agg(["count", "mean", "std", lambda s: s.abs().mean()])
    mon.columns = ["count", "mean", "std", "mean_abs"]
    mon.to_csv(out / "logerror_by_month.csv")
    sp = df.groupby("split")["logerror"].agg(["count", "mean", "std", "skew", "min", "max"])
    sp.to_csv(out / "logerror_by_split.csv")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(mon.index, mon["mean"], "o-", label="mean"); ax[0].plot(mon.index, mon["std"], "s-", label="std")
    ax[0].tick_params(axis="x", rotation=90); ax[0].legend(); ax[0].set_title("logerror by month")
    ax[1].bar(mon.index, mon["count"]); ax[1].tick_params(axis="x", rotation=90); ax[1].set_title("transactions per month")
    fig.tight_layout(); fig.savefig(out / "logerror_by_month.png", dpi=130); plt.close(fig)

    # ---- 3. missingness ----
    feat = [c for c in df.columns if c not in ("logerror", "transactiondate", "split", "ym", "parcelid")]
    miss = (tr[feat].isna().mean() * 100).sort_values(ascending=False)
    miss.to_csv(out / "missingness_train_pct.csv", header=["pct_missing"])
    fig, ax = plt.subplots(figsize=(7, 0.22 * len(miss) + 1))
    ax.barh(miss.index[::-1], miss.values[::-1]); ax.set_xlabel("% missing (train)"); ax.set_title("Missingness")
    fig.tight_layout(); fig.savefig(out / "missingness.png", dpi=130); plt.close(fig)
    summary["columns_over_90pct_missing"] = miss[miss > 90].index.tolist()

    # ---- 4. numeric feature outliers, skew, relationship with target ----
    skip = set(CAT_LOW + CAT_HIGH + FLAG_COLS + ["censustractandblock", "rawcensustractandblock", "parcelid", "propertyzoningdesc", "propertycountylandusecode"])
    num_cols = [c for c in feat if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for c in num_cols:
        v = tr[c].dropna()
        if len(v) < 20:
            continue
        a, b = v.quantile(0.25), v.quantile(0.75)
        f = 1.5 * (b - a)
        rho = tr[[c, "logerror"]].dropna().corr(method="spearman").iloc[0, 1] if v.nunique() > 1 else np.nan
        rows.append({"feature": c, "n": len(v), "skew": v.skew(), "min": v.min(), "p99.9": v.quantile(.999), "max": v.max(),
                     "pct_beyond_1.5IQR": float(((v < a - f) | (v > b + f)).mean() * 100) if f > 0 else 0.0,
                     "spearman_with_logerror": rho})
    ft = pd.DataFrame(rows).sort_values("skew", ascending=False)
    ft.to_csv(out / "numeric_feature_outliers_skew.csv", index=False)
    summary["right_skewed_features_skew_gt_2"] = ft.loc[ft["skew"] > 2, "feature"].tolist()
    top = ft.reindex(ft["spearman_with_logerror"].abs().sort_values(ascending=False).index).head(25)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(top["feature"][::-1], top["spearman_with_logerror"][::-1]); ax.set_title("Spearman correlation with logerror (train)")
    fig.tight_layout(); fig.savefig(out / "spearman_with_logerror.png", dpi=130); plt.close(fig)

    # ---- 5. collinearity (matters for OLS / ridge / PCR / PLS) ----
    sub = tr[num_cols].copy()
    sub = sub.loc[:, sub.notna().mean() > 0.3]
    cm = sub.corr()
    pairs = [(a, b, cm.loc[a, b]) for i, a in enumerate(cm.columns) for b in cm.columns[i + 1:] if abs(cm.loc[a, b]) > 0.9]
    pd.DataFrame(pairs, columns=["feature_a", "feature_b", "pearson"]).sort_values("pearson", key=abs, ascending=False).to_csv(out / "collinear_pairs_abs_gt_0.9.csv", index=False)
    summary["n_collinear_pairs_gt_0.9"] = len(pairs)
    keep = [c for c in top["feature"].head(20) if c in cm.columns]
    if len(keep) > 3:
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(cm.loc[keep, keep], vmin=-1, vmax=1, cmap="coolwarm"); ax.set_xticks(range(len(keep))); ax.set_yticks(range(len(keep)))
        ax.set_xticklabels(keep, rotation=90, fontsize=7); ax.set_yticklabels(keep, fontsize=7); fig.colorbar(im); ax.set_title("Correlation (top features)")
        fig.tight_layout(); fig.savefig(out / "correlation_heatmap.png", dpi=130); plt.close(fig)

    # ---- 6. categorical cardinality ----
    cards = []
    for c in CAT_LOW + CAT_HIGH:
        if c in tr.columns:
            cards.append({"column": c, "n_levels": int(tr[c].nunique()), "pct_missing": float(tr[c].isna().mean() * 100)})
    pd.DataFrame(cards).to_csv(out / "categorical_cardinality.csv", index=False)

    # ---- 7. geography ----
    if {"latitude", "longitude"} <= set(df.columns):
        s = tr.dropna(subset=["latitude", "longitude"]).sample(min(20000, len(tr)), random_state=0)
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(s["longitude"] / 1e6, s["latitude"] / 1e6, c=s["logerror"].clip(q[0.01], q[0.99]), s=2, cmap="coolwarm")
        fig.colorbar(sc); ax.set_title("logerror by location (clipped)"); fig.tight_layout(); fig.savefig(out / "geo_logerror.png", dpi=130); plt.close(fig)

    summary["design_decisions"] = [
        "logerror clipped at train p1/p99 for FITTING only; every metric is computed on raw logerror",
        "no transaction-date features (test months are unseen in train; seasonality cannot be learned from 14 train months)",
        "right-skewed non-negative features get log1p; numeric features winsorised at train p0.5/p99.5",
        "count/area columns where missing means 'none' are zero-filled; other numerics median-imputed + missing indicators (>5% missing)",
        "high-cardinality ids (zip, city, neighbourhood, zoning, tract) keep their top-25 levels, rest pooled into 'other'",
        "redundant columns (bathroomcnt/calculatedbathnbr/fullbathcnt, tax columns) are kept: collinearity is what ridge/PCR/PLS are for",
    ]
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger(f"EDA written to {out}")
    return summary
