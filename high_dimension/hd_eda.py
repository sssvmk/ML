"""hd_eda.py - exploratory data analysis of the TRAINING + validation rows (no test labels): response / class balance, gene scale and outliers, sample outliers, correlation structure, PCA, and the winsorising recommendation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hd_lib import jsonable


def run_eda(task, d, genes, out_dir, logger=print):
    out = Path(out_dir) / "eda"
    out.mkdir(parents=True, exist_ok=True)
    X = np.vstack([d["X_tr"], d["X_va"]]) if len(d["X_va"]) else d["X_tr"]
    y = np.r_[d["y_tr"], d["y_va"]] if len(d["X_va"]) else d["y_tr"]
    n, p = X.shape
    mu, sd = X.mean(0), X.std(0, ddof=1)
    Z = (X - mu) / np.where(sd > 0, sd, 1)
    frac_extreme = float(np.mean(np.abs(Z) > 4))
    s = {"task": task, "n_samples": int(n), "n_genes": int(p), "p_over_n": float(p / n), "n_constant_genes": int(np.sum(sd < 1e-10)), "gene_sd_median": float(np.median(sd)), "gene_sd_range": [float(sd.min()), float(sd.max())],
         "fraction_|z|>4": frac_extreme, "recommend_winsorize": bool(frac_extreme > 0.005), "missing_values": int(np.isnan(X).sum())}
    # sample outliers: Mahalanobis distance in the leading principal components
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    k = min(5, len(S))
    sc = U[:, :k] * S[:k]
    m2 = ((sc / sc.std(0, ddof=1)) ** 2).sum(1)
    from scipy.stats import chi2
    thr = chi2.ppf(0.999, k)
    s["pca_var_explained_top5"] = (S[:k] ** 2 / (S ** 2).sum()).round(4).tolist()
    s["sample_outliers_pca_mahalanobis"] = [int(i) for i in np.where(m2 > thr)[0]]
    # correlation structure
    top = np.argsort(-sd)[:min(300, p)]
    C = np.corrcoef(X[:, top].T)
    s["fraction_gene_pairs_|r|>0.8_(top300_variance)"] = float((np.abs(C[np.triu_indices_from(C, 1)]) > 0.8).mean())
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].hist(sd, bins=50); ax[0, 0].set_title("gene standard deviations"); ax[0, 0].set_xlabel("sd")
    ax[0, 1].hist(mu, bins=50); ax[0, 1].set_title("gene means")
    ax[0, 2].plot(np.arange(1, len(S) + 1), S ** 2 / (S ** 2).sum(), "o-"); ax[0, 2].set_title("PCA scree"); ax[0, 2].set_xlabel("component")
    if task == "regression":
        q1, q3 = np.percentile(y, [25, 75])
        iqr = q3 - q1
        out_y = np.where((y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr))[0]
        s.update(y_mean=float(y.mean()), y_sd=float(y.std(ddof=1)), y_min=float(y.min()), y_max=float(y.max()), y_skew=float(pd.Series(y).skew()), response_outliers_iqr=[int(i) for i in out_y])
        ax[1, 0].hist(y, bins=20); ax[1, 0].set_title(f"response y (IQR outliers: {len(out_y)})")
        r = np.array([np.corrcoef(X[:, j], y)[0, 1] if sd[j] > 0 else 0 for j in range(p)])
        s["max_abs_gene_response_correlation"] = float(np.nanmax(np.abs(r)))
        s["top10_genes_by_abs_correlation"] = [genes[j] for j in np.argsort(-np.abs(np.nan_to_num(r)))[:10]]
        ax[1, 1].scatter(sc[:, 0], sc[:, 1], c=y, cmap="viridis", s=25); ax[1, 1].set_title("PC1 vs PC2 (colour = y)")
        ax[1, 2].hist(np.nan_to_num(r), bins=50); ax[1, 2].set_title("gene-response correlations")
    else:
        cnt = np.bincount(y)[1:]
        s["class_counts"] = cnt.tolist()
        from sklearn.feature_selection import f_classif
        F = np.nan_to_num(f_classif(X, y)[0])
        s["top10_genes_by_F"] = [genes[j] for j in np.argsort(-F)[:10]]
        ax[1, 0].bar(range(1, len(cnt) + 1), cnt); ax[1, 0].set_title("training class counts")
        ax[1, 1].scatter(sc[:, 0], sc[:, 1], c=y, cmap="tab10", s=25); ax[1, 1].set_title("PC1 vs PC2 (colour = class)")
        ax[1, 2].hist(np.log10(F + 1e-3), bins=50); ax[1, 2].set_title("log10 F statistic per gene")
    fig.tight_layout(); fig.savefig(out / "eda_overview.png", dpi=120); plt.close(fig)
    ordr = np.argsort(np.argsort(-sd))[:0]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1); ax.set_title("correlation of the 300 highest-variance genes"); fig.colorbar(im); fig.tight_layout(); fig.savefig(out / "gene_correlation.png", dpi=120); plt.close(fig)
    (out / "eda_summary.json").write_text(json.dumps(jsonable(s), indent=2))
    logger(f"EDA: n={n}, p={p}, constant genes={s['n_constant_genes']}, |z|>4 fraction={frac_extreme:.4f} -> winsorise recommended: {s['recommend_winsorize']}, sample outliers (PCA): {s['sample_outliers_pca_mahalanobis']}")
    return s
