"""
prep_clf.py - stratified 80/10/10 split and features for the Santander kernel-method study.
  200 features: winsorise (train 0.1/99.9 pct) -> rank-to-Gaussian -> standardise (train only) = used by naive Bayes and the Gaussian mixtures.
  top-k features by univariate train AUC (default 8) = used by N-W, local logistic and kernel density classification.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.special import ndtri


N_BINS = 20
WOE_FOLDS = 5


def load_table(csv_path):
    df = pd.read_csv(csv_path)
    return df


def identify_columns(df):
    tcol = [c for c in df.columns if str(c).lower() == "target"]
    if not tcol:
        raise ValueError("no 'Target' column found (case-insensitive)")
    tcol = tcol[0]
    idcols = [c for c in df.columns if str(c).lower() in ("id_code", "id", "idcode")]
    feats = [c for c in df.columns if c != tcol and c not in idcols and pd.api.types.is_numeric_dtype(df[c])]
    return tcol, idcols, feats


def stratified_split(y, seed, fractions=(0.8, 0.1, 0.1)):
    rng = np.random.RandomState(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        a = int(round(fractions[0] * len(idx)))
        b = a + int(round(fractions[1] * len(idx)))
        tr.append(idx[:a]); va.append(idx[a:b]); te.append(idx[b:])
    return [np.sort(np.concatenate(p)) for p in (tr, va, te)]


def _rankgauss_fit(x):
    return np.sort(x)


def _rankgauss_apply(sorted_tr, v):
    n = len(sorted_tr)
    lo, hi = np.searchsorted(sorted_tr, v, "left"), np.searchsorted(sorted_tr, v, "right")
    return ndtri(((lo + hi + 1) / 2.0) / (n + 1.0))


def build_prepared(df, out_root, cfg, logger=print):
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)
    tcol, idcols, feats = identify_columns(df)
    y = df[tcol].astype(int).values
    X = df[feats].values.astype(np.float64)
    n, p = X.shape
    logger(f"table: {n} rows, {p} feature columns, target='{tcol}', id columns={idcols}, positive rate={y.mean():.4f}")
    itr, iva, ite = stratified_split(y, cfg.seed)
    np.savez(out_root / "split_indices.npz", train=itr, val=iva, test=ite)
    logger(f"stratified split train/val/test = {len(itr)}/{len(iva)}/{len(ite)}; positive rate {y[itr].mean():.4f}/{y[iva].mean():.4f}/{y[ite].mean():.4f} (all {y.mean():.4f})")
    Xtr, Xva, Xte = X[itr], X[iva], X[ite]
    if np.isnan(X).any():
        med = np.nanmedian(Xtr, axis=0)
        for A in (Xtr, Xva, Xte):
            r, c = np.where(np.isnan(A)); A[r, c] = med[c]
    rep = pd.DataFrame([{"feature": f, "ks_train_val": ks_2samp(Xtr[:, j], Xva[:, j]).statistic, "ks_train_test": ks_2samp(Xtr[:, j], Xte[:, j]).statistic} for j, f in enumerate(feats)])
    rep.to_csv(out_root / "split_report.csv", index=False)
    logger(f"split check: max KS train-val {rep.ks_train_val.max():.4f}, train-test {rep.ks_train_test.max():.4f}")
    lo, hi = np.quantile(Xtr, 0.001, axis=0), np.quantile(Xtr, 0.999, axis=0)
    out_arrays = [np.empty_like(Xtr), np.empty_like(Xva), np.empty_like(Xte)]
    for j in range(p):                                  # winsorise (train 0.1/99.9 pct) -> rank-to-Gaussian (train ECDF)
        srt = _rankgauss_fit(np.clip(Xtr[:, j], lo[j], hi[j]))
        for A, B in zip((Xtr, Xva, Xte), out_arrays):
            B[:, j] = _rankgauss_apply(srt, np.clip(A[:, j], lo[j], hi[j]))
    mu, sd = out_arrays[0].mean(0), out_arrays[0].std(0); sd[sd < 1e-12] = 1.0
    Z = [(A - mu) / sd for A in out_arrays]
    # top features by univariate signal on TRAIN: |AUC - 0.5| of the rank-gaussed feature
    from scipy.stats import rankdata
    ytr = y[itr]; pos = int(ytr.sum()); neg = len(ytr) - pos
    auc = np.array([(rankdata(Z[0][:, j])[ytr == 1].sum() - pos * (pos + 1) / 2) / (pos * neg) for j in range(p)])
    order = np.argsort(-np.abs(auc - 0.5))
    top = order[: cfg.n_top_features]
    pd.DataFrame({"feature": np.array(feats)[order], "train_auc": auc[order]}).to_csv(out_root / "feature_ranking_train_auc.csv", index=False)
    logger(f"top {cfg.n_top_features} features by univariate train AUC: {[feats[i] for i in top]} (AUC {np.round(auc[top], 4).tolist()})")
    for s, A, yy in zip(("train", "val", "test"), Z, (ytr, y[iva], y[ite])):
        np.save(out_root / f"x200_{s}.npy", np.ascontiguousarray(A, dtype=np.float32))
        np.save(out_root / f"x8_{s}.npy", np.ascontiguousarray(A[:, top], dtype=np.float32))
        np.save(out_root / f"y_raw_{s}.npy", yy.astype(np.int8))
    meta = {"task": "classification", "input_kinds": ["x8", "x200"], "features": feats, "top_features": [feats[i] for i in top], "top_feature_index": top.tolist(), "target": tcol,
            "n": int(n), "positive_rate": {"train": float(y[itr].mean()), "val": float(y[iva].mean()), "test": float(y[ite].mean())}}
    (out_root / "prepared_meta.json").write_text(json.dumps(meta, indent=2))
    cfg.to_json(out_root / "run_config.json")
    return meta
