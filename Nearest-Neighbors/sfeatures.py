"""
features.py - stratified 80/10/10 split and the two feature sets (all transforms fitted on TRAIN only).

Split   : stratified on Target (class ratio identical in train/val/test). split_report.csv also lists, for every feature,
          the Kolmogorov-Smirnov distance train-vs-val and train-vs-test, to show the splits represent the full data.

lean    : 200 columns.  winsorise at train 0.1 / 99.9 percentiles -> rank-to-Gaussian (train empirical CDF -> normal quantile)
          -> standardise.  Keeps the Gaussian shape the discriminant methods assume.
rich    : 600 columns = lean + 200 frequency features + 200 weight-of-evidence (WoE) features
          * frequency  : log(1 + #train rows with the same raw value); for TRAIN rows the row itself is excluded (leave-one-out)
                         so train and val/test counts are distributed alike
          * WoE        : 20 train-quantile bins per variable, smoothed log-odds ratio of positives vs negatives; TRAIN rows use
                         5-fold out-of-fold tables (no target leakage into their own feature), val/test use the full train table
          all standardised with train mean / std.
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
    """Target + identifier / row-index-like columns (excluded) + numeric feature columns. Index-like = a known name (rn, row_number, index, id, ...) or an integer column that is
    (nearly) unique per row and monotone / consecutive (a row counter) - such columns can leak the target when the file was sorted by it."""
    tcol = [c for c in df.columns if str(c).lower() == "target"]
    if not tcol:
        raise ValueError("no 'Target' column found (case-insensitive)")
    tcol = tcol[0]
    names = {"rn", "row_number", "rownum", "row_num", "rowid", "row_id", "rownumber", "index", "idx", "unnamed: 0", "id", "id_code", "idcode"}
    n = len(df)
    idcols = []
    for c in df.columns:
        if c == tcol:
            continue
        if str(c).lower().strip() in names:
            idcols.append(c); continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and s.notna().all():
            v = s.values
            if np.all(np.mod(v, 1) == 0) and s.nunique() >= 0.98 * n and (s.is_monotonic_increasing or s.is_monotonic_decreasing or (v.max() - v.min() + 1) <= 1.02 * n):
                idcols.append(c)
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
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tcol, idcols, feats = identify_columns(df)
    y = df[tcol].astype(int).values
    X = df[feats].values.astype(np.float64)
    n, p = X.shape
    logger(f"table: {n} rows, {p} feature columns, target='{tcol}', id columns={idcols}, positive rate={y.mean():.4f}")
    if np.isnan(X).any():
        logger("  NaNs found: filling with train medians")
    itr, iva, ite = stratified_split(y, cfg.seed)
    np.savez(out_root / "split_indices.npz", train=itr, val=iva, test=ite)
    logger(f"split sizes train/val/test = {len(itr)}/{len(iva)}/{len(ite)} ; positive rate = "
           f"{y[itr].mean():.4f}/{y[iva].mean():.4f}/{y[ite].mean():.4f} (all data {y.mean():.4f})")
    Xtr, Xva, Xte = X[itr], X[iva], X[ite]
    ytr, yva, yte = y[itr], y[iva], y[ite]
    if np.isnan(X).any():
        med = np.nanmedian(Xtr, axis=0)
        for A in (Xtr, Xva, Xte):
            r, c = np.where(np.isnan(A))
            A[r, c] = med[c]

    # --- target-leakage guard: a single feature that (almost) separates the classes cannot be a real Santander feature (their univariate AUC <= ~0.6) ---
    _ytr = y[itr]
    leak_thr = float(getattr(cfg, "leak_auc", 0.9) or 0)
    excluded = []
    if leak_thr > 0.5:
        from scipy.stats import rankdata as _rk
        _pos = int(_ytr.sum()); _neg = len(_ytr) - _pos
        _auc = (_rk(Xtr, axis=0)[_ytr == 1].sum(0) - _pos * (_pos + 1) / 2) / (_pos * _neg)
        _bad = np.where((_auc >= leak_thr) | (_auc <= 1 - leak_thr))[0]
        for j in _bad:
            excluded.append({"column": feats[j], "reason": f"univariate train AUC {_auc[j]:.4f}: target leakage suspected", "train_auc": float(_auc[j])})
            logger(f"WARNING: excluding '{feats[j]}' - univariate train AUC {_auc[j]:.4f} (real Santander features reach ~0.6 at most): target leakage suspected")
        if len(_bad):
            _keep = np.setdiff1d(np.arange(len(feats)), _bad)
            feats = [feats[j] for j in _keep]
            Xtr, Xva, Xte = Xtr[:, _keep], Xva[:, _keep], Xte[:, _keep]
            p = len(feats)
    pd.DataFrame(excluded + [{"column": c, "reason": "identifier / row-index-like column", "train_auc": None} for c in idcols]).to_csv(out_root / "excluded_columns.csv", index=False)
    if idcols:
        logger(f"excluded identifier / row-index-like columns: {idcols}")
    # --- split representativeness report ---
    rows = []
    for j, name in enumerate(feats):
        k1, k2 = ks_2samp(Xtr[:, j], Xva[:, j]), ks_2samp(Xtr[:, j], Xte[:, j])
        rows.append({"feature": name, "ks_train_val": k1.statistic, "p_train_val": k1.pvalue,
                     "ks_train_test": k2.statistic, "p_train_test": k2.pvalue})
    rep = pd.DataFrame(rows)
    rep.to_csv(out_root / "split_report.csv", index=False)
    logger(f"split check: max KS train-val {rep.ks_train_val.max():.4f}, train-test {rep.ks_train_test.max():.4f}; "
           f"features with p<0.01: {(rep.p_train_val < 0.01).sum()} (val), {(rep.p_train_test < 0.01).sum()} (test) of {p} (~{0.01 * p:.0f} expected by chance)")

    # --- lean ---
    lo, hi = np.quantile(Xtr, 0.001, axis=0), np.quantile(Xtr, 0.999, axis=0)
    clipped = {"train": int(((Xtr < lo) | (Xtr > hi)).sum())}
    lean_tr, lean_va, lean_te = np.empty_like(Xtr), np.empty_like(Xva), np.empty_like(Xte)
    for j in range(p):
        srt = _rankgauss_fit(np.clip(Xtr[:, j], lo[j], hi[j]))
        for A, B in ((Xtr, lean_tr), (Xva, lean_va), (Xte, lean_te)):
            B[:, j] = _rankgauss_apply(srt, np.clip(A[:, j], lo[j], hi[j]))
    mu, sd = lean_tr.mean(0), lean_tr.std(0)
    sd[sd < 1e-12] = 1.0
    lean = [(A - mu) / sd for A in (lean_tr, lean_va, lean_te)]

    # --- frequency features ---
    freq_tr, freq_va, freq_te = np.empty_like(Xtr), np.empty_like(Xva), np.empty_like(Xte)
    for j in range(p):
        uniq, cnt = np.unique(Xtr[:, j], return_counts=True)
        pos = np.searchsorted(uniq, Xtr[:, j])
        freq_tr[:, j] = np.log1p(cnt[pos] - 1)
        for A, B in ((Xva, freq_va), (Xte, freq_te)):
            pos = np.clip(np.searchsorted(uniq, A[:, j]), 0, len(uniq) - 1)
            B[:, j] = np.log1p(np.where(uniq[pos] == A[:, j], cnt[pos], 0))
    fm, fs = freq_tr.mean(0), freq_tr.std(0)
    fs[fs < 1e-12] = 1.0
    freq = [(A - fm) / fs for A in (freq_tr, freq_va, freq_te)]

    # --- WoE features ---
    rng = np.random.RandomState(cfg.seed + 1)
    fold_id = np.zeros(len(ytr), dtype=int)
    for c in (0, 1):
        idx = rng.permutation(np.where(ytr == c)[0])
        for f, a in enumerate(np.array_split(idx, WOE_FOLDS)):
            fold_id[a] = f
    woe_tr, woe_va, woe_te = np.empty_like(Xtr), np.empty_like(Xva), np.empty_like(Xte)

    def table(bins, yy, nb):
        pos = np.bincount(bins[yy == 1], minlength=nb).astype(float)
        neg = np.bincount(bins[yy == 0], minlength=nb).astype(float)
        return np.log((pos + 0.5) / (pos.sum() + 0.5 * nb)) - np.log((neg + 0.5) / (neg.sum() + 0.5 * nb))

    for j in range(p):
        xw = np.clip(Xtr[:, j], lo[j], hi[j])
        edges = np.unique(np.quantile(xw, np.linspace(0, 1, N_BINS + 1)[1:-1]))
        nb = len(edges) + 1
        b_tr = np.searchsorted(edges, xw, side="right")
        for f in range(WOE_FOLDS):
            m = fold_id == f
            woe_tr[m, j] = table(b_tr[~m], ytr[~m], nb)[b_tr[m]]
        full = table(b_tr, ytr, nb)
        woe_va[:, j] = full[np.searchsorted(edges, np.clip(Xva[:, j], lo[j], hi[j]), side="right")]
        woe_te[:, j] = full[np.searchsorted(edges, np.clip(Xte[:, j], lo[j], hi[j]), side="right")]
    wm, ws = woe_tr.mean(0), woe_tr.std(0)
    ws[ws < 1e-12] = 1.0
    woe = [(A - wm) / ws for A in (woe_tr, woe_va, woe_te)]

    names_lean = [f"{c}__rankgauss" for c in feats]
    sets = {"lean": (lean, names_lean),
            "rich": ([np.hstack([a, b, c]).astype(np.float32) for a, b, c in zip(lean, freq, woe)],
                     names_lean + [f"{c}__freq" for c in feats] + [f"{c}__woe" for c in feats])}
    for fs_name, (arrs, names) in sets.items():
        d = out_root / fs_name
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("bundle_*.pkl"):                # drop CV caches built from a previous dataset
            old.unlink()
        for s, A, yy in zip(("train", "val", "test"), arrs, (ytr, yva, yte)):
            np.save(d / f"X_{s}.npy", np.ascontiguousarray(A, dtype=np.float32))
            np.save(d / f"y_{s}.npy", yy.astype(np.int8))
        (d / "feature_names.json").write_text(json.dumps(names))
        logger(f"  feature set '{fs_name}': {len(names)} columns")
    meta = {"features": feats, "target": tcol, "id_columns": idcols, "n": int(n), "winsorised_train_values": clipped,
            "woe_bins": N_BINS, "woe_oof_folds": WOE_FOLDS}
    (out_root / "feature_meta.json").write_text(json.dumps(meta, indent=2))
    cfg.to_json(out_root / "run_config.json")
    return meta
