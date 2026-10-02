"""
selection.py - variable ranking used by every method's "k = number of top variables" hyper-parameter.

For each CV fold (fitted on that fold's TRAINING rows only) and once for the whole tuning subsample (used for the validation curve and the final fit):
    score_j = mean of two ranks:  (a) univariate signal   regression: 3-fold time-blocked CV gain of a 20-bin step function of x_j
                                                          classification: |AUC - 0.5| of x_j
                                  (b) tree-ensemble importance (ExtraTrees: captures non-linear effects and interactions)
The methods then use the top-k columns of the ranking that belongs to the rows they are trained on, so variable selection is part of the cross-validated procedure (no leakage).
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_os.cpu_count() or 1))

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import common


def _uni_reg(X, y):
    n, p = X.shape
    e = np.linspace(0, n, 4).astype(int)
    gain = np.zeros(p)
    for j in range(p):
        q = np.unique(np.quantile(X[:, j], np.linspace(0, 1, 21)[1:-1]))
        b = np.searchsorted(q, X[:, j], side="right")
        sse = sse0 = 0.0
        for f in range(3):
            te = np.zeros(n, bool); te[e[f]:e[f + 1]] = True
            cnt = np.bincount(b[~te], minlength=len(q) + 1); sm = np.bincount(b[~te], weights=y[~te], minlength=len(q) + 1)
            mean = np.where(cnt > 0, sm / np.maximum(cnt, 1), y[~te].mean())
            sse += np.sum((y[te] - mean[b[te]]) ** 2); sse0 += np.sum((y[te] - y[~te].mean()) ** 2)
        gain[j] = 1 - sse / sse0
    return gain


def _uni_clf(X, y):
    pos = int(y.sum()); neg = len(y) - pos
    r = rankdata(X, axis=0)
    return np.abs((r[y == 1].sum(0) - pos * (pos + 1) / 2) / (pos * neg) - 0.5)


def rank_features(X, y_fit, task, seed, trees, max_rows):
    X = np.asarray(X, dtype=np.float32)
    n, p = X.shape
    if n > max_rows:
        if task == "regression":
            rows = np.unique(np.linspace(0, n - 1, max_rows).astype(int))
        else:
            rng = np.random.RandomState(seed); rows = np.sort(np.concatenate([rng.choice(np.where(y_fit == c)[0], int(round(max_rows * (y_fit == c).mean())), replace=False) for c in (0, 1)]))
        X, y_fit = X[rows], y_fit[rows]
    from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
    if task == "regression":
        uni = _uni_reg(X, y_fit)
        m = ExtraTreesRegressor(n_estimators=trees, max_depth=10, min_samples_leaf=max(20, len(y_fit) // 500), max_features=0.3, n_jobs=1, random_state=seed).fit(X, y_fit)
    else:
        uni = _uni_clf(X, y_fit.astype(int))
        m = ExtraTreesClassifier(n_estimators=trees, max_depth=10, min_samples_leaf=max(50, len(y_fit) // 300), max_features=0.3, n_jobs=1, random_state=seed).fit(X, y_fit.astype(int))
    imp = m.feature_importances_
    r_uni, r_imp = rankdata(-uni), rankdata(-imp)
    score = (r_uni + r_imp) / 2 + 1e-6 * r_imp
    order = np.argsort(score, kind="stable")
    # redundancy filter: a variable with |corr| > 0.97 to a better-ranked one is pushed to the back (near-duplicates make additive models / MARS split one effect in two)
    C = np.abs(np.corrcoef(X.T.astype(np.float64)))
    np.nan_to_num(C, copy=False)
    kept, dup = [], []
    for j in order:
        (dup if kept and C[j, kept].max() > 0.97 else kept).append(j)
    return np.array(kept + dup, dtype=np.int32), uni, imp


def _job(prep_dir, cfg_dict, k):
    prep = common.Prepared(prep_dir)
    cfg = common.RunConfig(**cfg_dict)
    task = prep.task
    y_fit = prep.y_fit_train if task == "regression" else prep.y("train").astype(float)
    sub = common.tuning_subsample(prep, cfg)
    Xs, ys = np.asarray(prep.X("train")[sub]), y_fit[sub]
    if k >= 0:
        folds = common.make_folds(prep.y("train")[sub], cfg.cv_folds, task, cfg.seed)
        tr = np.ones(len(ys), bool); tr[folds[k]] = False
        Xs, ys = Xs[tr], ys[tr]
    return k, rank_features(Xs, ys, task, cfg.seed + 7 + k, cfg.rank_trees, cfg.rank_rows)


def build_rankings(prep_dir, cfg, workers=1, logger=print):
    prep_dir = Path(prep_dir)
    prep = common.Prepared(prep_dir)
    K = cfg.cv_folds
    logger(f"ranking {prep.p} candidate variables inside each of {K} CV folds + the full tuning subsample ({workers} worker(s))")
    res = {}
    ids = list(range(K)) + [-1]
    if workers <= 1:                                   # serial: no process pool (also avoids spawn issues)
        for k in ids:
            res[k] = _job(str(prep_dir), cfg.__dict__, k)[1]
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for k, out in ex.map(_job, [str(prep_dir)] * (K + 1), [cfg.__dict__] * (K + 1), ids):
                res[k] = out
    folds = np.stack([res[k][0] for k in range(K)])
    sub, uni, imp = res[-1]
    np.savez(prep_dir / "rankings.npz", folds=folds, sub=sub)
    names = np.array(prep.feature_names)
    df = pd.DataFrame({"rank": np.arange(1, prep.p + 1), "variable": names[sub], "univariate_score": uni[sub], "tree_importance": imp[sub]})
    top = {m: pd.Series(np.bincount(folds[:, :m].ravel(), minlength=prep.p), index=names) for m in (10, 20, 40)}
    df["in_top20_of_folds"] = [int(top[20][v]) for v in df["variable"]]
    df.to_csv(prep_dir / "variable_ranking.csv", index=False)
    stab = {f"top{m}": float(np.mean([len(set(folds[i, :m]) & set(sub[:m])) / m for i in range(K)])) for m in (10, 20, 40)}
    (prep_dir / "variable_selection_stability.json").write_text(json.dumps({"mean_overlap_of_fold_top_k_with_full_top_k": stab}, indent=2))
    logger(f"top 10 variables: {list(df['variable'].head(10))} | fold-vs-full top-k overlap {stab}")
    return stab
