"""
isle_lib.py - glue for the Chapter 16 entries. All fitting is done by scikit-learn:
  stage 1 (dictionary)  GradientBoosting* with subsample eta and shrinkage nu (ISLE, Alg. 16.2) / RandomForest* with max_samples = eta (ISLE-RF) / the trees of a boosted model turned into RULES (RuleFit)
  stage 2 (post-processing)  Lasso (regression) or L1-penalised LogisticRegression (classification) fitted along a path of penalties with warm starts (16.9)
Rules: every non-root node of every tree is a rule R(x) = product of the split indicators on its path; scikit-learn's decision_path() returns exactly these indicators (the linear expansion in the node
indicators reproduces the tree, tested). Linear terms are winsorised at the 5 / 95 % quantiles and standardised.
Variable policy as in the earlier chapters: k = round(frac * p) fold-ranked variables, tuned; frac < 1 counts as MORE complex (all variables are kept unless dropping some is clearly better).
"""
import _bootstrap  # noqa: F401
import itertools
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

import common

FRACS = [1.0, 0.5, 0.25]
LAMR_REG = [1.0, 0.5, 0.3, 0.2, 0.12, 0.08, 0.05, 0.03, 0.02, 0.012, 0.008, 0.005, 0.003, 0.002, 0.001]     # lambda / lambda_max
LAMR_CLF = [1.0, 0.5, 0.3, 0.2, 0.12, 0.08, 0.05, 0.03, 0.02, 0.012, 0.008, 0.005]                         # C = C_min / lamr


def sample_configs(space, n, seed):
    keys = list(space)
    combos = list(itertools.product(*[space[k] for k in keys]))
    rng = np.random.RandomState(seed)
    if n and len(combos) > n:
        combos = [combos[i] for i in sorted(rng.choice(len(combos), n, replace=False))]
    cfgs = [dict(zip(keys, c)) for c in combos]
    if "frac" in space and 1.0 in space["frac"] and not any(c["frac"] == 1.0 for c in cfgs):
        cfgs[-1] = {**cfgs[-1], "frac": 1.0}
    return cfgs


def with_k(configs, p):
    return [{**c, "k": max(1, int(round(c["frac"] * p)))} for c in configs]


def drop_penalty(hp):
    return 1e6 * (1.0 - hp["frac"])


def cap_rows(n, cap):
    return np.arange(n) if n <= cap else np.unique(np.linspace(0, n - 1, cap).astype(int))


def trees_of(model):
    est = model.estimators_
    return list(est[:, 0]) if getattr(est, "ndim", 1) == 2 else list(est)


def tree_matrix(trees, X):
    """Dictionary of tree outputs T_m(x): GradientBoosting trees (regression trees fitted to gradients) -> predict; RandomForest classifier trees -> P(class 1)."""
    cols = [t.predict_proba(X)[:, 1] if hasattr(t, "predict_proba") else t.predict(X) for t in trees]
    return np.column_stack(cols).astype(np.float64)


def rule_matrix(trees, X):
    """Indicators of every non-root node of every tree (decision_path); a rule = the product of the split indicators on the path to its node (16.14)."""
    return sp.hstack([t.decision_path(X)[:, 1:] for t in trees], format="csr", dtype=np.float32)


def rule_meta(trees, feature_names):
    """Depth, variables and a readable text for every column of rule_matrix (same order)."""
    meta = []
    for ti, t in enumerate(trees):
        tr = t.tree_
        n = tr.node_count
        parent, side = np.full(n, -1), np.zeros(n, int)
        for node in range(n):
            l, r = tr.children_left[node], tr.children_right[node]
            if l != -1:
                parent[l], side[l], parent[r], side[r] = node, 0, node, 1
        for node in range(1, n):
            conds, v = [], node
            while parent[v] != -1:
                pn = parent[v]
                conds.append((int(tr.feature[pn]), "<=" if side[v] == 0 else ">", float(tr.threshold[pn])))
                v = pn
            conds = conds[::-1]
            meta.append({"tree": ti, "node": node, "depth": len(conds), "features": sorted({c[0] for c in conds}), "text": " & ".join(f"{feature_names[f]} {op} {thr:.3g}" for f, op, thr in conds)})
    return meta


class Winsorizer:
    def fit(self, X, q=0.05):
        self.lo, self.hi = np.quantile(X, q, axis=0), np.quantile(X, 1 - q, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lo, self.hi)


class Dictionary:
    """Scaled dictionary of basis functions: tree outputs ('trees') or rules (+ winsorised linear terms) ('rules')."""
    def __init__(self, kind, linear=False):
        self.kind, self.linear = kind, linear

    def fit_transform(self, trees, X):
        from sklearn.preprocessing import StandardScaler
        self.trees = trees
        if self.kind == "trees":
            self.sc = StandardScaler().fit(tree_matrix(trees, X))
            return self.sc.transform(tree_matrix(trees, X))
        R = rule_matrix(trees, X)
        self.sc = StandardScaler(with_mean=False).fit(R)
        self.rule_support = np.asarray(R.mean(0)).ravel()
        self.n_rules = R.shape[1]
        parts = [self.sc.transform(R)]
        if self.linear:
            self.win = Winsorizer().fit(X)
            L = self.win.transform(X)
            self.sc_lin = StandardScaler().fit(L)
            parts.append(sp.csr_matrix(self.sc_lin.transform(L)))
        return sp.hstack(parts, format="csr") if len(parts) > 1 else parts[0]

    def transform(self, X):
        if self.kind == "trees":
            return self.sc.transform(tree_matrix(self.trees, X))
        parts = [self.sc.transform(rule_matrix(self.trees, X))]
        if self.linear:
            parts.append(sp.csr_matrix(self.sc_lin.transform(self.win.transform(X))))
        return sp.hstack(parts, format="csr") if len(parts) > 1 else parts[0]


def make_l1_logreg(C, seed):
    import sklearn
    from sklearn.linear_model import LogisticRegression
    kw = dict(l1_ratio=1.0) if tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 8) else dict(penalty="l1")
    return LogisticRegression(solver="saga", C=C, max_iter=300, tol=1e-3, warm_start=True, random_state=seed, **kw)


def lasso_path_fit(Ts, y, Tq_list, lamrs, seed=0):
    """Lasso along relative penalties lambda / lambda_max (warm-started coordinate descent): predictions on every query matrix, coefficient counts and the coefficients."""
    from sklearn.linear_model import Lasso
    yc = np.asarray(y, float) - float(np.mean(y))
    amax = float(np.max(np.abs(Ts.T @ yc))) / len(yc)
    m = Lasso(alpha=amax, fit_intercept=True, warm_start=True, max_iter=3000, tol=1e-4)
    preds = [[None] * len(lamrs) for _ in Tq_list]
    nnz, coefs = np.zeros(len(lamrs), int), [None] * len(lamrs)
    for j in np.argsort(-np.asarray(lamrs)):
        m.set_params(alpha=amax * lamrs[j])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Ts, y)
        for qi, Tq in enumerate(Tq_list):
            preds[qi][j] = m.predict(Tq)
        nnz[j], coefs[j] = int(np.count_nonzero(m.coef_)), (m.coef_.copy(), float(m.intercept_))
    return preds, nnz, coefs


def logistic_path_fit(Ts, y, Tq_list, lamrs, seed=0):
    """L1-penalised logistic regression along C = C_min / lamr (warm-started saga)."""
    from sklearn.svm import l1_min_c
    cmin = float(l1_min_c(Ts, y, loss="log"))
    lr = make_l1_logreg(cmin / max(lamrs), seed)
    preds = [[None] * len(lamrs) for _ in Tq_list]
    nnz, coefs = np.zeros(len(lamrs), int), [None] * len(lamrs)
    for j in np.argsort(-np.asarray(lamrs)):
        lr.set_params(C=cmin / lamrs[j])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr.fit(Ts, y)
        for qi, Tq in enumerate(Tq_list):
            preds[qi][j] = lr.predict_proba(Tq)[:, 1]
        nnz[j], coefs[j] = int(np.count_nonzero(lr.coef_)), (lr.coef_.ravel().copy(), float(lr.intercept_[0]))
    return preds, nnz, coefs


def path_fit(task, *a, **k):
    return (lasso_path_fit if task == "regression" else logistic_path_fit)(*a, **k)


def make_path_predict(stage1, kind):
    """predict_path for the lasso-post-processed entries (ISLE-GBM, ISLE-RF, RuleFit): one stage-1 ensemble per configuration, then the whole penalty path."""
    def predict_path(Xref, yref, grid, Xq, cfg, ctx):
        task, order = ctx["task"], ctx["order"]
        reg = task == "regression"
        out = np.empty((len(grid), len(Xq)))
        nact = np.full(len(grid), np.nan)
        groups = {}
        for h, hp in enumerate(grid):
            groups.setdefault(tuple(sorted((k, v) for k, v in hp.items() if k != "lamr")), []).append(h)
        for key, hs in groups.items():
            conf = dict(key)
            cols = order[:conf["k"]]
            rows = cap_rows(len(yref), cfg.isle_final_rows) if ctx.get("final") else None
            Xr = np.asarray(Xref[:, cols] if rows is None else Xref[rows][:, cols], np.float32)
            y = yref if rows is None else yref[rows]
            y = np.asarray(y, float) if reg else np.asarray(y).astype(int)
            model = stage1(conf, task, cfg)
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(Xr, y)
            trees = trees_of(model)
            D = Dictionary(kind, linear=bool(conf.get("linear", False)))
            Ts = D.fit_transform(trees, Xr)
            Xqk = np.asarray(Xq[:, cols], np.float32)
            Tq = D.transform(Xqk)
            lamrs = sorted({grid[h]["lamr"] for h in hs}, reverse=True)
            preds, nnz, coefs = path_fit(task, Ts, y, [Tq], lamrs, cfg.seed)
            sec = time.time() - t0
            for h in hs:
                j = lamrs.index(grid[h]["lamr"])
                out[h] = preds[0][j]
                nact[h] = nnz[j]
                if ctx.get("final"):
                    ctx["store"].update(model=model, trees=trees, D=D, Ts=Ts, y=y, conf=conf, cols=cols, hp=grid[h], task=task, coef=coefs[j][0], intercept=coefs[j][1], kind=kind)
                    ctx["store"]["summary"] = {"fit_seconds_final": round(sec, 1), "n_fit_rows": int(len(y)), "n_features_used": int(len(cols))}
        ctx["metrics"]["n_active"] = nact
        return out
    return predict_path


def trajectory_metrics(task, y, p):
    if task == "regression":
        return {"test_mse": float(np.mean((y - p) ** 2))}
    return {"test_auc": common.auc_score(y, p), "test_log_loss": float(common.logloss_vec(y, common.clip_prob(p)).mean())}


def running_curve(T, y, task):
    """Performance of the plain average of the first m columns of a tree-output matrix (a forest-type ensemble)."""
    rows = []
    cum = np.cumsum(T, axis=1) / np.arange(1, T.shape[1] + 1)
    for m in sorted(set(np.unique(np.linspace(1, T.shape[1], 25).astype(int)))):
        rows.append({"trees": int(m), **trajectory_metrics(task, y, cum[:, m - 1])})
    return pd.DataFrame(rows)


def staged_curve(model, X, y, task, n=25):
    it = model.staged_predict(X) if task == "regression" else (p[:, 1] for p in model.staged_predict_proba(X))
    B = model.n_estimators
    want = set(np.unique(np.linspace(1, B, n).astype(int)))
    rows = [{"trees": m, **trajectory_metrics(task, y, np.asarray(p, float))} for m, p in enumerate(it, 1) if m in want]
    return pd.DataFrame(rows)


def l1_margin(y01, f, alpha_l1):
    """Normalised L1 margin (16.7): min_i y_i f(x_i) / sum_k |alpha_k|, y in {-1, +1}."""
    return float(np.min((2 * np.asarray(y01) - 1) * np.asarray(f)) / max(float(alpha_l1), 1e-300))
