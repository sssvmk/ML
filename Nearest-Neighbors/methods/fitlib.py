"""
fitlib.py - glue for the one-model-per-configuration methods (K-means, LVQ, Gaussian mixtures).

* sample_configs / with_k     random-search configurations; k = round(frac * p) top-ranked variables of the fold-wise ranking (variable selection is part of the tuned model)
* make_predict_path           the `predict_path` called by common.run_method: one fitted model per (configuration, CV fold), scores for the query rows, optional per-fit diagnostics
                              (support fractions, eps-insensitive loss, df, held-out likelihood ...) that the driver averages over the folds
* cap_rows                    systematic (order-preserving) row cap, used for the FINAL fit of the O(N^2) kernel methods
"""
import _bootstrap  # noqa: F401
import itertools
import time

import numpy as np

import common

FRACS = [1.0, 0.5, 0.25, 0.1]


def sample_configs(space, n, seed):
    keys = list(space)
    combos = list(itertools.product(*[space[k] for k in keys]))
    rng = np.random.RandomState(seed)
    if len(combos) > n:
        combos = [combos[i] for i in sorted(rng.choice(len(combos), n, replace=False))]
    cfgs = [dict(zip(keys, c)) for c in combos]
    if "frac" in space and 1.0 in space["frac"] and not any(c["frac"] == 1.0 for c in cfgs):          # always evaluate "keep every variable"
        cfgs[-1] = {**cfgs[-1], "frac": 1.0}
    return cfgs


def with_k(configs, p):
    return [{**c, "k": max(1, int(round(c["frac"] * p)))} for c in configs]


def drop_penalty(hp):
    return 1e6 * (1.0 - hp["frac"])


def cap_rows(n, cap):
    return np.arange(n) if n <= cap else np.unique(np.linspace(0, n - 1, cap).astype(int))


def make_predict_path(build, score, capped=lambda hp: False, diag=None):
    def predict_path(Xref, yref, grid, Xq, cfg, ctx):
        task, order = ctx["task"], ctx["order"]
        out = np.empty((len(grid), len(Xq)))
        met = {}
        for h, hp in enumerate(grid):
            cols = order[:hp["k"]]
            rows = cap_rows(len(yref), cfg.svm_final_rows) if (ctx.get("final") and capped(hp)) else None
            Xr = np.asarray(Xref[:, cols] if rows is None else Xref[rows][:, cols], dtype=np.float64)
            y = yref if rows is None else yref[rows]
            y = np.asarray(y).astype(int) if task == "classification" else np.asarray(y, float)
            model = build(hp, task, cfg, y)
            t0 = time.time()
            model.fit(Xr, y)
            sec = time.time() - t0
            Xqk = np.asarray(Xq[:, cols], dtype=np.float64)
            out[h] = score(model, Xqk, task)
            if diag is not None:
                yq = ctx.get("yq")
                d = diag(model, hp, Xr, y, Xqk, None if (yq is None or len(yq) != len(Xq)) else np.asarray(yq, float), task)
                for k_, v in d.items():
                    met.setdefault(k_, np.full(len(grid), np.nan))[h] = v
            if ctx.get("final"):
                ctx["store"].update(model=model, cols=cols, hp=hp, task=task, Xfit=Xr, yfit=y)
                ctx["store"]["summary"] = {"fit_seconds_final": round(sec, 1), "n_fit_rows": int(len(y)), "n_features_used": int(len(cols)),
                                           **({k_: float(v[h]) for k_, v in met.items()} if False else {})}
        for k_, v in met.items():
            ctx["metrics"][k_] = v
        return out
    return predict_path


def margin_score(model, X, task):
    if task == "regression":
        return model.predict(X)
    return model.decision_function(X) if hasattr(model, "decision_function") else model.predict_proba(X)[:, 1]


def eps_loss(y, f, eps):
    return float(np.mean(np.maximum(0.0, np.abs(y - f) - eps)))
