"""
nnlib.py - nearest-neighbour glue on top of scikit-learn NearestNeighbors.

* one neighbour search per (number of variables k) serves EVERY neighbourhood size nn of that group: predictions for each nn are read off the sorted neighbour list
  (plain average / vote fraction, or 1/d distance weights exactly as scikit-learn's weights="distance"; tested against KNeighborsRegressor / KNeighborsClassifier)
* classification scores are vote fractions, smoothed (p*nn + base_rate)/(nn + 1) so log-loss is finite; the smoothing is monotone, so AUC is unaffected
* tangent_distance / tangent_projection: the invariant metric of ESLII 13.3.3
"""
import _bootstrap  # noqa: F401
import itertools

import numpy as np
from sklearn.neighbors import NearestNeighbors

import common


def sample_configs(space, n, seed):
    keys = list(space)
    combos = list(itertools.product(*[space[k] for k in keys]))
    rng = np.random.RandomState(seed)
    if n and len(combos) > n:
        combos = [combos[i] for i in sorted(rng.choice(len(combos), n, replace=False))]
    return [dict(zip(keys, c)) for c in combos]


def knn_scores(dist, Y, nn, weights, task, base):
    d, y = dist[:, :nn], Y[:, :nn]
    if weights == "uniform":
        p = y.mean(1)
    else:                                                                   # scikit-learn convention: exact matches take all the weight
        zero = d <= 1e-12
        w = 1.0 / np.maximum(d, 1e-12)
        anyz = zero.any(1)
        w[anyz] = zero[anyz].astype(float)
        p = (w * y).sum(1) / w.sum(1)
    return (p * nn + base) / (nn + 1.0) if task == "classification" else p


def make_predict_path(weights, p, project=None):
    def predict_path(Xref, yref, grid, Xq, cfg, ctx):
        task, order = ctx["task"], ctx["order"]
        out = np.empty((len(grid), len(Xq)))
        y = np.asarray(yref, float)
        base = float(y.mean())
        for kk in sorted({hp["k"] for hp in grid}):
            idx = [h for h, hp in enumerate(grid) if hp["k"] == kk]
            cols = order[:kk]
            Xr, Xqk = np.asarray(Xref[:, cols], np.float64), np.asarray(Xq[:, cols], np.float64)
            if project is not None:
                Xr, Xqk = project(Xr), project(Xqk)
            nmax = min(max(grid[h]["nn"] for h in idx), len(Xr))
            nbrs = NearestNeighbors(n_neighbors=nmax, p=p, n_jobs=cfg.n_jobs).fit(Xr)
            for a in range(0, len(Xqk), 4000):
                dist, ind = nbrs.kneighbors(Xqk[a:a + 4000])
                Y = y[ind]
                for h in idx:
                    out[h, a:a + len(dist)] = knn_scores(dist, Y, min(grid[h]["nn"], nmax), weights, task, base)
            if ctx.get("final"):
                ctx["store"].update(Xr=Xr, yr=y, cols=cols, hp=grid[idx[0]], task=task, p=p, project=project)
                ctx["store"]["summary"] = {"n_reference_rows": int(len(Xr)), "n_features_used": int(len(cols)), "metric_p": p, "weights": weights}
        return out
    return predict_path


def loo_curve(X, y, nn_list, weights, p, task="regression", base=0.0, n_jobs=1):
    """Leave-one-out predictions for every nn from ONE search with nmax + 1 neighbours (the point itself is removed)."""
    n, nmax = len(X), max(nn_list)
    dist, ind = NearestNeighbors(n_neighbors=nmax + 1, p=p, n_jobs=n_jobs).fit(X).kneighbors(X)
    has_self = ind == np.arange(n)[:, None]
    drop = np.where(has_self.any(1), has_self.argmax(1), nmax)                    # position of the point itself (else the farthest neighbour)
    keep = np.ones_like(ind, bool)
    keep[np.arange(n), drop] = False
    ind2, dist2 = ind[keep].reshape(n, nmax), dist[keep].reshape(n, nmax)
    Y = np.asarray(y, float)[ind2]
    return {nn: knn_scores(dist2, Y, nn, weights, task, base) for nn in nn_list}


def tangent_distance(x, x2, T_x=None, T_x2=None):
    """Two-sided tangent distance (ESLII 13.3.3): min over a, b of || (x + T_x a) - (x2 + T_x2 b) ||. Tangent matrices are (p, m); with no tangents it is the Euclidean distance."""
    d = np.asarray(x2, float) - np.asarray(x, float)
    cols = [t for t in (T_x, None if T_x2 is None else -np.asarray(T_x2)) if t is not None and np.size(t)]
    if not cols:
        return float(np.linalg.norm(d))
    A = np.hstack(cols)
    coef = np.linalg.lstsq(A, d, rcond=None)[0]
    return float(np.linalg.norm(d - A @ coef))


def tangent_projection(T):
    """Global tangent space T (p, m): projecting it out gives the (one-sided, shared-tangent) tangent distance with the ordinary Euclidean machinery. T empty -> identity."""
    if T is None or np.size(T) == 0:
        return None
    Q, _ = np.linalg.qr(np.asarray(T, float))
    return lambda X: X - (X @ Q) @ Q.T
