"""
kern.py - numerics for ESLII Ch. 6 kernel smoothing (memory-based methods).

Kernel: tri-cube D(t) = (1 - |t|^3)^3 on |t| <= 1 with a NEAREST-NEIGHBOUR bandwidth h(x0) = distance to the k-th neighbour, k = span * N  (ESLII 6.1);
`span` is the smoothing parameter and is a FRACTION of the reference set, so the same span is used on the tuning subsample and on the full data.

1-D local polynomial / Nadaraya-Watson (degree 0) / local likelihood : fitted on a GRID of target points and interpolated (the loess strategy of ESLII 6.9),
    returning the self-leverage l_x(x) (-> df = trace(S), leave-one-out CV, GCV, Cp) and the squared equivalent-kernel norm (-> pointwise SE).
Varying-coefficient (structured) local regression : kernel in ONE conditioning variable z, locally linear in z, coefficients of the other predictors vary with z (ESLII 6.4.2).
Multivariate kNN tools (classification) : KD-tree neighbour queries, kernel-weighted class proportions, batched local logistic regression, kernel density estimates.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def tricube(t):
    t = np.clip(t, 0.0, 1.0)
    return (1.0 - t ** 3) ** 3


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def make_grid_points(xs, G=400):
    g = np.unique(np.quantile(xs, np.linspace(0, 1, G)))
    return g


def _window(xs, g, k):
    n = len(xs)
    pos = np.searchsorted(xs, g)
    lo, hi = max(0, pos - k), min(n, pos + k)
    d = np.abs(xs[lo:hi] - g)
    kk = min(k, len(d))
    h = np.partition(d, kk - 1)[kk - 1] * (1 + 1e-12) + 1e-12
    a, b = np.searchsorted(xs, g - h, "left"), np.searchsorted(xs, g + h, "right")
    return a, b, h


def local_glm_1d(xs, ys, grid, span, degree, family="gaussian", ridge=1e-8, irls_iter=25):
    """Local polynomial fit of degree `degree` at each grid point. family 'gaussian' (identity link; one IRLS step) or 'binomial' (logit).
    xs must be sorted. Returns dict(fit, self_lev, varfac, coef)."""
    n = len(xs)
    k = int(max(np.ceil(span * n), degree + 2))
    G = len(grid)
    fit, slev, vfac = np.empty(G), np.empty(G), np.empty(G)
    for gi, g in enumerate(grid):
        a, b, h = _window(xs, g, k)
        u = (xs[a:b] - g) / h
        w = tricube(np.abs(u))
        B = np.vander(u, degree + 1, increasing=True)
        y = ys[a:b]
        if family == "gaussian":
            Bw = B * w[:, None]
            M = Bw.T @ B
            Mi = np.linalg.pinv(M, rcond=1e-12)
            beta = Mi @ (Bw.T @ y)
            fit[gi] = beta[0]
            lvec = w * (B @ Mi[:, 0])
            slev[gi], vfac[gi] = Mi[0, 0], lvec @ lvec
        else:
            p0 = np.clip((w @ y) / max(w.sum(), 1e-12), 1e-4, 1 - 1e-4)
            beta = np.zeros(degree + 1); beta[0] = np.log(p0 / (1 - p0))
            for _ in range(irls_iter):
                eta = B @ beta
                mu = sigmoid(eta)
                v = np.clip(mu * (1 - mu), 1e-6, None)
                wt = w * v
                A = (B * wt[:, None]).T @ B + ridge * np.eye(degree + 1)
                z = eta + (y - mu) / v
                new = np.linalg.solve(A, (B * wt[:, None]).T @ z)
                done = np.max(np.abs(new - beta)) < 1e-9
                beta = new
                if done:
                    break
            fit[gi] = beta[0]
            Ai = np.linalg.inv(A)
            slev[gi], vfac[gi] = Ai[0, 0] * wt.sum() / max(len(wt), 1), Ai[0, 0]
    return {"fit": fit, "self_lev": slev, "varfac": vfac}


def local_poly_1d(xs, ys, grid, span, degree):
    return local_glm_1d(xs, ys, grid, span, degree, "gaussian")


def diag_from_grid(xs, ys, grid, res, sig2=None):
    """df = trace(S) = sum_i l_i(x_i), RSS, leave-one-out CV (Ex. 6.7) and GCV from a grid fit (xs sorted)."""
    fh = np.interp(xs, grid, res["fit"])
    L = np.clip(np.interp(xs, grid, res["self_lev"]), 0.0, 0.999)
    n = len(xs)
    rss = float(np.sum((ys - fh) ** 2))
    df = float(L.sum())
    return {"df": df, "rss": rss, "loocv": float(np.mean(((ys - fh) / (1 - L)) ** 2)), "gcv": float((rss / n) / (1 - df / n) ** 2), "n": n}


# ------------------------------------------------------------------------------------------------------------
# structured local regression: varying-coefficient model  y = a(z) + sum_m b_m(z) x_m
# ------------------------------------------------------------------------------------------------------------
def local_vc_1d(zs, Xr, ys, grid, span, m_use, coef_degree):
    """zs sorted; Xr (n, m_use) covariates aligned with zs. Local linear in z for the intercept, coefficients of degree `coef_degree` in z."""
    n = len(zs)
    k = int(max(np.ceil(span * n), 4 * (m_use + 2)))
    G = len(grid)
    P = 2 + m_use * (coef_degree + 1)
    coef = np.zeros((G, P)); hg = np.empty(G)
    lev = np.zeros(n)
    cell_edges = np.r_[-np.inf, 0.5 * (grid[1:] + grid[:-1]), np.inf]
    cell = np.searchsorted(cell_edges, zs, side="right") - 1
    for gi, g in enumerate(grid):
        a, b, h = _window(zs, g, k)
        u = (zs[a:b] - g) / h
        w = tricube(np.abs(u))
        cols = [np.ones_like(u), u]
        for j in range(m_use):
            xj = Xr[a:b, j]
            cols.append(xj)
            if coef_degree == 1:
                cols.append(xj * u)
        B = np.column_stack(cols)
        Bw = B * w[:, None]
        Mi = np.linalg.pinv(Bw.T @ B + 1e-9 * np.eye(P), rcond=1e-12)
        coef[gi] = Mi @ (Bw.T @ ys[a:b])
        hg[gi] = h
        own = np.where(cell[a:b] == gi)[0]
        if len(own):
            lev[a + own] = w[own] * np.einsum("ij,jk,ik->i", B[own], Mi, B[own])
    return {"coef": coef, "h": hg, "lev": lev, "grid": grid, "m_use": m_use, "coef_degree": coef_degree}


def local_vc_predict(res, z, Xr):
    grid, coef, hg, m, cd = res["grid"], res["coef"], res["h"], res["m_use"], res["coef_degree"]
    pos = np.clip(np.searchsorted(grid, z) - 1, 0, len(grid) - 2)
    t = np.clip((z - grid[pos]) / np.maximum(grid[pos + 1] - grid[pos], 1e-300), 0.0, 1.0)

    def at(gi):
        u = (z - grid[gi]) / hg[gi]
        c = coef[gi]
        f = c[:, 0] + c[:, 1] * u
        col = 2
        for j in range(m):
            f = f + (c[:, col] + (c[:, col + 1] * u if cd == 1 else 0.0)) * Xr[:, j]
            col += 1 + cd
        return f
    return (1 - t) * at(pos) + t * at(pos + 1)


# ------------------------------------------------------------------------------------------------------------
# multivariate neighbour tools
# ------------------------------------------------------------------------------------------------------------
def knn_chunks(tree, Xq, kmax, max_elems=1.5e7):
    chunk = int(max(1, max_elems // max(kmax, 1)))
    for a in range(0, len(Xq), chunk):
        d, i = tree.query(Xq[a:a + chunk], k=kmax, workers=1)
        if kmax == 1:
            d, i = d[:, None], i[:, None]
        yield a, d, i


def kernel_weights(D, k):
    h = D[:, k - 1:k] * (1 + 1e-12) + 1e-12
    return tricube(D[:, :k] / h)


def local_logistic_batch(Xc, Y, W, ridge=1e-2, iters=8):
    """Batched local-linear logistic regression. Xc (Q,k,p) neighbours centred at the query, Y (Q,k), W (Q,k) kernel weights.
    Returns eta0 (logit at the query) and its standard error. Slopes are ridge-penalised (relative), the intercept is not."""
    Q, k, p = Xc.shape
    D = np.concatenate([np.ones((Q, k, 1)), Xc], axis=2)
    p0 = np.clip((W * Y).sum(1) / np.maximum(W.sum(1), 1e-12), 1e-3, 1 - 1e-3)
    beta = np.zeros((Q, p + 1)); beta[:, 0] = np.log(p0 / (1 - p0))
    pen = np.r_[0.0, np.ones(p)]
    for _ in range(iters):
        eta = np.clip(np.einsum("qkj,qj->qk", D, beta), -20, 20)
        mu = sigmoid(eta)
        v = np.clip(mu * (1 - mu), 1e-6, None)
        wt = W * v
        z = eta + (Y - mu) / v
        A = np.einsum("qki,qk,qkj->qij", D, wt, D)
        scale = np.einsum("qii->q", A)[:, None, None] / (p + 1)
        A = A + ridge * scale * np.diag(pen)[None] + 1e-9 * scale * np.eye(p + 1)[None]
        beta = np.linalg.solve(A, np.einsum("qki,qk,qk->qi", D, wt, z)[..., None])[..., 0]
    se = np.sqrt(np.maximum(np.linalg.inv(A)[:, 0, 0], 0.0))
    return beta[:, 0], se


def kmeans(X, M, iters=20, seed=0, max_rows=30000):
    rng = np.random.RandomState(seed)
    S = X[rng.choice(len(X), min(len(X), max_rows), replace=False)]
    C = S[rng.choice(len(S), M, replace=False)].copy()
    for _ in range(iters):
        d2 = (S ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None] - 2 * S @ C.T
        lab = np.argmin(d2, axis=1)
        for j in range(M):
            m = lab == j
            if m.any():
                C[j] = S[m].mean(0)
    return C
