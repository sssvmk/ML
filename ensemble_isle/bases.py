"""
bases.py - basis functions and penalties for ESLII Ch. 5 (all one- or two-dimensional smoothers).

  trunc_power_basis      piecewise polynomials / regression splines (truncated power basis, ESLII 5.2)
  natural_cubic_basis    natural cubic spline basis N_1..N_K (ESLII 5.2.1): linear beyond the boundary knots
  bs_*                   cubic B-spline basis, exact integrated-second-derivative penalty  (ESLII appendix)
  tps_*                  reduced-rank thin-plate spline basis with k-means knots (ESLII 5.7)
  df_to_lambda           effective-df parameterisation, df(lambda) = trace(S_lambda) (ESLII 5.4.1/5.5.1)
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import eigh


def quantile_knots(x, K):
    if K <= 0:
        return np.array([])
    return np.unique(np.quantile(x, np.linspace(0, 1, K + 2)[1:-1]))


def trunc_power_basis(x, knots, degree=3):
    """[1, x, .., x^d, (x-xi_1)_+^d, ..]; degree 0 = piecewise constant (indicator jumps), 1 = continuous piecewise linear, 3 = cubic spline."""
    cols = [np.ones_like(x)] + [x ** j for j in range(1, degree + 1)]
    for k in knots:
        cols.append((x > k).astype(float) if degree == 0 else np.maximum(x - k, 0.0) ** degree)
    return np.column_stack(cols)


def ncs_knots(x, K):
    """K knots: the data minimum, the maximum and K-2 quantiles in between (K >= 2)."""
    if K <= 2:
        return np.array([x.min(), x.max()])
    return np.unique(np.r_[x.min(), np.quantile(x, np.linspace(0, 1, K)[1:-1]), x.max()])


def natural_cubic_basis(x, knots):
    """N_1 = 1, N_2 = x, N_{k+2} = d_k - d_{K-1}, d_k = [(x-xi_k)_+^3 - (x-xi_K)_+^3] / (xi_K - xi_k)."""
    K = len(knots)
    cols = [np.ones_like(x), x]
    if K > 2:
        def d(k):
            return (np.maximum(x - knots[k], 0.0) ** 3 - np.maximum(x - knots[-1], 0.0) ** 3) / (knots[-1] - knots[k])
        dK1 = d(K - 2)
        for k in range(K - 2):
            cols.append(d(k) - dK1)
    return np.column_stack(cols)


def bs_knots(lo, hi, interior, deg=3):
    """Clamped knot vector; interior knots are made unique and strictly inside (lo, hi) (ties in x can put quantiles on the boundary)."""
    it = np.unique(np.asarray(interior, float))
    it = it[(it > lo) & (it < hi)]
    return np.r_[[lo] * (deg + 1), it, [hi] * (deg + 1)]


def bs_design(x, t, deg=3):
    lo, hi = t[deg], t[-deg - 1]
    return BSpline.design_matrix(np.clip(x, lo, hi), t, deg).toarray()


def bs_penalty(t, deg=3):
    """Omega_jk = integral B_j''(u) B_k''(u) du, exact (Gauss-Legendre per knot interval; B'' is piecewise linear)."""
    m = len(t) - deg - 1
    lo, hi = t[deg], t[-deg - 1]
    br = np.unique(t[(t >= lo) & (t <= hi)])
    gx, gw = np.polynomial.legendre.leggauss(3)
    D2 = BSpline(t, np.eye(m), deg).derivative(2)
    Om = np.zeros((m, m))
    for a, b in zip(br[:-1], br[1:]):
        xq, wq = 0.5 * (b - a) * gx + 0.5 * (a + b), 0.5 * (b - a) * gw
        Dq = D2(xq)
        Om += (Dq * wq[:, None]).T @ Dq
    return Om


def _eta(r):
    out = np.zeros_like(r)
    m = r > 0
    out[m] = r[m] ** 2 * np.log(r[m])
    return out


def pairwise_dist(A, B):
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.sqrt(np.maximum(d2, 0.0))


def kmeans2d(X, m, iters=15, seed=0, max_rows=20000):
    rng = np.random.RandomState(seed)
    S = X[rng.choice(len(X), min(len(X), max_rows), replace=False)]
    C = S[rng.choice(len(S), m, replace=False)].copy()
    for _ in range(iters):
        lab = np.argmin(pairwise_dist(S, C), axis=1)
        for j in range(m):
            sel = lab == j
            if sel.any():
                C[j] = S[sel].mean(0)
    return C


def tps_setup(knots):
    """Reduced-rank thin-plate spline: f = sum_j a_j eta(|x - xi_j|) + b0 + b1 x1 + b2 x2, constraint T'a = 0, penalty a'Ea."""
    m = len(knots)
    E = _eta(pairwise_dist(knots, knots))
    T = np.c_[np.ones(m), knots]
    Q, _ = np.linalg.qr(T, mode="complete")
    Z = Q[:, 3:]
    Om = np.zeros((m, m))
    Om[: m - 3, : m - 3] = Z.T @ E @ Z
    return {"knots": knots, "Z": Z, "Omega": (Om + Om.T) / 2}


def tps_design(X, st, chunk=20000):
    out = np.empty((len(X), st["Z"].shape[1] + 3))
    for a in range(0, len(X), chunk):
        xb = X[a:a + chunk]
        out[a:a + chunk, :-3] = _eta(pairwise_dist(xb, st["knots"])) @ st["Z"]
        out[a:a + chunk, -3] = 1.0
        out[a:a + chunk, -2:] = xb
    return out


def df_to_lambda(G, Omega, targets, jitter=1e-10):
    """lambda with trace(S_lambda) = target, via the Demmler-Reinsch eigenvalues d_k of  Omega v = d G v."""
    m = G.shape[0]
    d = eigh(Omega, G + jitter * np.trace(G) / m * np.eye(m), eigvals_only=True)
    d = np.clip(d, 0.0, None)
    null_dim = int((d < 1e-9 * max(d.max(), 1e-300)).sum())
    lams = []
    for T in targets:
        T = float(np.clip(T, null_dim + 1e-3, m - 1e-3))
        lo, hi = -30.0, 30.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if np.sum(1.0 / (1.0 + np.exp(mid) * d)) > T:
                lo = mid
            else:
                hi = mid
        lams.append(float(np.exp(0.5 * (lo + hi))))
    return np.array(lams), null_dim
