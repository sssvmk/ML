"""
solvers.py - pathwise coordinate descent (Friedman, Hastie, Tibshirani; ESLII Sec. 3.8.6)

Used as the solver of BOTH the lasso and the elastic net (so it is not a separate method).
Works on the covariance form (Gram matrix) with warm starts along a decreasing lambda path
and active-set cycling.  Numba is used automatically if installed; otherwise pure numpy.

Objective (ESLII eq. 3.91 parametrisation, divided by 2n):
    RSS + lam * [ alpha * ||b||_2^2 + (1 - alpha) * ||b||_1 ]
    alpha = 0  -> lasso      (0 < alpha < 1) -> elastic net
"""
from __future__ import annotations

import numpy as np

try:                                   # optional acceleration
    from numba import njit
    _HAVE_NUMBA = True
except Exception:                      # pragma: no cover
    _HAVE_NUMBA = False


def _sweep_py(Gn, cn, diag, beta, q, idx, lam1, lam2):
    maxchg = 0.0
    for j in idx:
        bj = beta[j]
        z = cn[j] - q[j] + diag[j] * bj
        if z > lam1:
            nb = (z - lam1) / (diag[j] + lam2)
        elif z < -lam1:
            nb = (z + lam1) / (diag[j] + lam2)
        else:
            nb = 0.0
        if nb != bj:
            d = nb - bj
            q += Gn[j] * d
            beta[j] = nb
            chg = diag[j] * d * d
            if chg > maxchg:
                maxchg = chg
    return maxchg


_sweep = _sweep_py
if _HAVE_NUMBA:                         # pragma: no cover
    try:
        _sweep_nb = njit(cache=False)(_sweep_py)
    except Exception:
        _sweep_nb = None
else:
    _sweep_nb = None


def _run_sweep(*args):
    global _sweep_nb
    if _sweep_nb is not None:
        try:
            return _sweep_nb(*args)
        except Exception:               # compilation problem -> fall back permanently
            _sweep_nb = None
    return _sweep_py(*args)


def cd_path(Gn, cn, lam1s, lam2s, tol, max_inner=200, max_outer=50):
    """Warm-started pathwise CD. minimise 0.5 b'Gn b - cn'b + lam1*|b|_1 + 0.5*lam2*|b|^2 for each (lam1, lam2).
    Returns B (p x L)."""
    p = len(cn)
    beta, q = np.zeros(p), np.zeros(p)
    diag = np.maximum(np.diag(Gn).copy(), 1e-12)
    allidx = np.arange(p)
    B = np.empty((p, len(lam1s)))
    for l, (lam1, lam2) in enumerate(zip(lam1s, lam2s)):
        for _ in range(max_outer):
            chg = _run_sweep(Gn, cn, diag, beta, q, allidx, lam1, lam2)      # full cycle
            if chg < tol:
                break
            active = np.flatnonzero(beta)
            for _ in range(max_inner):                                      # iterate on active set
                if _run_sweep(Gn, cn, diag, beta, q, active, lam1, lam2) < tol:
                    break
        B[:, l] = beta
    return B


def enet_path_abs(st, lams, alpha):
    """Elastic-net path for absolute penalties `lams` (RSS + lam*[alpha*|b|^2 + (1-alpha)*|b|_1]).
    `st` is a Stats object. Returns (B, b0)."""
    Gc, cc, yyc, xbar, ybar = st.centered()
    n = st.n
    Gn, cn = Gc / n, cc / n
    lams = np.asarray(lams, float)
    lam1 = lams * (1 - alpha) / (2 * n)
    lam2 = lams * alpha / n
    tol = 1e-12 * max(yyc / n, 1e-12)
    B = cd_path(Gn, cn, lam1, lam2, tol)
    return B, ybar - xbar @ B


def lam_max_abs(st, alpha):
    """Smallest lam (absolute scale) for which every coefficient is zero (alpha < 1)."""
    Gc, cc, yyc, xbar, ybar = st.centered()
    return 2 * np.max(np.abs(cc)) / (1 - alpha)
