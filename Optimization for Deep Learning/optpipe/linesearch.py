"""Strong-Wolfe line search (Nocedal & Wright, Algorithms 3.5 and 3.6) and Armijo backtracking.

phi(alpha) -> (f, dphi, g)  where f = objective at x + alpha d, dphi = g . d, g = gradient there.
Returns (alpha, f, g, n_evals) or None when no acceptable step was found.
"""
from __future__ import annotations

import math


def _interp(lo, hi, f_lo, f_hi, dphi_lo):
    """Quadratic interpolation inside [lo, hi] safeguarded to the interior; bisection as fallback."""
    d = hi - lo
    denom = 2.0 * (f_hi - f_lo - dphi_lo * d)
    if denom > 0 and math.isfinite(denom):
        a = lo - dphi_lo * d * d / denom
    else:
        a = lo + 0.5 * d
    a_min, a_max = (lo, hi) if lo < hi else (hi, lo)
    margin = 0.1 * abs(d)
    return min(max(a, a_min + margin), a_max - margin)


def strong_wolfe(phi, f0, dphi0, alpha1=1.0, c1=1e-4, c2=0.9, alpha_max=100.0, max_iter=15, max_zoom=15):
    if dphi0 >= 0:
        return None
    evals = 0
    a_prev, f_prev, d_prev, g_prev = 0.0, f0, dphi0, None
    a = min(alpha1, alpha_max)

    def zoom(lo, hi, f_lo, f_hi, d_lo, g_lo):
        nonlocal evals
        for _ in range(max_zoom):
            a_j = _interp(lo, hi, f_lo, f_hi, d_lo)
            f, d, g = phi(a_j)
            evals += 1
            if f > f0 + c1 * a_j * dphi0 or f >= f_lo:
                hi, f_hi = a_j, f
            else:
                if abs(d) <= -c2 * dphi0:
                    return a_j, f, g, evals
                if d * (hi - lo) >= 0:
                    hi, f_hi = lo, f_lo
                lo, f_lo, d_lo, g_lo = a_j, f, d, g
        if lo > 0 and g_lo is not None and f_lo < f0:       # best sufficient-decrease point found
            return lo, f_lo, g_lo, evals
        return None

    for i in range(max_iter):
        f, d, g = phi(a)
        evals += 1
        if not math.isfinite(f):
            a = 0.5 * (a_prev + a)                          # overshoot into a non-finite region: shrink
            continue
        if f > f0 + c1 * a * dphi0 or (i > 0 and f >= f_prev):
            return zoom(a_prev, a, f_prev, f, d_prev, g_prev)
        if abs(d) <= -c2 * dphi0:
            return a, f, g, evals
        if d >= 0:
            return zoom(a, a_prev, f, f_prev, d, g)
        if a >= alpha_max:
            # The objective is still decreasing at the largest step we allow (nearly linear valley): accept the capped
            # step (it satisfies sufficient decrease) instead of spending further passes that cannot expand it.
            return a, f, g, evals
        a_prev, f_prev, d_prev, g_prev = a, f, d, g
        a = min(2.0 * a, alpha_max)
    if a_prev > 0 and g_prev is not None and f_prev < f0:
        return a_prev, f_prev, g_prev, evals
    return None


def backtracking(f_of, f0, dphi0, alpha=1.0, c1=1e-4, shrink=0.5, max_iter=10):
    """Armijo backtracking using function values only.  Returns (alpha, f) or None."""
    for _ in range(max_iter):
        f = f_of(alpha)
        if math.isfinite(f) and f <= f0 + c1 * alpha * dphi0:
            return alpha, f
        alpha *= shrink
    return None
