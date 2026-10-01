"""
Best-subset selection.  Objective: minimise RSS over all subsets of size k.
Metrics: 10-fold CV MSE vs subset size k (one-SE rule), AIC / BIC / Cp.

Because exhaustive search is infeasible on hundreds of engineered columns, the search runs on a candidate
POOL of `cfg.subset_pool` columns picked by greedy forward selection. The pool is re-selected INSIDE every
CV fold and in every refit (ESLII Sec. 7.10.2: selection must be part of the cross-validated procedure).
  * pool <= 22 : exact search over all 2^m subsets (Gray-code enumeration with the sweep operator)
  * pool  > 22 : forward-stepwise start + exchange (swap) refinement  (heuristic, flagged in the notes)
"""
import _bootstrap  # noqa: F401
import numpy as np

import common
from common import Path_, forward_pool, info_criteria, sigma2_full

NAME = "best_subset"


def _exact_masks(Gs, cs, yy, n):
    """Exhaustive best-subset RSS per size using Gray-code sweeps. Returns bitmask per size (0..m)."""
    m = len(cs)
    A = np.empty((m + 1, m + 1))
    A[:m, :m] = Gs / n
    A[:m, m] = A[m, :m] = cs / n
    A[m, m] = yy / n
    best_rss = np.full(m + 1, np.inf)
    best_mask = np.zeros(m + 1, dtype=np.int64)
    best_rss[0] = A[m, m]
    mask, size = 0, 0
    for i in range(1, 1 << m):
        k = (i & -i).bit_length() - 1
        d = A[k, k]
        rowk = A[k, :] / d
        colk = A[:, k].copy()
        A -= np.outer(colk, rowk)
        A[k, :] = rowk
        A[:, k] = -colk / d
        A[k, k] = 1.0 / d
        mask ^= (1 << k)
        size += 1 if (mask >> k) & 1 else -1
        r = A[m, m]
        if r < best_rss[size]:
            best_rss[size], best_mask[size] = r, mask
    return best_mask


def _rss_subset(Gs, cs, yy, S):
    if len(S) == 0:
        return yy
    Gss = Gs[np.ix_(S, S)]
    return yy - cs[S] @ np.linalg.solve(Gss, cs[S])


def _swap_masks(Gs, cs, yy, n):
    """Heuristic: forward-stepwise subsets refined by single exchanges."""
    m = len(cs)
    order = forward_pool(Gs, cs, m)
    masks = np.zeros(m + 1, dtype=np.int64)
    for k in range(1, len(order) + 1):
        S = list(order[:k])
        best = _rss_subset(Gs, cs, yy, S)
        improved = True
        while improved:
            improved = False
            for a in range(k):
                for j in range(m):
                    if j in S:
                        continue
                    T = S.copy(); T[a] = j
                    r = _rss_subset(Gs, cs, yy, T)
                    if r < best - 1e-12 * abs(best):
                        S, best, improved = T, r, True
        masks[k] = sum(1 << j for j in S)
    return masks


def make_grid(bundle, prep, cfg):
    return {"pool": int(cfg.subset_pool)}


def fit_path(st, grid, cfg):
    Gc, cc, yyc, xbar, ybar = st.centered()
    p, n = len(cc), st.n
    pool = forward_pool(Gc, cc, grid["pool"])
    m = len(pool)
    Gs, cs = Gc[np.ix_(pool, pool)], cc[pool]
    masks = _exact_masks(Gs, cs, yyc, n) if m <= 22 else _swap_masks(Gs, cs, yyc, n)
    B = np.zeros((p, m + 1))
    for k in range(1, m + 1):
        S = [j for j in range(m) if (masks[k] >> j) & 1]
        if S:
            B[pool[S], k] = np.linalg.solve(Gs[np.ix_(S, S)], cs[S])
    return Path_(B, ybar - xbar @ B, np.arange(m + 1, dtype=float), pool=np.full(m + 1, m))


def extra_curve(curve, path, bundle):
    n, s2 = bundle.tr_fit.n, sigma2_full(bundle.tr_fit)
    ic = info_criteria(curve["train_rss"].values, n, path.comp + 1, s2)
    for k, v in ic.items():
        curve[k] = v
    return curve


def final_extras(beta, b0, st, bundle):
    k = int((np.abs(beta) > 1e-12).sum())
    rss = st.sse(beta, b0)
    ic = info_criteria(rss, st.n, k + 1, sigma2_full(st))
    return {"k": k, "training_rss": rss, "AIC": float(ic["aic"]), "BIC": float(ic["bic"]), "Cp": float(ic["cp"])}


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    mode = "exact (Gray-code sweep)" if cfg.subset_pool <= 22 else "heuristic (forward + swap)"
    return common.run_path_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, fit_path=fit_path,
                                  comp_label="subset size k", extra_curve=extra_curve, final_extras=final_extras,
                                  notes=f"Candidate pool={cfg.subset_pool} columns chosen by forward selection inside each fold; "
                                        f"search mode: {mode}.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
