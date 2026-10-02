"""
Generalized additive models with BACKFITTING (ESLII 9.1) - REGRESSION and CLASSIFICATION.
regression     : penalised RSS  sum (y - a - sum_j f_j(x_j))^2 + sum_j lambda_j int f_j''^2 ; backfitting (Algorithm 9.1) = Gauss-Seidel on the block system, each f_j a
                 penalised cubic B-spline smoother (lambda_j set through the per-term df), centred.  Binary / 3-level columns enter as linear terms.
classification : additive LOGISTIC model, penalised negative binomial log-likelihood; fitted by LOCAL SCORING (Algorithm 9.2): IRLS whose weighted least-squares step is solved by backfitting.
Hyper-parameters (CV): k = number of top-ranked variables, df = degrees of freedom of every smooth term.   df_total ~ 1 + sum_j (df_j - 1).
Metrics: regression: test MSE +- SE, CV MSE vs (k, df) (one-SE rule on df_total), GCV, partial-dependence curves with approximate pointwise SE bands, backfitting convergence.
         classification: test log-loss +- SE, error +- SE, AUC +- SE, calibration, CV deviance vs df (selection by CV log-loss), partial-dependence curves (logit scale).
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve

import common
from bases import bs_design, bs_knots, bs_penalty, df_to_lambda, quantile_knots

NAME = "gam"
KS = [5, 10, 20, 40]
DFS = [2.5, 3, 4, 6]
M_INT = 8


def make_grid(cfg, prep, Xs, ys):
    ks = [k for k in KS if k <= prep.p] or [prep.p]
    return [{"k": k, "df": d} for k in ks for d in DFS]


class _Term:
    def __init__(self, x, dtype):
        self.nu = len(np.unique(x))
        self.linear = self.nu <= 3
        if self.linear:
            self.mu, self.sd = float(x.mean()), float(x.std()) or 1.0
            self.Om = np.zeros((1, 1))
        else:
            t = bs_knots(float(x.min()), float(x.max()), quantile_knots(x, min(M_INT, max(2, self.nu // 4))))
            self.t, self.Om = t, bs_penalty(t)
        self.B = self.basis(x).astype(dtype)

    def basis(self, x):
        x = np.asarray(x, float)
        return ((x - self.mu) / self.sd)[:, None] if self.linear else bs_design(x, self.t)


def _lams(terms, dfs):
    out = []
    for t in terms:
        if t.linear:
            out.append({d: 0.0 for d in dfs})
        else:
            G = (t.B.T @ t.B).astype(float)
            l, _ = df_to_lambda(G, t.Om, [min(d, t.B.shape[1] - 0.6) for d in dfs])
            out.append(dict(zip(dfs, l)))
    return out


class _Fit:
    pass


def _backfit(terms, lam, y, W, alpha, f, cyc_max, tol, sd_y):
    """Gauss-Seidel / backfitting (weights W or None). Updates f (k,n) in place; returns thetas, centres, history, per-term A matrices."""
    k, n = f.shape
    total = f.sum(0)
    th, cen, mats, hist = [None] * k, [0.0] * k, [None] * k, []
    for j, t in enumerate(terms):
        B = t.B.astype(np.float64) if t.B.dtype != np.float64 else t.B
        G = (B.T @ B) if W is None else (B * W[:, None]).T @ B
        A = G + lam[j] * t.Om
        A = A + 1e-10 * np.trace(A) / A.shape[0] * np.eye(A.shape[0])
        mats[j] = (G, cho_factor(A))
    for cyc in range(cyc_max):
        delta = 0.0
        for j, t in enumerate(terms):
            B = t.B
            r = y - alpha - total + f[j]
            rhs = B.T @ (r if W is None else W * r)
            th[j] = cho_solve(mats[j][1], rhs.astype(np.float64))
            fj = B @ th[j]
            c = fj.mean() if W is None else float((W * fj).sum() / W.sum())
            fj = fj - c
            cen[j] = c
            delta = max(delta, float(np.mean(np.abs(fj - f[j]))))
            total += fj - f[j]
            f[j] = fj
        hist.append(delta / sd_y)
        if delta / sd_y < tol:
            break
    return th, cen, mats, hist


def _fit_gam(terms, lam, y, task, cfg):
    n, k = len(y), len(terms)
    f = np.zeros((k, n))
    hist = []
    if task == "regression":
        alpha = float(y.mean())
        th, cen, mats, hist = _backfit(terms, lam, y, None, alpha, f, 60, 1e-5, float(y.std()))
        W = None
        rss = float(np.sum((y - alpha - f.sum(0)) ** 2))
    else:
        pb = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        alpha, eta, prev = np.log(pb / (1 - pb)), np.full(n, np.log(pb / (1 - pb))), np.inf
        for outer in range(12):
            p = common.sigmoid(eta)
            W = np.clip(p * (1 - p), 1e-5, None)
            z = eta + (y - p) / W
            alpha = float((W * (z - f.sum(0))).sum() / W.sum())
            th, cen, mats, h = _backfit(terms, lam, z, W, alpha, f, 8, 1e-4, float(z.std()))
            hist += h
            eta = alpha + f.sum(0)
            dev = float(-2 * np.sum(y * eta - np.logaddexp(0, eta)))
            if abs(prev - dev) / max(dev, 1e-9) < 1e-6:
                break
            prev = dev
        rss = float(dev)
    dfj = []
    for j, t in enumerate(terms):
        G, cf = mats[j]
        dfj.append(2.0 if t.linear else float(np.trace(cho_solve(cf, G))))
    m = _Fit()
    m.alpha, m.th, m.cen, m.mats, m.hist, m.dfj, m.rss, m.W, m.f = alpha, th, cen, mats, hist, dfj, rss, W, f
    m.df_total = float(1 + sum(d - 1 for d in dfj))
    m.sigma2 = rss / max(n - m.df_total, 1.0) if task == "regression" else 1.0
    return m


def _predict(m, terms, Xqk, task):
    eta = np.full(len(Xqk), m.alpha)
    for j, t in enumerate(terms):
        eta += t.basis(Xqk[:, j]) @ m.th[j] - m.cen[j]
    return eta if task == "regression" else common.sigmoid(eta)


def predict_path(Xref, yref, grid, Xq, cfg, ctx):
    task, order = ctx["task"], ctx["order"]
    out = np.empty((len(grid), len(Xq)))
    dtype = np.float64 if len(yref) <= 60000 else np.float32
    for k in sorted({hp["k"] for hp in grid}):
        cols = order[:k]
        Xk, Xqk = np.asarray(Xref[:, cols], float), np.asarray(Xq[:, cols], float)
        terms = [_Term(Xk[:, j], dtype) for j in range(k)]
        dfs = sorted({hp["df"] for hp in grid if hp["k"] == k})
        lam = _lams(terms, dfs)
        for h, hp in enumerate(grid):
            if hp["k"] != k:
                continue
            m = _fit_gam(terms, [l[hp["df"]] for l in lam], yref, task, cfg)
            out[h] = _predict(m, terms, Xqk, task)
            if ctx.get("final"):
                ctx["store"].update({"model": m, "terms": terms, "cols": cols, "Xk": Xk, "task": task, "lam": [l[hp["df"]] for l in lam]})
                ctx["store"]["summary"] = {"df_total": m.df_total, "backfitting_cycles": len(m.hist), "final_change": float(m.hist[-1]), "n_linear_terms": int(sum(t.linear for t in terms))}
    return out


def diagnostics(Xs, ys, grid, cfg, order):
    task = "regression" if not np.all((ys == 0) | (ys == 1)) else "classification"
    df, gcv, cyc = [], [], []
    for k in sorted({hp["k"] for hp in grid}):
        cols = order[:k]
        Xk = np.asarray(Xs[:, cols], float)
        terms = [_Term(Xk[:, j], np.float64) for j in range(k)]
        dfs = sorted({hp["df"] for hp in grid if hp["k"] == k})
        lam = _lams(terms, dfs)
        res = {}
        for d in dfs:
            m = _fit_gam(terms, [l[d] for l in lam], ys, task, cfg)
            n = len(ys)
            res[d] = (m.df_total, (m.rss / n) / (1 - m.df_total / n) ** 2 if task == "regression" else np.nan, len(m.hist))
        for hp in grid:
            if hp["k"] == k:
                df.append(res[hp["df"]][0]); gcv.append(res[hp["df"]][1]); cyc.append(res[hp["df"]][2])
    # reorder to grid order
    order_idx = [i for k in sorted({hp["k"] for hp in grid}) for i, hp in enumerate(grid) if hp["k"] == k]
    inv = np.argsort(order_idx)
    return {"complexity": np.array(df)[inv], "df_total": np.array(df)[inv], "gcv": np.array(gcv)[inv], "backfit_cycles": np.array(cyc)[inv]}


def extra(ctx):
    S = ctx["store"]
    m, terms, cols, Xk, task = S["model"], S["terms"], S["cols"], S["Xk"], S["task"]
    names = np.array(ctx["prep"].feature_names)[cols]
    sd = np.array([f.std() for f in m.f])
    imp = pd.DataFrame({"variable": names, "sd_of_f_j": sd, "df_j": m.dfj, "linear_term": [t.linear for t in terms]}).sort_values("sd_of_f_j", ascending=False)
    imp.to_csv(ctx["out"] / "term_importance.csv", index=False)
    top = list(np.argsort(-sd)[:6])
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for a, j in zip(axes.ravel(), top):
        t, x = terms[j], Xk[:, j]
        g = np.linspace(*np.quantile(x, [0.01, 0.99]), 100)
        B = t.basis(g)
        fh = B @ m.th[j] - m.cen[j]
        G, cf = m.mats[j]
        Ai = cho_solve(cf, np.eye(G.shape[0]))
        se = np.sqrt(np.maximum(m.sigma2 * np.sum((B @ (Ai @ G @ Ai)) * B, axis=1), 0))
        a.plot(g, fh, "r-"); a.fill_between(g, fh - 2 * se, fh + 2 * se, color="red", alpha=0.2)
        a.plot(x[::max(1, len(x) // 400)], np.full(len(x[::max(1, len(x) // 400)]), a.get_ylim()[0]), "|", color="k", alpha=0.3)
        a.set_title(f"{names[j]}  (df {m.dfj[j]:.1f})", fontsize=9); a.set_ylabel("f_j" + (" (logit)" if task == "classification" else ""))
        rows += [{"variable": names[j], "x": float(xx), "f": float(ff), "se": float(ss)} for xx, ff, ss in zip(g, fh, se)]
    fig.suptitle("GAM partial-dependence curves +-2 approximate pointwise SE (conditional on the other terms)")
    fig.tight_layout(); fig.savefig(ctx["out"] / "partial_dependence.png", dpi=120); plt.close(fig)
    pd.DataFrame(rows).to_csv(ctx["out"] / "partial_dependence.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(m.hist, "o-", ms=3); ax.set_xlabel("backfitting cycle"); ax.set_ylabel("max mean |change in f_j| / sd"); ax.set_title("Backfitting convergence (final fit)")
    fig.tight_layout(); fig.savefig(ctx["out"] / "backfitting_convergence.png", dpi=120); plt.close(fig)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=predict_path, diagnostics=diagnostics, extra=extra,
                             select_by="mse" if task == "regression" else "log_loss", notes=f"Penalised cubic B-spline smoothers ({M_INT} interior knots), df-parameterised.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
