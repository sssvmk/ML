"""
MARS - multivariate adaptive regression splines (ESLII 9.4) - REGRESSION and CLASSIFICATION (0/1 response, squared-error loss).
Model f(X) = b0 + sum_m b_m h_m(X), h_m = products (up to `degree` factors) of reflected hinge pairs (x_j - t)_+ , (t - x_j)_+  (9.18-9.19), fitted by least squares.
Forward pass: greedy addition of the (parent term, variable, knot) pair that most reduces the RSS, using an orthogonalised basis (Fast-MARS style: candidates are screened by
their correlation with the residual, then evaluated exactly); done on <= 15k rows. Backward pass: delete terms one at a time (least RSS increase) on ALL reference rows and
score each size by GCV(lambda) = (RSS/N) / (1 - C(M)/N)^2, C(M) = M + d K  (d = 3 with interactions, 2 additive; K = number of knots).
Hyper-parameters (CV): k top variables, degree (1 additive, 2 interactions), number of terms. Regression selects on CV MSE; classification on CV AUC, and the raw 0/1-coded
scores are turned into probabilities by Platt scaling fitted on the VALIDATION set (log-loss is only reported after that calibration).
Metrics: test MSE +- SE / error +- SE, AUC, GCV, CV error vs terms and degree (one-SE rule), terms, interaction order, variable importance.
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common

NAME = "mars"
KS = [10, 25, 50]
DEGREES = [1, 2]
NTERMS = [3, 5, 7, 9, 11, 15, 19]
MAX_TERMS = 21


def make_grid(cfg, prep, Xs, ys):
    return [{"k": min(k, prep.p), "degree": d, "n_terms": t} for k in KS for d in DEGREES for t in NTERMS]


def complexity(hp):
    return hp["n_terms"] + 0.0001 * hp["k"] + 0.001 * hp["degree"]


def _basis(X, terms):
    B = np.ones((len(X), 1 + len(terms)))
    for i, (par, j, t, sign) in enumerate(terms):
        h = np.maximum(X[:, j] - t, 0.0) if sign > 0 else np.maximum(t - X[:, j], 0.0)
        B[:, i + 1] = B[:, par] * h
    return B


def _forward(Xf, yf, degree, max_terms, nknots=10, shortlist=24):
    n, k = Xf.shape
    knots, Hp, Hn = [], [], []
    for j in range(k):
        u = np.unique(Xf[:, j])
        kn = np.array([u.mean()]) if len(u) == 2 else (np.unique(np.quantile(Xf[:, j], np.linspace(0.05, 0.95, nknots))) if len(u) > 2 else np.array([]))
        knots.append(kn)
        Hp.append(np.maximum(Xf[:, [j]] - kn[None], 0) if len(kn) else None)
        Hn.append(np.maximum(kn[None] - Xf[:, [j]], 0) if len(kn) else None)
    B, deg, vs, terms = [np.ones(n)], [0], [frozenset()], []
    Q = np.ones((n, 1)) / np.sqrt(n)
    r = yf - Q @ (Q.T @ yf)
    yy = float(yf @ yf)
    while len(B) + 2 <= max_terms:
        scores = []
        for m in range(len(B)):
            if deg[m] >= degree:
                continue
            for j in range(k):
                if Hp[j] is None or j in vs[m]:
                    continue
                C, D = B[m][:, None] * Hp[j], B[m][:, None] * Hn[j]
                s = max(np.max(np.abs(C.T @ r) / (np.linalg.norm(C, axis=0) + 1e-12)), np.max(np.abs(D.T @ r) / (np.linalg.norm(D, axis=0) + 1e-12)))
                scores.append((s, m, j))
        if not scores:
            break
        scores.sort(reverse=True)
        best = None
        for _, m, j in scores[:shortlist]:
            C, D = B[m][:, None] * Hp[j], B[m][:, None] * Hn[j]
            Ct, Dt = C - Q @ (Q.T @ C), D - Q @ (Q.T @ D)
            a, b = Ct.T @ r, Dt.T @ r
            aa, bb, ab = np.einsum("ik,ik->k", Ct, Ct) + 1e-12, np.einsum("ik,ik->k", Dt, Dt) + 1e-12, np.einsum("ik,ik->k", Ct, Dt)
            gain = (bb * a * a - 2 * ab * a * b + aa * b * b) / np.maximum(aa * bb - ab ** 2, 1e-18)
            ik = int(np.argmax(gain))
            if best is None or gain[ik] > best[0]:
                best = (float(gain[ik]), m, j, ik)
        if best is None or best[0] <= 1e-9 * yy:
            break
        _, m, j, ik = best
        t = float(knots[j][ik])
        for sign, H in ((1, Hp), (-1, Hn)):
            B.append(B[m] * H[j][:, ik]); deg.append(deg[m] + 1); vs.append(vs[m] | {j}); terms.append((m, j, t, sign))
        for col in (B[-2], B[-1]):
            v = col - Q @ (Q.T @ col)
            nv = np.linalg.norm(v)
            if nv > 1e-8 * np.linalg.norm(col) + 1e-12:
                Q = np.c_[Q, v / nv]
        r = yf - Q @ (Q.T @ yf)
    return terms


def _fit(X, y, degree, seed, max_terms=MAX_TERMS):
    n = len(y)
    rows = np.random.RandomState(seed).choice(n, 15000, replace=False) if n > 15000 else np.arange(n)
    terms = _forward(X[rows], y[rows], degree, max_terms)
    B = _basis(X, terms)
    G, c, yy = B.T @ B, B.T @ y, float(y @ y)
    d = 3.0 if degree > 1 else 2.0

    def rss(S):
        S = list(S)
        return yy - c[S] @ np.linalg.solve(G[np.ix_(S, S)] + 1e-9 * np.eye(len(S)), c[S])

    def n_knots(S):
        return len({(terms[s - 1][1], round(terms[s - 1][2], 12)) for s in S if s > 0})
    S = list(range(B.shape[1]))
    seq = {len(S): (list(S), rss(S))}
    while len(S) > 1:
        cand = [(rss([s for s in S if s != p]), [s for s in S if s != p]) for p in S[1:]]
        r_, S = min(cand, key=lambda z: z[0])
        seq[len(S)] = (list(S), r_)
    gcv = {m: (v[1] / n) / (1 - (m + d * n_knots(v[0])) / n) ** 2 for m, v in seq.items()}
    coefs = {m: np.linalg.solve(G[np.ix_(v[0], v[0])] + 1e-9 * np.eye(len(v[0])), c[v[0]]) for m, v in seq.items()}
    return {"terms": terms, "seq": seq, "gcv": gcv, "coefs": coefs, "G": G, "c": c, "yy": yy, "rss": rss, "n": n}


def _predict(model, X, size):
    size = max(s for s in model["seq"] if s <= size)
    S = model["seq"][size][0]
    return _basis(X, model["terms"])[:, S] @ model["coefs"][size]


def predict_path(Xref, yref, grid, Xq, cfg, ctx):
    order = ctx["order"]
    out = np.empty((len(grid), len(Xq)))
    for (k, d) in sorted({(hp["k"], hp["degree"]) for hp in grid}):
        cols = order[:k]
        Xk, Xqk = np.asarray(Xref[:, cols], float), np.asarray(Xq[:, cols], float)
        m = _fit(Xk, yref.astype(float), d, cfg.seed)
        for h, hp in enumerate(grid):
            if (hp["k"], hp["degree"]) == (k, d):
                out[h] = _predict(m, Xqk, hp["n_terms"])
                if ctx.get("final"):
                    best = min(m["gcv"], key=m["gcv"].get)
                    ctx["store"].update({"model": m, "cols": cols, "hp": hp})
                    ctx["store"]["summary"] = {"terms_forward": len(m["terms"]) + 1, "terms_selected": hp["n_terms"], "gcv_best_terms": int(best), "gcv_at_best": float(m["gcv"][best]),
                                               "gcv_at_selected": float(m["gcv"][max(s for s in m["gcv"] if s <= hp["n_terms"])])}
    return out


def extra(ctx):
    S = ctx["store"]
    m, cols, hp = S["model"], S["cols"], S["hp"]
    names = np.array(ctx["prep"].feature_names)[cols]
    size = max(s for s in m["seq"] if s <= hp["n_terms"])
    sel = m["seq"][size][0]
    full_rss = m["seq"][size][1]
    terms = m["terms"]
    imp = {}
    for j in range(len(cols)):
        S_j = [s for s in sel if s == 0 or not _involves(terms, s, j)]
        if len(S_j) < len(sel):
            imp[names[j]] = m["rss"](S_j) - full_rss
    df = pd.DataFrame({"variable": list(imp), "rss_increase_when_removed": list(imp.values())})
    if len(df):
        df["importance_0_100"] = 100 * df["rss_increase_when_removed"] / df["rss_increase_when_removed"].max()
        df = df.sort_values("importance_0_100", ascending=False)
    df.to_csv(ctx["out"] / "variable_importance.csv", index=False)
    rows = []
    for s in sel[1:]:
        chain, p = [], s
        while p > 0:
            par, j, t, sign = terms[p - 1]
            chain.append(f"{'(' + names[j] + f' - {t:.3g})+' if sign > 0 else f'({t:.3g} - ' + names[j] + ')+'}")
            p = par
        rows.append({"term": s, "interaction_order": len(chain), "basis": " * ".join(chain[::-1])})
    pd.DataFrame(rows).to_csv(ctx["out"] / "mars_terms.csv", index=False)
    ctx["summary"]["max_interaction_order"] = int(max([r["interaction_order"] for r in rows], default=0))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ms = sorted(m["gcv"]); ax.plot(ms, [m["gcv"][s] for s in ms], "o-"); ax.axvline(size, color="purple", ls="--", label=f"selected ({size} terms)")
    ax.set_xlabel("number of terms"); ax.set_ylabel("GCV"); ax.legend(); ax.set_title("MARS: GCV along the backward sequence")
    fig.tight_layout(); fig.savefig(ctx["out"] / "mars_gcv.png", dpi=120); plt.close(fig)


def _involves(terms, s, j):
    while s > 0:
        par, v, t, sign = terms[s - 1]
        if v == j:
            return True
        s = par
    return False


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=predict_path, complexity=complexity, extra=extra,
                             select_by="mse" if task == "regression" else "auc", platt=task == "classification", scores_are_probs=task == "regression",
                             notes="Own numpy MARS; forward pass on <= 15k rows, backward pass + coefficients on all reference rows.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
