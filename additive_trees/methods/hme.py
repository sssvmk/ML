"""
Hierarchical mixture of experts (ESLII 9.5) - REGRESSION and CLASSIFICATION.
A tree of depth D with K-way SOFT gating networks g(x) = softmax(gamma' x) (9.25-9.26); the experts at the K^D leaves are
  regression     : Gaussian linear models  y ~ N(beta' x, sigma^2)                      (9.28)  -> objective = negative Gaussian-mixture log-likelihood (9.30)
  classification : linear logistic models  Pr(y = 1 | x) = 1 / (1 + exp(-theta' x))      (9.29)  -> objective = negative Bernoulli-mixture log-likelihood
Fitted by EM: E-step = leaf responsibilities; M-step = weighted least squares / weighted logistic regression for the experts and a softmax regression (L-BFGS) with the
expected branch probabilities as soft targets for every gate. Initialised from k-means; EM restarts (final fit) keep the best likelihood.
Hyper-parameters (CV): k top variables and the topology (depth D, branching K). Regression selects on CV MSE, classification on CV log-loss (one-SE rule on the number of experts).
EM is run on <= 15k rows in the CV folds and <= 60k rows in the final fit (cost control).
Metrics: test MSE +- SE (regression) or log-loss/error/AUC/calibration (classification); held-out log-likelihood; CV error vs depth and K; EM convergence and restarts; gating maps.
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import common

NAME = "hme"
KS = [10, 20]
TOPOLOGIES = [(1, 2), (2, 2), (1, 4), (2, 3)]        # (depth D, branching K)


def make_grid(cfg, prep, Xs, ys):
    return [{"k": min(k, prep.p), "depth": D, "K": K} for k in KS for D, K in TOPOLOGIES]


def complexity(hp):
    return hp["K"] ** hp["depth"] + 0.001 * hp["k"]


def _paths(D, K):
    off = np.cumsum([0] + [K ** d for d in range(D)])
    paths = []
    for l in range(K ** D):
        paths.append([(int(off[d] + l // K ** (D - d)), (l // K ** (D - 1 - d)) % K) for d in range(D)])
    return paths, int(off[-1])


def _gate_logp(Xa, Wg, paths, D, K, n_nodes):
    lg = []
    for v in range(n_nodes):
        Z = Xa @ Wg[v]
        Z = Z - Z.max(1, keepdims=True)
        lg.append(Z - np.log(np.exp(Z).sum(1, keepdims=True)))
    out = np.zeros((len(Xa), len(paths)))
    for l, path in enumerate(paths):
        for node, ch in path:
            out[:, l] += lg[node][:, ch]
    return out


def _softmax_fit(Xa, T, W0, ridge, iters=25):
    p1, K = Xa.shape[1], T.shape[1]
    w = T.sum(1)

    def f(wf):
        W = wf.reshape(p1, K)
        Z = Xa @ W
        Z -= Z.max(1, keepdims=True)
        S = np.exp(Z)
        S /= S.sum(1, keepdims=True)
        return -float((T * np.log(S + 1e-12)).sum()) + 0.5 * ridge * float((W ** 2).sum()), (Xa.T @ (S * w[:, None] - T) + ridge * W).ravel()
    return minimize(f, W0.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": iters}).x.reshape(p1, K)


def _kmeans(X, L, seed, iters=10):
    rng = np.random.RandomState(seed)
    C = X[rng.choice(len(X), L, replace=False)].copy()
    for _ in range(iters):
        lab = np.argmin((X ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None] - 2 * X @ C.T, axis=1)
        for j in range(L):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return lab


def _expert_fit(Xa, y, r, task, prev):
    s = r.sum() + 1e-9
    ridge = 1e-3 * s
    if task == "regression":
        Xw = Xa * r[:, None]
        beta = np.linalg.solve(Xw.T @ Xa + ridge * np.eye(Xa.shape[1]), Xw.T @ y)
        res = y - Xa @ beta
        return beta, max(float((r * res ** 2).sum() / s), 1e-3 * float(np.var(y)))
    th = prev if prev is not None else np.zeros(Xa.shape[1])
    for _ in range(3):
        eta = np.clip(Xa @ th, -15, 15)
        p = common.sigmoid(eta)
        v = np.clip(p * (1 - p), 1e-6, None)
        wt = r * v
        A = (Xa * wt[:, None]).T @ Xa
        A += 1e-2 * np.trace(A) / A.shape[0] * np.eye(A.shape[0]) + 1e-8 * np.eye(A.shape[0])
        th = np.linalg.solve(A, (Xa * wt[:, None]).T @ (eta + (y - p) / v))
    return th, 1.0


def _leaf_loglik(Xa, y, E, task):
    if task == "regression":
        mu = np.stack([Xa @ b for b, _ in E], 1)
        s2 = np.array([s for _, s in E])
        return -0.5 * np.log(2 * np.pi * s2)[None] - 0.5 * (y[:, None] - mu) ** 2 / s2[None], mu
    p = common.sigmoid(np.clip(np.stack([Xa @ b for b, _ in E], 1), -15, 15))
    return y[:, None] * np.log(np.clip(p, 1e-9, 1)) + (1 - y[:, None]) * np.log(np.clip(1 - p, 1e-9, 1)), p


def _em(Xa, y, task, D, K, iters, seed):
    n, p1 = Xa.shape
    L = K ** D
    paths, n_nodes = _paths(D, K)
    rng = np.random.RandomState(seed)
    ys_ = (y - y.mean()) / (y.std() + 1e-9)
    lab = _kmeans(np.c_[Xa[:, 1:], 1.5 * ys_], L, seed)      # initialise from k-means on (x, y): regimes show up in the joint space
    resp = 0.8 * np.eye(L)[lab] + 0.2 / L
    Wg = [0.01 * rng.randn(p1, K) for _ in range(n_nodes)]
    E = [None] * L
    hist = []
    prev_ll = -np.inf
    for it in range(iters):
        E = [_expert_fit(Xa, y, resp[:, l], task, E[l][0] if (E[l] is not None and task != "regression") else None) for l in range(L)]
        level_of = lambda v: next(d for d in range(D) if v < sum(K ** q for q in range(d + 1)))
        for v in range(n_nodes):
            d = level_of(v)
            pre = v - sum(K ** q for q in range(d))
            T = np.zeros((n, K))
            for l, path in enumerate(paths):
                node, ch = path[d]
                if node == v:
                    T[:, ch] += resp[:, l]
            Wg[v] = _softmax_fit(Xa, T, Wg[v], 1e-3 * T.sum() + 1e-6)
        ll, _ = _leaf_loglik(Xa, y, E, task)
        lg = _gate_logp(Xa, Wg, paths, D, K, n_nodes)
        a = ll + lg
        m = a.max(1, keepdims=True)
        tot = float((m[:, 0] + np.log(np.exp(a - m).sum(1))).sum())
        resp = np.exp(a - m)
        resp /= resp.sum(1, keepdims=True)
        hist.append(tot / n)
        if abs(tot - prev_ll) < 1e-5 * abs(tot):
            break
        prev_ll = tot
    return {"E": E, "Wg": Wg, "paths": paths, "n_nodes": n_nodes, "D": D, "K": K, "hist": hist, "ll": hist[-1]}


def _prep_X(X, mu, sd):
    return np.c_[np.ones(len(X)), np.clip((X - mu) / sd, -6, 6)]


def _predict(m, Xa, task, y=None):
    lg = _gate_logp(Xa, m["Wg"], m["paths"], m["D"], m["K"], m["n_nodes"])
    g = np.exp(lg)
    if task == "regression":
        mu = np.stack([Xa @ b for b, _ in m["E"]], 1)
    else:
        mu = common.sigmoid(np.clip(np.stack([Xa @ b for b, _ in m["E"]], 1), -15, 15))
    pred = (g * mu).sum(1)
    hl = None
    if y is not None:
        ll, _ = _leaf_loglik(Xa, y, m["E"], task)
        a = ll + lg
        mx = a.max(1, keepdims=True)
        hl = float(np.mean(mx[:, 0] + np.log(np.exp(a - mx).sum(1))))
    return pred, hl, g


def predict_path(Xref, yref, grid, Xq, cfg, ctx):
    task, order, final = ctx["task"], ctx["order"], ctx.get("final", False)
    out = np.empty((len(grid), len(Xq)))
    hl = np.full(len(grid), np.nan)
    rng = np.random.RandomState(cfg.seed)
    cap = 60000 if final else 15000
    for h, hp in enumerate(grid):
        cols = order[:hp["k"]]
        rows = rng.choice(len(yref), cap, replace=False) if len(yref) > cap else np.arange(len(yref))
        Xk = np.asarray(Xref[:, cols], float)
        mu, sd = Xk[rows].mean(0), Xk[rows].std(0) + 1e-9
        Xa = _prep_X(Xk[rows], mu, sd)
        y = yref[rows].astype(float)
        fits = [_em(Xa, y, task, hp["depth"], hp["K"], 25 if final else 15, cfg.seed + 11 * r) for r in range(cfg.hme_restarts if final else 1)]
        m = max(fits, key=lambda f: f["ll"])
        Xqa = _prep_X(np.asarray(Xq[:, cols], float), mu, sd)
        yq = ctx.get("yq")
        pred, hl_h, g = _predict(m, Xqa, task, None if yq is None or len(yq) != len(Xq) else yq.astype(float))
        out[h] = pred
        hl[h] = hl_h if hl_h is not None else np.nan
        if final:
            ctx["store"].update({"model": m, "fits": fits, "cols": cols, "mu": mu, "sd": sd, "hp": hp, "gates_test": g[ctx["n_val"]:], "task": task})
            ctx["store"]["summary"] = {"n_experts": hp["K"] ** hp["depth"], "em_iterations": [len(f["hist"]) for f in fits], "restart_loglik_per_row": [float(f["ll"]) for f in fits],
                                       "restart_spread": float(max(f["ll"] for f in fits) - min(f["ll"] for f in fits)), "rows_used_for_EM": int(len(rows))}
    ctx["metrics"]["heldout_loglik"] = hl
    return out


def extra(ctx):
    S = ctx["store"]
    fits, m, cols, task = S["fits"], S["model"], S["cols"], S["task"]
    names = np.array(ctx["prep"].feature_names)[cols]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, f in enumerate(fits):
        ax.plot(range(1, len(f["hist"]) + 1), f["hist"], "o-", ms=3, label=f"restart {i + 1} (final {f['ll']:.4f})")
    ax.set_xlabel("EM iteration"); ax.set_ylabel("log-likelihood per row"); ax.legend(fontsize=8); ax.set_title("HME: EM convergence and restarts (final fit)")
    fig.tight_layout(); fig.savefig(ctx["out"] / "em_convergence.png", dpi=120); plt.close(fig)
    g = S["gates_test"]
    pd.DataFrame({"expert": np.arange(g.shape[1]), "mean_gate_prob_test": g.mean(0), "share_of_rows_where_dominant": np.bincount(g.argmax(1), minlength=g.shape[1]) / len(g)}).to_csv(ctx["out"] / "gating_summary.csv", index=False)
    # dominant expert over the two top variables (others at their mean)
    X = np.asarray(ctx["X_tr"][::max(1, len(ctx["y_tr"]) // 20000)][:, cols], float)
    a, b = 0, 1 if len(cols) > 1 else 0
    ga, gb = np.linspace(*np.quantile(X[:, a], [0.02, 0.98]), 70), np.linspace(*np.quantile(X[:, b], [0.02, 0.98]), 70)
    G1, G2 = np.meshgrid(ga, gb)
    pts = np.tile(S["mu"], (G1.size, 1)); pts[:, a], pts[:, b] = G1.ravel(), G2.ravel()
    _, _, gg = _predict(m, _prep_X(pts, S["mu"], S["sd"]), task)
    pr, _, _ = _predict(m, _prep_X(pts, S["mu"], S["sd"]), task)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    im = ax[0].pcolormesh(G1, G2, gg.argmax(1).reshape(G1.shape), cmap="tab10"); ax[0].set_title("dominant expert"); fig.colorbar(im, ax=ax[0])
    im = ax[1].pcolormesh(G1, G2, pr.reshape(G1.shape), cmap="viridis"); ax[1].set_title("fitted " + ("mean" if task == "regression" else "probability")); fig.colorbar(im, ax=ax[1])
    for q in ax:
        q.set_xlabel(names[a]); q.set_ylabel(names[b])
    fig.suptitle("HME gating map (other variables at their mean)"); fig.tight_layout(); fig.savefig(ctx["out"] / "gating_map.png", dpi=120); plt.close(fig)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=predict_path, complexity=complexity, extra=extra,
                             select_by="mse" if task == "regression" else "log_loss", notes="EM on capped rows; k-means init; soft gating by softmax regression.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
