"""
Discriminant adaptive nearest-neighbour, DANN (ESLII 13.4) - CLASSIFICATION. scikit-learn NearestNeighbors for all neighbour searches; the local metric is the only hand-written step (as agreed).
At each query x0: take its K_M nearest neighbours, form the local within-class matrix W and between-class matrix B, and use the metric
      Sigma = W^(-1/2) [ W^(-1/2) B W^(-1/2) + eps I ] W^(-1/2)        (13.8)
which stretches neighbourhoods along directions that do not discriminate and shrinks them along those that do; then classify by the k nearest neighbours of x0 under d(x, x0) = (x - x0)' Sigma (x - x0).
Practical approximation: the final k neighbours are chosen among the 400 Euclidean-nearest candidates of x0 (not all training rows). If the K_M neighbours contain one class only, B = 0 and the metric falls back to Euclidean.
Scores: distance-weighted vote fractions (smoothed for log-loss). Hyper-parameters (CV): k, K_M, eps, number of top-ranked variables; random search; CV selects on AUC.
Metrics: test error +- SE and AUC (compared with k-NN and LVQ in the comparison table), CV error over (k, K_M, eps), and the GLOBAL dimension-reduction variant (13.4.2): the averaged between matrix B-bar (13.10),
subspace dimension chosen on the validation AUC of a k-NN rule in the leading eigen-directions.
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

import common
import nnlib

NAME = "dann"
SPACE = {"k": [3, 5, 8, 12, 20], "KM": [25, 50, 100], "eps": [1.0, 0.1]}
NN = [5, 15, 30, 60]
CAND = 400


def local_metric(Xn, yn, eps):
    """Sigma of (13.8) from a neighbourhood (Xn, yn); None (-> Euclidean) when a class is missing."""
    n, p = Xn.shape
    if min((yn == 0).sum(), (yn == 1).sum()) == 0:
        return None
    m = Xn.mean(0)
    W, B = np.zeros((p, p)), np.zeros((p, p))
    for c in (0, 1):
        Xc = Xn[yn == c]
        mc = Xc.mean(0)
        D = Xc - mc
        W += D.T @ D
        B += (len(Xc) / n) * np.outer(mc - m, mc - m)
    W = W / n + 1e-6 * np.trace(W / n) / p * np.eye(p)
    w, V = np.linalg.eigh(W)
    Wm = (V * w ** -0.5) @ V.T
    return Wm @ (Wm @ B @ Wm + eps * np.eye(p)) @ Wm


def make_grid(cfg, prep, Xs, ys):
    base = [{"k": k, "KM": km, "eps": e} for k in SPACE["k"] if k <= prep.p for km in SPACE["KM"] for e in SPACE["eps"]]
    chosen = [base[c["i"]] for c in nnlib.sample_configs({"i": list(range(len(base)))}, max(1, cfg.n_configs_nn // 2), cfg.seed)]
    return [{**c, "nn": nn} for c in chosen for nn in NN]


def complexity(hp):
    return 1000.0 / hp["nn"] + hp["k"] + 0.01 * hp["KM"]


def predict_path(Xref, yref, grid, Xq, cfg, ctx):
    order = ctx["order"]
    out = np.empty((len(grid), len(Xq)))
    y = np.asarray(yref, float)
    base = float(y.mean())
    for kk in sorted({hp["k"] for hp in grid}):
        cols = order[:kk]
        Xr, Xqk = np.asarray(Xref[:, cols], np.float64), np.asarray(Xq[:, cols], np.float64)
        kc = min(CAND, len(Xr))
        nbrs = NearestNeighbors(n_neighbors=kc, n_jobs=cfg.n_jobs).fit(Xr)
        by_metric = {}
        for h, hp in enumerate(grid):
            if hp["k"] == kk:
                by_metric.setdefault((hp["KM"], hp["eps"]), []).append(h)
        for a in range(0, len(Xqk), 2000):
            _, ind = nbrs.kneighbors(Xqk[a:a + 2000])
            for i in range(len(ind)):
                C, yc = Xr[ind[i]], y[ind[i]]
                diff = C - Xqk[a + i]
                for (KM, eps), hs in by_metric.items():
                    S = local_metric(C[:KM], yc[:KM], eps)
                    d2 = (diff ** 2).sum(1) if S is None else np.einsum("ij,jk,ik->i", diff, S, diff)
                    o = np.argsort(d2)
                    for h in hs:
                        nn = grid[h]["nn"]
                        sel = o[:nn]
                        d = np.sqrt(np.maximum(d2[sel], 0.0))
                        out[h, a + i] = nnlib.knn_scores(d[None, :], yc[sel][None, :], min(nn, len(sel)), "distance", "classification", base)[0]
        if ctx.get("final"):
            ctx["store"].update(Xr=Xr, yr=y, cols=cols, hp=grid[by_metric[next(iter(by_metric))][0]])
            ctx["store"]["summary"] = {"n_reference_rows": int(len(Xr)), "candidate_pool": kc}
    return out


def extra(ctx):
    S = ctx["store"]
    cols, hp = S["cols"], S["hp"]
    Xs, ys = np.asarray(ctx["Xs"][:, cols], np.float64), ctx["ys_ev"].astype(int)
    Xva = np.asarray(ctx["X_va"][:, cols], np.float64)
    rng = np.random.RandomState(0)
    sample = rng.choice(len(Xs), min(1500, len(Xs)), replace=False)
    ind = NearestNeighbors(n_neighbors=min(hp["KM"], len(Xs))).fit(Xs).kneighbors(Xs[sample], return_distance=False)
    p = Xs.shape[1]
    Bbar = np.zeros((p, p))
    for i in range(len(sample)):
        Xn, yn = Xs[ind[i]], ys[ind[i]]
        if min((yn == 0).sum(), (yn == 1).sum()) == 0:
            continue
        m = Xn.mean(0)
        for c in (0, 1):
            mc = Xn[yn == c].mean(0)
            Bbar += ((yn == c).mean()) * np.outer(mc - m, mc - m)
    Bbar /= len(sample)                                                              # (13.10)
    w, V = np.linalg.eigh(Bbar)
    V = V[:, ::-1]
    from sklearn.neighbors import KNeighborsClassifier
    rows = []
    for s in sorted({d for d in (1, 2, 3, 5, 8, p) if d <= p}):
        knn = KNeighborsClassifier(n_neighbors=min(hp["nn"], len(Xs) - 1), weights="distance").fit(Xs @ V[:, :s], ys)
        rows.append({"subspace_dim": s, "validation_auc": common.auc_score(ctx["y_va"], knn.predict_proba(Xva @ V[:, :s])[:, 1])})
    df = pd.DataFrame(rows)
    df.to_csv(ctx["out"] / "global_dimension_reduction.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(df["subspace_dim"], df["validation_auc"], "o-"); ax[0].set_xlabel("subspace dimension (leading eigenvectors of B-bar)"); ax[0].set_ylabel("validation AUC"); ax[0].set_title("DANN global dimension reduction (13.4.2)")
    ax[1].plot(w[::-1], "o-"); ax[1].set_xlabel("eigenvalue rank"); ax[1].set_ylabel("eigenvalue of B-bar"); ax[1].set_title("Averaged between matrix")
    fig.tight_layout(); fig.savefig(ctx["out"] / "dann_global_dimension_reduction.png", dpi=120); plt.close(fig)
    ctx["summary"]["dann"] = {"hp": hp, "best_subspace_dim": int(df.loc[df["validation_auc"].idxmax(), "subspace_dim"]), "best_subspace_validation_auc": float(df["validation_auc"].max()), "candidate_pool": CAND}


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    if task != "classification":
        raise ValueError("DANN is a classification method - run it with --task classification")
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=predict_path, complexity=complexity, extra=extra, select_by="auc",
                             notes="Local metric (13.8) from K_M neighbours; final neighbours among 400 Euclidean candidates.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
