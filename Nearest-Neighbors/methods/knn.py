"""
k-NN classifier (ESLII 13.3) - CLASSIFICATION. Standard package: scikit-learn NearestNeighbors (Euclidean distance on the standardised variables).
Rule: majority vote among the k nearest training points (implicit 0-1 loss; no training loss, k chosen by CV). Scores are DISTANCE-WEIGHTED vote fractions (so AUC is not crippled by ties), smoothed
(p k + base rate)/(k + 1) for log-loss and Brier (monotone, AUC unchanged).
Hyper-parameters (CV, one-SE rule: the LARGEST k, fewest variables within one paired SE): k in {1 ... 200}, number of top-ranked variables in {3 ... 40} (ranking recomputed inside every fold). CV selects on AUC.
Metrics: test error +- SE, CV error / AUC vs k, AUC +- DeLong SE, log-loss and Brier (smoothed proportions, clipped), 1-NN error and the Cover-Hart lower bound on the Bayes error.
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

import common
import nnlib

NAME = "knn"
NN = [1, 3, 5, 10, 15, 25, 50, 100, 200]
KV = [3, 5, 8, 12, 20, 40]


def make_grid(cfg, prep, Xs, ys):
    return [{"k": k, "nn": nn} for k in KV if k <= prep.p for nn in NN if nn < len(ys)]


def complexity(hp):
    return 1000.0 / hp["nn"] + hp["k"]


def extra(ctx):
    S = ctx["store"]
    cols, Xr, yr = S["cols"], S["Xr"], S["yr"]
    Xte = np.asarray(ctx["X_te"][:, cols], np.float64)
    if S.get("project") is not None:
        Xte = S["project"](Xte)
    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=ctx["cfg"].n_jobs).fit(Xr)
    pred1 = yr[nbrs.kneighbors(Xte, return_distance=False)[:, 0]].astype(int)
    err1 = float(np.mean(pred1 != ctx["y_te"]))
    bound = 0.5 * (1 - np.sqrt(max(0.0, 1 - 2 * err1))) if err1 <= 0.5 else float("nan")
    res = {"one_nn_test_error": err1, "cover_hart_lower_bound_on_bayes_error": bound, "chosen_k": S["hp"]["nn"], "chosen_test_error": ctx["summary"]["test"]["error"], "majority_class_error": ctx["summary"]["test"]["majority_class_error"],
           "note": "asymptotically err(1-NN) <= 2 E*(1 - E*) (ESLII 13.2-13.5); the bound inverts this for a binary problem"}
    ctx["summary"]["knn"] = res
    (ctx["out"] / "knn_extra_metrics.json").write_text(json.dumps(common._jsonable(res), indent=2))
    cu = ctx["curve"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for kv, d in cu.groupby("k"):
        ax[0].plot(d["nn"], d["cv_auc"], "o-", ms=3, label=f"{kv} variables")
        if "cv_error_pooled_youden" in d:
            ax[1].plot(d["nn"], d["cv_error_pooled_youden"], "o-", ms=3)
    ax[0].set_xscale("log"); ax[0].set_xlabel("neighbours k"); ax[0].set_ylabel("CV AUC"); ax[0].legend(fontsize=7); ax[0].set_title("k-NN: CV AUC vs k")
    ax[1].set_xscale("log"); ax[1].set_xlabel("neighbours k"); ax[1].set_ylabel("CV error (pooled Youden threshold)"); ax[1].set_title("k-NN: CV error vs k")
    fig.tight_layout(); fig.savefig(ctx["out"] / "knn_cv_curves.png", dpi=120); plt.close(fig)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    if task != "classification":
        raise ValueError("The k-NN classifier is a classification method - run it with --task classification")
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=nnlib.make_predict_path("distance", 2), complexity=complexity, extra=extra,
                             select_by="auc", notes="scikit-learn NearestNeighbors, distance-weighted vote fractions, Euclidean.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
