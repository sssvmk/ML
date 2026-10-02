"""
k-NN regression (not in ESLII Ch. 13; the uniform-kernel Nadaraya-Watson smoother of the earlier chapters) - REGRESSION on Zillow, four fixed entries {plain, distance-weighted} x {Euclidean, Manhattan}.
Loss: squared error with a locally constant fit, yhat(x0) = (weighted) mean of y over the k nearest points; there is no training loss, so k is chosen by CV.
scikit-learn NearestNeighbors does the search (Minkowski p = 2 Euclidean, p = 1 Manhattan); one search per number of variables serves every k.
Hyper-parameters (CV, time-blocked folds, one-SE rule: the LARGEST k, fewest variables within one paired SE): neighbours k in {5 ... 400}, number of top-ranked variables in {3 ... 40} (ranking recomputed inside every fold).
Metrics: test MSE +- SE, CV MSE vs k (and vs the number of variables), leave-one-out CV MSE vs k on the tuning subsample, plain vs distance-weighted and Euclidean vs Manhattan across the four entries.
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common
import nnlib

NN = [5, 10, 20, 50, 100, 200, 400]
KV = [3, 5, 8, 12, 20, 40]


def make(weights, p, name):
    def make_grid(cfg, prep, Xs, ys):
        return [{"k": k, "nn": nn} for k in KV if k <= prep.p for nn in NN if nn < len(ys)]

    def complexity(hp):
        return 1000.0 / hp["nn"] + hp["k"]

    def extra(ctx):
        S = ctx["store"]
        cols, hp = S["cols"], S["hp"]
        Xs = np.asarray(ctx["Xs"][:, cols], np.float64)
        nn_list = [n for n in NN if n < len(Xs)]
        loo = nnlib.loo_curve(Xs, ctx["ys_fit"], nn_list, weights, p, "regression", n_jobs=ctx["cfg"].n_jobs)
        df = pd.DataFrame({"nn": nn_list, "loo_mse": [float(np.mean((ctx["ys_ev"] - loo[n]) ** 2)) for n in nn_list]})
        df.to_csv(ctx["out"] / "loo_cv_vs_k.csv", index=False)
        cu = ctx["curve"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].plot(df["nn"], df["loo_mse"], "o-", label="leave-one-out (tuning subsample)")
        b = cu[cu["k"] == hp["k"]].set_index("nn")["cv_mse"]
        ax[0].plot(b.index, b.values, "s--", label="time-blocked CV (tuning subsample)"); ax[0].set_xscale("log"); ax[0].set_xlabel("neighbours k"); ax[0].set_ylabel("MSE"); ax[0].legend(); ax[0].set_title(f"{name}: CV and LOO vs k")
        for kv, d in cu.groupby("k"):
            ax[1].plot(d["nn"], d["cv_mse"], label=f"{kv} variables")
        ax[1].set_xscale("log"); ax[1].set_xlabel("neighbours k"); ax[1].set_ylabel("CV MSE"); ax[1].legend(fontsize=7); ax[1].set_title("CV MSE by number of variables")
        fig.tight_layout(); fig.savefig(ctx["out"] / "knn_cv_curves.png", dpi=120); plt.close(fig)
        ctx["summary"]["knn_regression"] = {"weights": weights, "metric": "euclidean" if p == 2 else "manhattan", "k_neighbours": hp["nn"], "n_variables": hp["k"], "best_loo_k": int(df.loc[df["loo_mse"].idxmin(), "nn"])}

    def run(prep_dir, out_dir, cfg, progress=None, console=False):
        task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
        if task != "regression":
            raise ValueError("k-NN regression is a regression method - run it with --task regression")
        return common.run_method(name, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=nnlib.make_predict_path(weights, p), complexity=complexity, extra=extra, select_by="mse",
                                 notes=f"scikit-learn NearestNeighbors; {weights} weights; Minkowski p = {p}.", console=console)
    return make_grid, run
