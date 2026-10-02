"""
K-means prototype classifier (ESLII 13.2.1) - CLASSIFICATION. Standard package: scikit-learn KMeans.
Procedure: run K-means separately inside each class (R prototypes per class), label the prototypes, classify a point to the class of the nearest prototype.
Loss: within-cluster sum of squares  sum_r sum_{x in cluster r} ||x - m_r||^2, minimised separately per class (classification error is NOT optimised directly; the CV curve logs the training value per row).
Score: distance to the nearest class-0 prototype minus distance to the nearest class-1 prototype (>0 favours class 1); AUC uses this score, probabilities / log-loss exist only after Platt scaling on VALIDATION.
Hyper-parameters (CV): prototypes per class R in {1 ... 20}, number of top-ranked variables k (ranking recomputed inside each CV fold). CV selects on AUC.
Metrics: test error +- SE (validation-tuned threshold), AUC +- DeLong SE, CV error / AUC vs R, log-loss after Platt, prototype map on the two top variables.
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import matplotlib.pyplot as plt

import common
import fitlib as fl
import nnlib

NAME = "kmeans"
SPACE = {"k": [3, 5, 8, 12, 20, 40], "R": [1, 2, 3, 5, 8, 12, 20]}


class KMeansProto:
    def __init__(self, R, seed):
        self.R, self.seed = R, seed

    def fit(self, X, y):
        from sklearn.cluster import KMeans
        self.centers, self.wcss = {}, 0.0
        for c in (0, 1):
            Xc = X[y == c]
            km = KMeans(n_clusters=min(self.R, len(Xc)), n_init=3, random_state=self.seed).fit(Xc)
            self.centers[c] = km.cluster_centers_
            self.wcss += float(km.inertia_)
        self.wcss_per_row = self.wcss / len(X)
        return self

    def class_distances(self, X):
        from sklearn.metrics import pairwise_distances
        return [pairwise_distances(X, self.centers[c]).min(1) for c in (0, 1)]

    def decision_function(self, X):
        d0, d1 = self.class_distances(X)
        return d0 - d1


def build(hp, task, cfg, y):
    return KMeansProto(hp["R"], cfg.seed)


def make_grid(cfg, prep, Xs, ys):
    return [{"k": k, "R": R} for k in SPACE["k"] if k <= prep.p for R in SPACE["R"]]


def complexity(hp):
    return hp["R"] * hp["k"]


def diag(model, hp, Xr, y, Xq, yq, task):
    return {"wcss_per_row": model.wcss_per_row}


def extra(ctx):
    S = ctx["store"]
    model, cols = S["model"], S["cols"]
    names = np.array(ctx["prep"].feature_names)[cols]
    X = np.asarray(ctx["X_te"][:3000][:, cols], np.float64)
    y = ctx["y_te"][:3000]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].scatter(X[:, 0], X[:, 1], c=y, s=4, alpha=0.3, cmap="coolwarm")
    for c, mk in ((0, "o"), (1, "*")):
        ax[0].scatter(model.centers[c][:, 0], model.centers[c][:, 1], s=140, marker=mk, edgecolors="k", c=[c] * len(model.centers[c]), cmap="coolwarm", vmin=0, vmax=1)
    ax[0].set_xlabel(names[0]); ax[0].set_ylabel(names[1] if len(names) > 1 else ""); ax[0].set_title("K-means prototypes (top-2 variables) and test points")
    cu = ctx["curve"]
    for kv, d in cu.groupby("k"):
        ax[1].plot(d["R"], d["cv_auc"], "o-", ms=3, label=f"{kv} variables")
    ax[1].set_xlabel("prototypes per class R"); ax[1].set_ylabel("CV AUC"); ax[1].legend(fontsize=7); ax[1].set_title("K-means classifier: CV AUC vs R")
    fig.tight_layout(); fig.savefig(ctx["out"] / "kmeans_prototypes_and_cv.png", dpi=120); plt.close(fig)
    ctx["summary"]["kmeans"] = {"R": S["hp"]["R"], "k_variables": S["hp"]["k"], "within_cluster_ss_per_row": model.wcss_per_row, "fit_rows": int(len(S["yfit"]))}


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    if task != "classification":
        raise ValueError("The K-means prototype classifier is a classification method - run it with --task classification")
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=fl.make_predict_path(build, fl.margin_score, diag=diag), complexity=complexity, extra=extra,
                             select_by="auc", platt=True, scores_are_probs=False, notes="scikit-learn KMeans per class; score = nearest-class-0 distance - nearest-class-1 distance; Platt on validation.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
