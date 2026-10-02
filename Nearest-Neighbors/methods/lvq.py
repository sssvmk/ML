"""
Learning vector quantization, LVQ1 (ESLII 13.2.2) - CLASSIFICATION. No standard Python package implements Kohonen's LVQ1, so this is the minimal faithful loop (as agreed), started from scikit-learn KMeans prototypes.
No fixed optimisation criterion - an online algorithm: for each training point x_i, find the nearest prototype m_v;  m_v <- m_v + eps (x_i - m_v) if the classes agree, m_v <- m_v - eps (x_i - m_v) otherwise,
with eps decreasing linearly to 0 over the epochs. Score: distance to the nearest class-0 prototype minus distance to the nearest class-1 prototype; probabilities only after Platt scaling on VALIDATION.
Hyper-parameters (CV): prototypes per class R, initial learning rate eps, number of top-ranked variables k; 5 epochs. CV selects on AUC.
Metrics: test error +- SE, AUC +- DeLong SE, CV error / AUC vs R, improvement over the K-means start (AUC gain logged per configuration), log-loss after Platt.
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import matplotlib.pyplot as plt

import common
import fitlib as fl
import nnlib

NAME = "lvq"
SPACE = {"k": [3, 5, 8, 12, 20, 40], "R": [2, 3, 5, 8, 12, 20], "eps": [0.01, 0.03, 0.1]}
EPOCHS = 5


def lvq_step(protos, labels, x, y, eps):
    """One LVQ1 update (in place): nearest prototype attracted by a same-class point, repelled by a different-class point."""
    j = int(np.argmin(((protos - x) ** 2).sum(1)))
    protos[j] += (eps if labels[j] == y else -eps) * (x - protos[j])
    return j


class LVQ1:
    def __init__(self, R, eps, seed, epochs=EPOCHS):
        self.R, self.eps, self.seed, self.epochs = R, eps, seed, epochs

    def fit(self, X, y):
        from sklearn.cluster import KMeans
        cents, labs = [], []
        for c in (0, 1):
            Xc = X[y == c]
            km = KMeans(n_clusters=min(self.R, len(Xc)), n_init=2, random_state=self.seed).fit(Xc)
            cents.append(km.cluster_centers_); labs += [c] * len(km.cluster_centers_)
        self.protos, self.labels = np.vstack(cents), np.array(labs)
        self.start = self.protos.copy()                                            # the K-means solution
        rng = np.random.RandomState(self.seed)
        T, t = self.epochs * len(X), 0
        for _ in range(self.epochs):
            for i in rng.permutation(len(X)):
                lvq_step(self.protos, self.labels, X[i], y[i], self.eps * (1.0 - t / T))
                t += 1
        return self

    @staticmethod
    def _score(protos, labels, X):
        from sklearn.metrics import pairwise_distances
        D = pairwise_distances(X, protos)
        return D[:, labels == 0].min(1) - D[:, labels == 1].min(1)

    def decision_function(self, X):
        return self._score(self.protos, self.labels, X)

    def start_scores(self, X):
        return self._score(self.start, self.labels, X)


def build(hp, task, cfg, y):
    return LVQ1(hp["R"], hp["eps"], cfg.seed)


def make_grid(cfg, prep, Xs, ys):
    base = [{"k": k, "R": R, "eps": e} for k in SPACE["k"] if k <= prep.p for R in SPACE["R"] for e in SPACE["eps"]]
    return [base[c["i"]] for c in nnlib.sample_configs({"i": list(range(len(base)))}, cfg.n_configs_nn, cfg.seed)]


def complexity(hp):
    return hp["R"] * hp["k"]


def diag(model, hp, Xr, y, Xq, yq, task):
    d = {}
    if yq is not None and len(np.unique(yq)) == 2:
        a_l, a_k = common.auc_score(yq.astype(int), model.decision_function(Xq)), common.auc_score(yq.astype(int), model.start_scores(Xq))
        d.update({"auc_lvq": a_l, "auc_kmeans_start": a_k, "auc_gain_over_kmeans": a_l - a_k})
    return d


def extra(ctx):
    S = ctx["store"]
    model, cols = S["model"], S["cols"]
    y = ctx["y_te"]
    X = np.asarray(ctx["X_te"][:, cols], np.float64)
    a_l, a_k = common.auc_score(y, model.decision_function(X)), common.auc_score(y, model.start_scores(X))
    names = np.array(ctx["prep"].feature_names)[cols]
    ctx["summary"]["lvq"] = {"R": S["hp"]["R"], "eps": S["hp"]["eps"], "epochs": EPOCHS, "test_auc_lvq": a_l, "test_auc_kmeans_start": a_k, "test_auc_gain": a_l - a_k,
                             "mean_prototype_displacement": float(np.linalg.norm(model.protos - model.start, axis=1).mean()), "fit_rows": int(len(S["yfit"]))}
    fig, ax = plt.subplots(figsize=(6, 5))
    Xp = X[:3000]
    ax.scatter(Xp[:, 0], Xp[:, 1] if X.shape[1] > 1 else 0 * Xp[:, 0], c=y[:3000], s=4, alpha=0.25, cmap="coolwarm")
    for P, mk, lab in ((model.start, "o", "K-means start"), (model.protos, "*", "after LVQ1")):
        ax.scatter(P[:, 0], P[:, 1] if P.shape[1] > 1 else 0 * P[:, 0], c=model.labels, marker=mk, s=110, edgecolors="k", cmap="coolwarm", label=lab)
    ax.set_xlabel(names[0]); ax.set_title("LVQ1: prototypes before / after"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ctx["out"] / "lvq_prototypes.png", dpi=120); plt.close(fig)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    if task != "classification":
        raise ValueError("LVQ is a classification method - run it with --task classification")
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=fl.make_predict_path(build, fl.margin_score, diag=diag), complexity=complexity, extra=extra,
                             select_by="auc", platt=True, scores_are_probs=False, notes="LVQ1 loop started from scikit-learn KMeans; Platt on validation.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
