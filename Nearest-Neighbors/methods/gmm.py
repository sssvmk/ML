"""
Gaussian mixture classifier (ESLII 13.2.3, 6.8, 12.7) - CLASSIFICATION. Standard package: scikit-learn GaussianMixture, one mixture per class, posteriors by Bayes' rule with the training class proportions.
Loss: negative mixture log-likelihood per class, -sum_i log sum_r pi_r phi(x_i; mu_r, Sigma_r), fitted by EM (soft clustering: responsibilities instead of K-means' hard assignments).
Hyper-parameters (CV): components R per class in {1 ... 8}, covariance type (diag / full), number of top-ranked variables k. CV selects on AUC.
Metrics: test error +- SE, AUC, log-loss (smooth posteriors, clipped), held-out log-likelihood, BIC / AIC and CV error vs R, calibration.
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import matplotlib.pyplot as plt

import common
import fitlib as fl
import nnlib

NAME = "gmm"
SPACE = {"k": [3, 5, 8, 12, 20, 40], "R": [1, 2, 3, 5, 8], "cov": ["diag", "full"]}


class GMMClassifier:
    def __init__(self, R, cov, seed):
        self.R, self.cov, self.seed = R, cov, seed

    def fit(self, X, y):
        from sklearn.mixture import GaussianMixture
        self.gm, self.n_c = {}, {}
        for c in (0, 1):
            Xc = X[y == c]
            self.gm[c] = GaussianMixture(n_components=min(self.R, max(1, len(Xc) // 20)), covariance_type=self.cov, reg_covar=1e-3, max_iter=150, n_init=1, random_state=self.seed).fit(Xc)
            self.n_c[c] = len(Xc)
        self.prior = np.array([self.n_c[0], self.n_c[1]], float) / len(X)
        self.bic = sum(self.gm[c].bic(X[y == c]) for c in (0, 1))
        self.aic = sum(self.gm[c].aic(X[y == c]) for c in (0, 1))
        return self

    def predict_proba(self, X):
        lp = np.stack([self.gm[c].score_samples(X) + np.log(self.prior[c]) for c in (0, 1)], 1)
        p1 = common.sigmoid(np.clip(lp[:, 1] - lp[:, 0], -30, 30))
        return np.c_[1 - p1, p1]

    def heldout_ll(self, X, y):
        return float(np.mean([self.gm[int(c)].score_samples(x[None])[0] + np.log(self.prior[int(c)]) for x, c in zip(X, y)])) if len(X) < 300 else float(
            sum((self.gm[c].score_samples(X[y == c]) + np.log(self.prior[c])).sum() for c in (0, 1)) / len(X))


def build(hp, task, cfg, y):
    return GMMClassifier(hp["R"], hp["cov"], cfg.seed)


def make_grid(cfg, prep, Xs, ys):
    base = [{"k": k, "R": R, "cov": c} for k in SPACE["k"] if k <= prep.p for R in SPACE["R"] for c in SPACE["cov"]]
    return [base[c["i"]] for c in nnlib.sample_configs({"i": list(range(len(base)))}, cfg.n_configs_nn, cfg.seed)]


def complexity(hp):
    return hp["R"] * hp["k"] * (hp["k"] if hp["cov"] == "full" else 1)


def diag(model, hp, Xr, y, Xq, yq, task):
    d = {"bic_train": model.bic, "aic_train": model.aic}
    if yq is not None:
        d["heldout_loglik"] = model.heldout_ll(Xq, yq.astype(int))
    return d


def extra(ctx):
    cu = ctx["curve"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for (cv_, kv), d in cu.groupby(["cov", "k"]):
        d = d.sort_values("R")
        ax[0].plot(d["R"], d["cv_auc"], "o-", ms=3, label=f"{cv_}, {kv} vars")
        if "cv_heldout_loglik" in d:
            ax[1].plot(d["R"], d["cv_heldout_loglik"], "o-", ms=3)
        if "cv_bic_train" in d:
            ax[2].plot(d["R"], d["cv_bic_train"] - d["cv_bic_train"].min(), "o-", ms=3)
    ax[0].set_xlabel("components per class R"); ax[0].set_ylabel("CV AUC"); ax[0].legend(fontsize=6); ax[1].set_xlabel("R"); ax[1].set_ylabel("held-out log-likelihood p(x, class)"); ax[2].set_xlabel("R"); ax[2].set_ylabel("BIC - min (training fits)")
    fig.suptitle("Gaussian mixture classifier: CV AUC, held-out likelihood and BIC vs R"); fig.tight_layout(); fig.savefig(ctx["out"] / "gmm_cv_overview.png", dpi=120); plt.close(fig)
    S = ctx["store"]
    ctx["summary"]["gmm"] = {"R": S["hp"]["R"], "cov": S["hp"]["cov"], "k_variables": S["hp"]["k"], "bic_train": float(S["model"].bic), "aic_train": float(S["model"].aic), "fit_rows": int(len(S["yfit"]))}


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    if task != "classification":
        raise ValueError("The Gaussian mixture classifier is a classification method - run it with --task classification")
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=fl.make_predict_path(build, fl.margin_score, diag=diag), complexity=complexity, extra=extra,
                             select_by="auc", notes="scikit-learn GaussianMixture per class + Bayes' rule.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
