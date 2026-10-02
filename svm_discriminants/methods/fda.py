"""
Flexible discriminant analysis (ESLII 12.5) - CLASSIFICATION. Built from standard components exactly as in Hastie, Tibshirani & Buja (1994):
  1. non-parametric regression of the class-indicator matrix Y on a basis h(x)   (scikit-learn SplineTransformer or Nystroem-RBF basis + Ridge)  -> fitted values Y_hat
  2. LDA on the fitted values (scikit-learn LinearDiscriminantAnalysis)          -> optimal scores / discriminant coordinates, Gaussian posteriors, nearest-centroid classification.
Loss (optimal scoring): min over scores theta and coefficients beta of the average squared residual  sum_i (theta(g_i) - h(x_i)'beta)^2  - FDA is LDA in the enlarged feature space.
For two classes there is a single discriminant coordinate. Hyper-parameters (CV, random search): basis (spline with 3 / 5 knots, Nystroem-RBF with 100 / 300 components), ridge penalty, kernel width, k = number of
top-ranked variables. CV selects on AUC. Final fit uses <= svm_final_rows rows.
Metrics: test error +- SE, AUC and log-loss from the LDA posteriors (clipped), CV error / AUC vs the regression method's complexity, canonical-coordinate (discriminant score) plot.
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import matplotlib.pyplot as plt

import common
import svmlib as sl

NAME = "fda"
SPACE = {"frac": sl.FRACS, "basis": ["spline3", "spline5", "nystroem100", "nystroem300"], "alpha": [1.0, 100.0, 10000.0], "gc": [0.1, 0.3, 1.0]}


class FDAModel:
    def __init__(self, basis, alpha, gc, seed):
        self.basis, self.alpha, self.gc, self.seed = basis, alpha, gc, seed

    def fit(self, X, y):
        from sklearn.linear_model import Ridge
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.preprocessing import SplineTransformer
        from sklearn.kernel_approximation import Nystroem
        n, k = X.shape
        if self.basis.startswith("spline"):
            self.T = SplineTransformer(n_knots=int(self.basis[6:]), degree=3, knots="quantile", extrapolation="constant")
        else:
            self.T = Nystroem(kernel="rbf", gamma=self.gc / k, n_components=min(int(self.basis[8:]), n), random_state=self.seed)
        H = self.T.fit_transform(X)
        Y = np.eye(int(y.max()) + 1)[y]                                           # class-indicator matrix
        self.reg = Ridge(alpha=self.alpha).fit(H, Y)                              # (penalised) regression of Y on the basis
        Yhat = self.reg.predict(H)
        self.lda = LinearDiscriminantAnalysis(solver="svd").fit(Yhat[:, 1:], y)   # LDA on the fitted values (K-1 coordinates)
        return self

    def scores(self, X):
        return self.reg.predict(self.T.transform(X))

    def predict_proba(self, X):
        return self.lda.predict_proba(self.scores(X)[:, 1:])


def build(hp, task, cfg, y):
    return FDAModel(hp["basis"], hp["alpha"], hp["gc"], cfg.seed)


def make_grid(cfg, prep, Xs, ys):
    cfgs = sl.with_k(sl.sample_configs(SPACE, cfg.n_configs_svm, cfg.seed), prep.p)
    ok = [c for c in cfgs if not (c["basis"].startswith("spline") and c["k"] * (int(c["basis"][6:]) + 2) * cfg.svm_final_rows > 3e8)]
    return ok or cfgs[:1]


def complexity(hp):
    n = int(hp["basis"].replace("spline", "").replace("nystroem", ""))
    return sl.drop_penalty(hp) + n - 2 * np.log10(hp["alpha"])


def extra(ctx):
    S = ctx["store"]
    model, cols = S["model"], S["cols"]
    sc = model.scores(np.asarray(ctx["X_te"][:, cols], np.float64))[:, 1]
    y = ctx["y_te"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist(sc[y == 0], bins=60, alpha=0.6, density=True, label="class 0"); ax[0].hist(sc[y == 1], bins=60, alpha=0.6, density=True, label="class 1")
    ax[0].set_xlabel("discriminant (canonical) coordinate = fitted indicator"); ax[0].legend(); ax[0].set_title("FDA: test-set discriminant coordinate")
    cu = ctx["curve"]
    for b, d in cu.groupby("basis"):
        ax[1].scatter(d["alpha"], d["cv_auc"], label=b, s=30)
    ax[1].set_xscale("log"); ax[1].set_xlabel("ridge penalty"); ax[1].set_ylabel("CV AUC"); ax[1].legend(fontsize=8); ax[1].set_title("FDA: CV AUC by basis / penalty")
    fig.tight_layout(); fig.savefig(ctx["out"] / "fda_discriminant_coordinate.png", dpi=120); plt.close(fig)
    ctx["summary"]["fda"] = {"basis": S["hp"]["basis"], "basis_columns": int(model.T.transform(np.asarray(ctx["X_te"][:200, cols], np.float64)).shape[1]), "fit_rows": int(len(S["yfit"]))}


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    if task != "classification":
        raise ValueError("FDA is a classification method - run it with --task classification")
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=sl.make_predict_path(build, sl.margin_score, capped=lambda hp: True), complexity=complexity,
                             extra=extra, select_by="auc", notes="Ridge regression of the class indicators on a spline / Nystroem basis + LDA on the fitted values.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
