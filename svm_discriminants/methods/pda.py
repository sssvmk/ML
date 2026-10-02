"""
Penalized discriminant analysis (ESLII 12.6) - CLASSIFICATION. Penalised optimal scoring from standard components:
  1. PENALISED regression of the class-indicator matrix on the features: min sum_i (theta(g_i) - x_i'beta)^2 + lambda beta' Omega beta  (scikit-learn Ridge, Omega = I)
  2. LDA on the fitted values (scikit-learn LinearDiscriminantAnalysis)
which is LDA with the within-class covariance Sigma_w + lambda * Omega. Omega = I because the Santander variables are not ordered signals (a roughness Omega only makes sense for ordered features such as signals / images);
the discriminant vector's roughness along the variable order is reported anyway.
Hyper-parameters (CV): lambda (one-SE rule: the LARGEST lambda within one SE), k = number of top-ranked variables; effective df(lambda) = sum d_j^2 / (d_j^2 + lambda) is logged. CV selects on AUC.
Metrics: test error +- SE, AUC, log-loss (LDA posteriors, clipped), CV error / AUC vs lambda or df, coefficient plot and roughness of the discriminant vector.
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import matplotlib.pyplot as plt

import common
import svmlib as sl

NAME = "pda"
SPACE = {"frac": sl.FRACS, "alpha": [1.0, 10.0, 100.0, 1e3, 1e4, 1e5]}


class PDAModel:
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        from sklearn.linear_model import Ridge
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        Y = np.eye(int(y.max()) + 1)[y]
        self.reg = Ridge(alpha=self.alpha).fit(X, Y)
        self.lda = LinearDiscriminantAnalysis(solver="svd").fit(self.reg.predict(X)[:, 1:], y)
        s = np.linalg.svd(X - X.mean(0), compute_uv=False)
        self.df = float(np.sum(s ** 2 / (s ** 2 + self.alpha)))
        return self

    def predict_proba(self, X):
        return self.lda.predict_proba(self.reg.predict(X)[:, 1:])


def roughness(beta):
    return float(np.sum(np.diff(beta) ** 2) / max(2 * np.sum(beta ** 2), 1e-300))


def build(hp, task, cfg, y):
    return PDAModel(hp["alpha"])


def make_grid(cfg, prep, Xs, ys):
    return sl.with_k(sl.sample_configs(SPACE, cfg.n_configs_svm, cfg.seed), prep.p)


def complexity(hp):
    return sl.drop_penalty(hp) - 10 * np.log10(hp["alpha"])


def diag(model, hp, Xr, y, Xq, yq, task):
    return {"df": model.df, "roughness": roughness(model.reg.coef_[1])}


def extra(ctx):
    S = ctx["store"]
    model, cols = S["model"], S["cols"]
    beta = model.reg.coef_[1]
    names = np.array(ctx["prep"].feature_names)[cols]
    res = {"lambda": S["hp"]["alpha"], "effective_df": model.df, "roughness_along_variable_order": roughness(beta), "n_features_used": int(len(cols)), "fit_rows": int(len(S["yfit"]))}
    ctx["summary"]["pda"] = res
    (ctx["out"] / "pda_extra_metrics.json").write_text(json.dumps(common._jsonable(res), indent=2))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(beta, lw=0.8); ax[0].set_xlabel("variable (ranking order)"); ax[0].set_ylabel("discriminant coefficient"); ax[0].set_title(f"PDA discriminant vector (lambda = {S['hp']['alpha']:g})")
    cu = ctx["curve"]
    d = cu.groupby("alpha").agg(cv_auc=("cv_auc", "max"), df=("cv_df", "mean") if "cv_df" in cu else ("cv_auc", "max"))
    ax[1].plot(d.index, d["cv_auc"], "o-"); ax[1].set_xscale("log"); ax[1].set_xlabel("lambda"); ax[1].set_ylabel("best CV AUC"); ax[1].set_title("PDA: CV AUC vs lambda")
    fig.tight_layout(); fig.savefig(ctx["out"] / "pda_discriminant_vector.png", dpi=120); plt.close(fig)
    np.save(ctx["out"] / "pda_names.npy", names)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    if task != "classification":
        raise ValueError("PDA is a classification method - run it with --task classification")
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=sl.make_predict_path(build, sl.margin_score, diag=diag), complexity=complexity, extra=extra,
                             select_by="auc", notes="Ridge (Omega = I) regression of the class indicators + LDA on the fitted values; all training rows.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
