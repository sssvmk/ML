"""
Support vector classifier (ESLII 12.2) - CLASSIFICATION, scikit-learn LinearSVC (all training rows).
Loss: hinge + penalty  sum_i [1 - y_i f(x_i)]_+ + (lambda/2) ||beta||^2  (12.25), lambda = 1/C, i.e. min 1/2 ||beta||^2 + C sum xi_i.  The hinge loss estimates sign[P(Y=1|x) - 1/2], not a probability (Table 12.1):
raw margins f(x) are the scores; probabilities (log-loss) exist only after Platt scaling fitted on the VALIDATION set.
Hyper-parameters (CV, random search): C, class weights (imbalance), k = number of top-ranked variables (ranking recomputed inside every CV fold). CV selects on AUC.
Metrics: test error +- SE (validation-tuned threshold), AUC +- DeLong SE from f(x), CV error and AUC vs C (one-SE rule), fraction of support points (y f <= 1), margin 1/||beta||, log-loss after Platt.
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import matplotlib.pyplot as plt

import common
import svmlib as sl

NAME = "svc"
SPACE = {"frac": sl.FRACS, "C": [1e-4, 1e-3, 1e-2, 1e-1, 1.0], "cw": [None, "balanced"]}


def build(hp, task, cfg, y):
    from sklearn.svm import LinearSVC
    return LinearSVC(loss="hinge", C=hp["C"], class_weight=hp["cw"], dual=True, max_iter=5000, tol=1e-3, random_state=cfg.seed)


def make_grid(cfg, prep, Xs, ys):
    return sl.with_k(sl.sample_configs(SPACE, cfg.n_configs_svm, cfg.seed), prep.p)


def complexity(hp):
    return sl.drop_penalty(hp) + 10 * np.log10(hp["C"])


def diag(model, hp, Xr, y, Xq, yq, task):
    f = model.decision_function(Xr)
    ypm = 2 * y - 1
    return {"support_fraction": float(np.mean(ypm * f <= 1 + 1e-6)), "margin": float(1.0 / max(np.linalg.norm(model.coef_), 1e-12))}


def extra(ctx):
    S = ctx["store"]
    model = S["model"]
    f = model.decision_function(S["Xfit"])
    ypm = 2 * S["yfit"] - 1
    res = {"support_fraction_training": float(np.mean(ypm * f <= 1 + 1e-6)), "margin_1_over_norm_beta": float(1.0 / np.linalg.norm(model.coef_)),
           "hinge_loss_training": float(np.mean(np.maximum(0, 1 - ypm * f))), "n_features_used": int(len(S["cols"])), "C": S["hp"]["C"]}
    ctx["summary"]["svc"] = res
    (ctx["out"] / "svc_extra_metrics.json").write_text(json.dumps(common._jsonable(res), indent=2))
    cu = ctx["curve"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for cw, d in cu.groupby(cu["cw"].astype(str)):
        b = d.groupby("C")["cv_auc"].max(); ax[0].plot(b.index, b.values, "o-", label=f"class_weight={cw}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("C"); ax[0].set_ylabel("best CV AUC"); ax[0].legend(); ax[0].set_title("Support vector classifier: CV AUC vs C")
    if "cv_error_pooled_youden" in cu:
        for cw, d in cu.groupby(cu["cw"].astype(str)):
            b = d.groupby("C")["cv_error_pooled_youden"].min(); ax[1].plot(b.index, b.values, "o-", label=f"class_weight={cw}")
        ax[1].set_xscale("log"); ax[1].set_xlabel("C"); ax[1].set_ylabel("CV error (pooled Youden threshold)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(ctx["out"] / "svc_cv_vs_C.png", dpi=120); plt.close(fig)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    if task != "classification":
        raise ValueError("The support vector classifier is a classification method - run it with --task classification")
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=sl.make_predict_path(build, sl.margin_score, diag=diag), complexity=complexity, extra=extra,
                             select_by="auc", platt=True, scores_are_probs=False, notes="scikit-learn LinearSVC (hinge loss), all training rows; Platt scaling on validation for log-loss.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
