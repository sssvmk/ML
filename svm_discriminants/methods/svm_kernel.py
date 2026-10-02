"""
Support vector machine with kernels (ESLII 12.3) - CLASSIFICATION, scikit-learn SVC.
Loss: the same hinge loss in the kernel feature space, f(x) = sum_i alpha_i y_i K(x, x_i) + beta_0 (12.20), equivalently a penalised RKHS problem (12.29). Kernels: radial basis exp(-gamma ||x - x'||^2) and degree-d polynomial (1 + gamma <x, x'>)^d.
Raw margins are the scores (AUC); probabilities / log-loss only after Platt scaling fitted on the VALIDATION set.
Hyper-parameters (CV, random search): kernel, C, gamma (as gamma_c / k), polynomial degree, class weights, k = number of top-ranked variables. CV selects on AUC.
Compute: tuned by CV on the tuning subsample; the FINAL fit uses <= svm_final_rows training rows (kernel SVMs cost ~ N^2 - N^3).
Metrics: test error +- SE, AUC +- SE, CV error and AUC over (C, gamma / degree), support-point fraction, calibration and log-loss after Platt scaling, confusion matrix.
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import matplotlib.pyplot as plt

import common
import svmlib as sl

NAME = "svm"
SPACE = {"frac": sl.FRACS, "kernel": ["rbf", "poly"], "C": [0.1, 1.0, 10.0], "gc": [0.1, 0.3, 1.0], "degree": [2, 3], "cw": [None, "balanced"]}


def build(hp, task, cfg, y):
    from sklearn.svm import SVC
    return SVC(kernel=hp["kernel"], C=hp["C"], gamma=hp["gc"] / hp["k"], degree=hp["degree"], coef0=1.0, class_weight=hp["cw"], cache_size=1000, max_iter=300000, probability=False, random_state=cfg.seed)


def make_grid(cfg, prep, Xs, ys):
    cfgs = sl.sample_configs(SPACE, cfg.n_configs_svm, cfg.seed)
    for c in cfgs:
        if c["kernel"] == "rbf":
            c["degree"] = 0                                                      # unused for rbf
    return sl.with_k(cfgs, prep.p)


def complexity(hp):
    return sl.drop_penalty(hp) + 10 * np.log10(hp["C"]) + 10 * np.log10(hp["gc"]) + 2 * hp["degree"]


def diag(model, hp, Xr, y, Xq, yq, task):
    return {"support_fraction": float(model.n_support_.sum() / len(y))}


def extra(ctx):
    S = ctx["store"]
    model, hp = S["model"], S["hp"]
    res = {"support_fraction": float(model.n_support_.sum() / len(S["yfit"])), "n_support": int(model.n_support_.sum()), "fit_rows": int(len(S["yfit"])), "kernel": hp["kernel"], "C": hp["C"],
           "gamma": hp["gc"] / hp["k"], "degree": hp["degree"], "n_features_used": int(len(S["cols"]))}
    ctx["summary"]["svm"] = res
    (ctx["out"] / "svm_extra_metrics.json").write_text(json.dumps(common._jsonable(res), indent=2))
    cu = ctx["curve"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for kern, d in cu.groupby("kernel"):
        ax[0].scatter(d["C"], d["cv_auc"], s=30 + 60 * d["gc"] / d["gc"].max(), label=f"{kern} (marker size ~ gamma_c)", alpha=0.7)
    ax[0].set_xscale("log"); ax[0].set_xlabel("C"); ax[0].set_ylabel("CV AUC"); ax[0].legend(fontsize=8); ax[0].set_title("Kernel SVM: CV AUC over (C, gamma, kernel)")
    if "cv_support_fraction" in cu:
        ax[1].scatter(cu["C"], cu["cv_support_fraction"], s=20); ax[1].set_xscale("log"); ax[1].set_xlabel("C"); ax[1].set_ylabel("fraction of support points (CV folds)"); ax[1].set_title("Support-point fraction")
    fig.tight_layout(); fig.savefig(ctx["out"] / "svm_cv_overview.png", dpi=120); plt.close(fig)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    if task != "classification":
        raise ValueError("The kernel SVM classifier is a classification method - run it with --task classification")
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid,
                             predict_path=sl.make_predict_path(build, sl.margin_score, capped=lambda hp: True, diag=diag), complexity=complexity, extra=extra,
                             select_by="auc", platt=True, scores_are_probs=False, notes="scikit-learn SVC; final fit capped at svm_final_rows; Platt scaling on validation for log-loss.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
