"""
AdaBoost.M1 (ESLII 10.1, 10.4-10.5) - CLASSIFICATION ONLY (scikit-learn AdaBoostClassifier, discrete SAMME = AdaBoost.M1 for two classes), trees of depth d as the weak learner.
Loss: exponential  exp(-y f(x)); weights w_i = exp(-y_i f_{m-1}(x_i)), beta_m = 1/2 log((1 - err_m)/err_m) (10.12); f estimates one-half the log-odds (10.16).
Scores: f(x) = 1/2 sum_m alpha_m G_m(x) (from sklearn's staged_decision_function); probabilities p = 1 / (1 + e^(-2f)) (10.17) - used for log-loss and a calibration check (AdaBoost's p is known to be poorly calibrated).
Hyper-parameters (CV, random search): M, depth d of the weak tree, learning rate (shrinkage), fraction of the ranked variables kept (all unless dropping helps).
Metrics: test error +- SE, exponential loss, AUC, log-loss (after the p mapping), CV AUC vs M, training error and exponential loss vs M (error can hit zero while the exponential loss keeps falling).
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common
import boostlib as bl

NAME = "adaboost"
M_GRID = [10, 25, 50, 100, 200]


def build(conf, M, task, cfg, y):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=conf["depth"], random_state=cfg.seed), n_estimators=M, learning_rate=conf["lr"], random_state=cfg.seed)


def make_grid(cfg, prep, Xs, ys):
    space = {"frac": bl.FRACS, "depth": [1, 2, 3], "lr": [0.5, 1.0]}
    return bl.expand(bl.sample_configs(space, cfg.n_configs_sk, cfg.seed), M_GRID, prep.p, cfg.max_iter)


def complexity(hp):
    return bl.drop_penalty(hp) + hp["M"] * hp["depth"]


def extra(ctx):
    bl.standard_extra(ctx, group_col="depth", pdp=False)
    S = ctx["store"]
    cols, conf, cfg = S["cols"], S["conf"], ctx["cfg"]
    r0 = bl.cap_rows(len(ctx["ys_ev"]), cfg.diag_rows)
    Xs, ys = np.asarray(ctx["Xs"][r0][:, cols], np.float32), ctx["ys_ev"][r0].astype(int)
    rows = np.random.RandomState(0).choice(len(ctx["y_te"]), min(10000, len(ctx["y_te"])), replace=False)
    Xte, yte = np.asarray(ctx["X_te"][rows][:, cols], np.float32), ctx["y_te"][rows]
    m = build(conf, max(bl.grid_m(M_GRID, cfg.max_iter)), "classification", cfg, ys).fit(Xs, ys)
    cum = np.cumsum(m.estimator_weights_)
    out = []
    ftr, fte = list(m.staged_decision_function(Xs)), list(m.staged_decision_function(Xte))
    for i in range(len(ftr)):
        a_tr, a_te = cum[i] / 4.0 * ftr[i], cum[i] / 4.0 * fte[i]
        r = {"M": i + 1}
        for nm, f, y in (("train", a_tr, ys), ("test", a_te, yte)):
            ypm = 2 * y - 1
            r[f"{nm}_exp_loss"] = float(np.mean(np.exp(-ypm * f)))
            r[f"{nm}_error_sign_f"] = float(np.mean(np.sign(f) != ypm))
            r[f"{nm}_auc"] = common.auc_score(y, f)
        out.append(r)
    df = pd.DataFrame(out)
    df.to_csv(ctx["out"] / "train_vs_test_curves.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(df["M"], df["train_error_sign_f"], label="training error"); ax[0].plot(df["M"], df["test_error_sign_f"], label="test error"); ax[0].set_xlabel("M"); ax[0].set_ylabel("misclassification (sign f)"); ax[0].legend()
    ax[1].plot(df["M"], df["train_exp_loss"], label="training"); ax[1].plot(df["M"], df["test_exp_loss"], label="test"); ax[1].set_xlabel("M"); ax[1].set_ylabel("mean exp(-y f)"); ax[1].legend()
    fig.suptitle("AdaBoost: training error vs exponential loss"); fig.tight_layout(); fig.savefig(ctx["out"] / "adaboost_error_vs_exp_loss.png", dpi=120); plt.close(fig)
    ctx["summary"]["adaboost_curves_final_row"] = df.iloc[-1].to_dict()


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    if task != "classification":
        raise ValueError("AdaBoost is a classification-only method in ESLII - run it with --task classification")
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=bl.make_predict_path(build, "ada"), complexity=complexity, extra=extra,
                             select_by="auc", notes="scikit-learn AdaBoostClassifier (SAMME); p = 1/(1+exp(-2f)).", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
