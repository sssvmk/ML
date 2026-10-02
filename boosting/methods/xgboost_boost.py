"""
XGBoost (not in ESLII) - prebuilt `xgboost` package: regularised gradient boosting, objective  sum_i l(y_i, yhat_i) + sum_k [gamma T_k + 1/2 lambda ||w_k||^2 + alpha ||w_k||_1],
optimised with a second-order Taylor expansion of the loss (histogram tree method).
  regression     : squared error 1/2 (y - yhat)^2 (reg:squarederror)
  classification : logistic loss (binary:logistic); scale_pos_weight in {1, sqrt(neg/pos), neg/pos} for the class imbalance (judged by AUC)
Hyper-parameters (CV, random search): max_depth, eta, gamma, lambda, alpha, subsample, colsample_bytree, min_child_weight, M (iterations), fraction of the ranked variables kept (all unless dropping helps).
Early stopping on the VALIDATION set (never the test set) is reported separately next to the CV-chosen M.
Metrics: regression: test MSE +- SE, CV MSE over the grid, early stopping, gain importance.  classification: test log-loss +- SE, error +- SE, AUC +- SE, calibration, CV AUC vs M, gain importance.
"""
import _bootstrap  # noqa: F401
import json

import numpy as np

import common
import boostlib as bl

NAME = "xgboost"
M_GRID = [20, 50, 100, 200, 300, 500, 800, 1200]


def build(conf, M, task, cfg, y, early=None):
    import xgboost as xgb
    kw = dict(n_estimators=M, max_depth=conf["max_depth"], learning_rate=conf["lr"], gamma=conf["gamma"], reg_lambda=conf["lam"], reg_alpha=conf["alpha"], subsample=conf["subsample"],
              colsample_bytree=conf["colsample"], min_child_weight=conf["min_child_weight"], tree_method="hist", n_jobs=cfg.n_jobs, random_state=cfg.seed, verbosity=0)
    if early:
        kw["early_stopping_rounds"] = early
    if task == "regression":
        return xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **kw)
    return xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", scale_pos_weight=conf["spw"], **kw)


def make_grid(cfg, prep, Xs, ys):
    space = {"frac": bl.FRACS, "max_depth": [2, 3, 4, 6], "lr": [0.02, 0.05, 0.1], "gamma": [0, 0.1, 1], "lam": [1, 10, 50], "alpha": [0, 1], "subsample": [0.5, 0.8, 1.0],
             "colsample": [0.3, 0.6, 1.0], "min_child_weight": [1, 10, 50]}
    if prep.task == "classification":
        r = float((1 - ys.mean()) / ys.mean())
        space["spw"] = [1.0, round(float(np.sqrt(r)), 3), round(r, 3)]
    return bl.expand(bl.sample_configs(space, cfg.n_configs, cfg.seed), M_GRID, prep.p, cfg.max_iter)


def complexity(hp):
    return bl.drop_penalty(hp) + hp["M"] * (2 ** hp["max_depth"]) * hp["lr"]


def extra(ctx):
    bl.standard_extra(ctx, group_col="lr")
    S, cfg = ctx["store"], ctx["cfg"]
    cols, conf, task = S["cols"], S["conf"], S["task"]
    reg = task == "regression"
    Xtr = np.asarray(ctx["X_tr"][:, cols], np.float32)
    y = ctx["y_fit_tr"] if reg else ctx["y_tr"].astype(int)
    Xva, Xte = np.asarray(ctx["X_va"][:, cols], np.float32), np.asarray(ctx["X_te"][:, cols], np.float32)
    m = build(conf, 3000, task, cfg, y, early=50)
    m.fit(Xtr, y, eval_set=[(Xva, ctx["y_va"])], verbose=False)
    best = int(m.best_iteration) + 1
    if reg:
        pv, pt = m.predict(Xva, iteration_range=(0, best)), m.predict(Xte, iteration_range=(0, best))
        res = {"best_iteration_on_validation": best, "validation_metrics": common.reg_metrics(ctx["y_va"], pv, float(ctx["y_tr"].mean())), "test_metrics": common.reg_metrics(ctx["y_te"], pt, float(ctx["y_tr"].mean())),
               "test_mse_with_CV_chosen_M": ctx["summary"]["test"]["mse"]}
    else:
        pv, pt = m.predict_proba(Xva, iteration_range=(0, best))[:, 1], m.predict_proba(Xte, iteration_range=(0, best))[:, 1]
        thr = common.youden_threshold(ctx["y_va"], pv)
        res = {"best_iteration_on_validation": best, "test_metrics": common.clf_metrics(ctx["y_te"], pt, thr), "test_auc_with_CV_chosen_M": ctx["summary"]["test"]["auc"]}
    res["CV_chosen_M"] = int(S["summary"]["M"])
    (ctx["out"] / "early_stopping_on_validation.json").write_text(json.dumps(common._jsonable(res), indent=2))
    ctx["summary"]["early_stopping_validation"] = {k: v for k, v in res.items() if k in ("best_iteration_on_validation", "CV_chosen_M")}


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=bl.make_predict_path(build, "xgb"), complexity=complexity, extra=extra,
                             select_by="mse" if task == "regression" else "auc", notes="XGBoost sklearn API (hist); M evaluated with iteration_range.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
