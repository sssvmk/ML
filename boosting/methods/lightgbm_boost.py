"""
LightGBM (not in ESLII) - prebuilt `lightgbm` package: gradient boosting on histogram-binned features with LEAF-WISE tree growth and L2 regularisation on the leaf values.
  regression     : objective L2 (squared error), L1 (absolute) or Huber (delta = 0.9-quantile of |y - median|); robust objectives may be trained on the clipped or the RAW target
  classification : binary logistic loss; scale_pos_weight in {1, sqrt(neg/pos), neg/pos} for the class imbalance (judged by AUC)
Hyper-parameters (CV, random search): num_leaves, learning_rate, min_child_samples, reg_lambda, subsample (bagging), colsample_bytree, M (iterations: the CV curve over M replaces
early stopping inside CV), fraction of the ranked variables kept (all unless dropping helps).
Early stopping on the VALIDATION set (never the test set) is reported separately: best iteration, and its test metrics next to the CV-chosen model's.
Metrics: regression: test MSE +- SE (MAE reported for robust objectives), CV MSE over the grid, gain importance, partial dependence.  classification: test log-loss +- SE, error +- SE, AUC +- SE, calibration, CV AUC vs M.
"""
import _bootstrap  # noqa: F401
import json

import numpy as np

import common
import boostlib as bl

NAME = "lightgbm"
M_GRID = [20, 50, 100, 200, 300, 500, 800, 1200]


def build(conf, M, task, cfg, y):
    import lightgbm as lgb
    kw = dict(n_estimators=M, num_leaves=conf["num_leaves"], learning_rate=conf["lr"], min_child_samples=conf["min_child"], reg_lambda=conf["lam"], subsample=conf["subsample"],
              subsample_freq=1 if conf["subsample"] < 1 else 0, colsample_bytree=conf["colsample"], n_jobs=cfg.n_jobs, random_state=cfg.seed, verbose=-1)
    if task == "regression":
        extra = {"alpha": float(np.quantile(np.abs(y - np.median(y)), 0.9))} if conf["objective"] == "huber" else {}
        return lgb.LGBMRegressor(objective=conf["objective"], **kw, **extra)
    return lgb.LGBMClassifier(objective="binary", scale_pos_weight=conf["spw"], **kw)


def make_grid(cfg, prep, Xs, ys):
    space = {"frac": bl.FRACS, "num_leaves": [7, 15, 31, 63], "lr": [0.02, 0.05, 0.1], "min_child": [20, 50, 100, 200], "lam": [0, 1, 10, 50], "subsample": [0.5, 0.8, 1.0],
             "colsample": [0.3, 0.6, 1.0]}
    if prep.task == "regression":
        space.update({"objective": ["regression", "regression_l1", "huber"], "fit_target": ["clipped", "raw"]})
    else:
        r = float((1 - ys.mean()) / ys.mean())
        space["spw"] = [1.0, round(float(np.sqrt(r)), 3), round(r, 3)]
    return bl.expand(bl.sample_configs(space, cfg.n_configs, cfg.seed), M_GRID, prep.p, cfg.max_iter)


def complexity(hp):
    return bl.drop_penalty(hp) + hp["M"] * hp["num_leaves"] * hp["lr"]


def extra(ctx):
    import lightgbm as lgb
    bl.standard_extra(ctx, group_col="lr")
    S, cfg = ctx["store"], ctx["cfg"]
    cols, conf, task = S["cols"], S["conf"], S["task"]
    reg = task == "regression"
    Xtr = np.asarray(ctx["X_tr"][:, cols], np.float32)
    y = ctx["y_fit_tr"] if reg else ctx["y_tr"].astype(int)
    Xva, Xte = np.asarray(ctx["X_va"][:, cols], np.float32), np.asarray(ctx["X_te"][:, cols], np.float32)
    m = build(conf, 3000, task, cfg, y)
    m.set_params(metric="l2" if reg else "auc")
    m.fit(Xtr, y, eval_set=[(Xva, ctx["y_va"])], callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)])
    best = int(m.best_iteration_ or m.n_estimators)
    if reg:
        pv, pt = m.predict(Xva, num_iteration=best), m.predict(Xte, num_iteration=best)
        res = {"best_iteration_on_validation": best, "validation_metrics": common.reg_metrics(ctx["y_va"], pv, float(ctx["y_tr"].mean())), "test_metrics": common.reg_metrics(ctx["y_te"], pt, float(ctx["y_tr"].mean()))}
        res["test_mse_with_CV_chosen_M"] = ctx["summary"]["test"]["mse"]
    else:
        pv, pt = m.predict_proba(Xva, num_iteration=best)[:, 1], m.predict_proba(Xte, num_iteration=best)[:, 1]
        thr = common.youden_threshold(ctx["y_va"], pv)
        res = {"best_iteration_on_validation": best, "test_metrics": common.clf_metrics(ctx["y_te"], pt, thr), "test_auc_with_CV_chosen_M": ctx["summary"]["test"]["auc"]}
    res["CV_chosen_M"] = int(conf.get("M", S["summary"]["M"]))
    (ctx["out"] / "early_stopping_on_validation.json").write_text(json.dumps(common._jsonable(res), indent=2))
    ctx["summary"]["early_stopping_validation"] = {k: v for k, v in res.items() if k in ("best_iteration_on_validation", "CV_chosen_M")}
    imp = m.booster_.feature_importance("split")
    import pandas as pd
    pd.DataFrame({"variable": np.array(ctx["prep"].feature_names)[cols], "gain": m.booster_.feature_importance("gain"), "split_count": imp}).sort_values("gain", ascending=False).to_csv(ctx["out"] / "importance_gain_and_split.csv", index=False)


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=bl.make_predict_path(build, "lgb"), complexity=complexity, extra=extra,
                             select_by="mse" if task == "regression" else "auc", notes="LightGBM sklearn API; M evaluated with num_iteration.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
