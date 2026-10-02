"""
Gradient boosting machine (ESLII 10.10) via scikit-learn GradientBoosting*: trees fit by least squares to the negative gradient (10.37), shrinkage nu = 0.1.
  regression     : loss in {squared error 1/2 (y-f)^2, absolute |y-f|, Huber (10.23, delta = 0.9-quantile of |residual|)}; for the robust losses the model may be fitted on the clipped or the RAW target
  classification : binomial deviance -sum[y' log p + (1-y') log(1-p)] (trees fit to y - p)
Hyper-parameters (CV, random search): M, J (max leaf nodes), loss, fit target, fraction of the ranked variables kept (all unless dropping helps).
Metrics: regression: test MSE +- SE (and MAE - reported separately for robust losses), CV loss vs M, tree size, importance, partial dependence.  classification: test log-loss +- SE, error +- SE, AUC +- SE, calibration,
CV deviance / AUC vs M (one-SE rule), importance, partial dependence.
"""
import _bootstrap  # noqa: F401
import json

import common
import boostlib as bl

NAME = "gbm"
M_GRID = [20, 50, 100, 150, 200]


def build(conf, M, task, cfg, y):
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    import numpy as np
    kw = dict(learning_rate=0.1, n_estimators=M, max_leaf_nodes=conf["J"], random_state=cfg.seed)
    if task == "classification":
        return GradientBoostingClassifier(loss="log_loss", **kw)
    return GradientBoostingRegressor(loss=conf["loss"], alpha=0.9, **kw)               # sklearn's Huber delta = alpha-quantile of |residual|, as in ESLII


def make_grid(cfg, prep, Xs, ys):
    space = {"frac": bl.FRACS, "J": [4, 8]}
    if prep.task == "regression":
        space.update({"loss": ["squared_error", "absolute_error", "huber"], "fit_target": ["clipped", "raw"]})
    return bl.expand(bl.sample_configs(space, cfg.n_configs_sk, cfg.seed), M_GRID, prep.p, cfg.max_iter)


def complexity(hp):
    return bl.drop_penalty(hp) + hp["M"] * hp["J"]


def extra(ctx):
    bl.standard_extra(ctx, group_col="loss" if ctx["prep"].task == "regression" else "J")
    if ctx["prep"].task == "regression":
        import numpy as np
        pd_ = ctx["curve"].copy()
        pd_[["loss", "fit_target", "M", "J", "cv_mse"]].groupby(["loss", "fit_target"])["cv_mse"].min().to_csv(ctx["out"] / "best_cv_mse_by_loss.csv")
        p = ctx["p_te"]
        ctx["summary"]["test_mae"] = float(np.mean(np.abs(ctx["y_te"] - p)))


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=bl.make_predict_path(build, "sk"), complexity=complexity, extra=extra,
                             select_by="mse" if task == "regression" else "auc", notes="scikit-learn GradientBoosting, learning_rate=0.1.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
