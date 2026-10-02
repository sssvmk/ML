"""
Forward stagewise additive modelling (ESLII 10.3-10.4), via scikit-learn GradientBoosting* with NO shrinkage (learning_rate = 1) and STUMPS as the basis b(x; gamma):
  regression     : at each step add the stump that best fits the current residuals, min sum (y - f_{m-1} - b)^2   (10.7, least-squares boosting)
  classification : the same recursion with an arbitrary loss: binomial deviance log(1 + e^(-2yf)) (10.18) or exponential loss exp(-y f) (10.8; = AdaBoost's criterion)
Hyper-parameters (CV, random search): M (iterations, one-SE rule), min samples per leaf, fraction of the ranked variables kept (all kept unless dropping helps), [loss].
Metrics: regression: test MSE +- SE, CV MSE vs M, training vs test MSE curves.  classification: test error +- SE, AUC, log-loss, CV vs M, training vs test deviance / AUC curves.
"""
import _bootstrap  # noqa: F401
import json

import common
import boostlib as bl

NAME = "forward_stagewise"
M_GRID = [5, 10, 20, 40, 80, 120, 160, 200]


def build(conf, M, task, cfg, y):
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    kw = dict(learning_rate=1.0, n_estimators=M, max_depth=1, min_samples_leaf=conf["min_leaf"], random_state=cfg.seed)
    return GradientBoostingRegressor(loss="squared_error", subsample=1.0, **kw) if task == "regression" else GradientBoostingClassifier(loss=conf["loss"], **kw)


def make_grid(cfg, prep, Xs, ys):
    space = {"frac": bl.FRACS, "min_leaf": [20, 100]}
    if prep.task == "classification":
        space["loss"] = ["log_loss", "exponential"]
    return bl.expand(bl.sample_configs(space, cfg.n_configs_sk, cfg.seed), M_GRID, prep.p, cfg.max_iter)


def complexity(hp):
    return bl.drop_penalty(hp) + hp["M"]


def extra(ctx):
    bl.standard_extra(ctx, group_col="min_leaf", pdp=False)
    df, secs = bl.staged_curves(ctx, build, "sk", max(bl.grid_m(M_GRID, ctx["cfg"].max_iter)))
    df.to_csv(ctx["out"] / "train_vs_test_curves.csv", index=False)
    bl.plot_curves(df, ctx["out"] / "train_vs_test_curves.png", "forward stagewise: training vs test")


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=bl.make_predict_path(build, "sk"), complexity=complexity, extra=extra,
                             select_by="mse" if task == "regression" else "auc", notes="scikit-learn GradientBoosting, learning_rate=1, stumps (forward stagewise).", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
