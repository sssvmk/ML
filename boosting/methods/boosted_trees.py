"""
Boosted trees (ESLII 10.9) via scikit-learn GradientBoosting* with no shrinkage (learning_rate = 1): the basis is a J-terminal-node tree T(x; Theta) with constants gamma_jm in its regions.
  regression     : squared error  sum (y - f_{m-1} - T)^2
  classification : binomial deviance (10.18) or exponential loss (10.8) - hyper-parameter `loss`
Hyper-parameters (CV, random search): M (iterations), J (max leaf nodes: 4 / 6 / 8, 'right-sized trees', 10.11), fraction of the ranked variables kept (all unless dropping helps).
Metrics: regression: test MSE +- SE, CV MSE vs (M, J), relative variable importance (10.13.1), partial dependence.  classification: test error +- SE, AUC, log-loss, calibration, CV vs (M, J), importance.
"""
import _bootstrap  # noqa: F401
import json

import common
import boostlib as bl

NAME = "boosted_trees"
M_GRID = [2, 5, 10, 20, 40, 80]


def build(conf, M, task, cfg, y):
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    kw = dict(learning_rate=1.0, n_estimators=M, max_leaf_nodes=conf["J"], random_state=cfg.seed)
    return GradientBoostingRegressor(loss="squared_error", **kw) if task == "regression" else GradientBoostingClassifier(loss=conf["loss"], **kw)


def make_grid(cfg, prep, Xs, ys):
    space = {"frac": bl.FRACS, "J": [4, 6, 8]}
    if prep.task == "classification":
        space["loss"] = ["log_loss", "exponential"]
    return bl.expand(bl.sample_configs(space, cfg.n_configs_sk, cfg.seed), M_GRID, prep.p, cfg.max_iter)


def complexity(hp):
    return bl.drop_penalty(hp) + hp["M"] * hp["J"]


def extra(ctx):
    bl.standard_extra(ctx, group_col="J")


def run(prep_dir, out_dir, cfg, progress=None, console=False):
    task = json.loads((common.Path(prep_dir) / "prepared_meta.json").read_text())["task"]
    return common.run_method(NAME, prep_dir, out_dir, cfg, progress, make_grid=make_grid, predict_path=bl.make_predict_path(build, "sk"), complexity=complexity, extra=extra,
                             select_by="mse" if task == "regression" else "auc", notes="scikit-learn GradientBoosting, learning_rate=1, J-leaf trees.", console=console)


if __name__ == "__main__":
    common.method_main(NAME, run)
