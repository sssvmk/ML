"""LightGBM: histogram-based gradient boosting with leaf-wise tree growth and GOSS/EFB optimisations.

Same objective family as XGBoost (logistic loss + regularisation) but trees grow leaf-wise (best-first), which
reaches lower loss with fewer leaves, and training is typically several times faster on ~10^6 rows.
Included next to XGBoost because the two make different inductive choices (level-wise vs leaf-wise growth) and
neither dominates on every dataset - the experiment decides.
"""
from __future__ import annotations

import warnings

import pandas as pd

from .. import assumptions as A
from .base import AlgorithmModule

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


class Algorithm(AlgorithmModule):
    name = "lightgbm_gradient_boosted_trees"
    display_name = "LightGBM"
    family = "boosting"
    description = "Leaf-wise gradient boosting on histogram-binned features; fast and strong on large tabular data."
    complexity_rank = 4
    loss_function = "Binary logistic loss + L1/L2 leaf-weight penalties (reg_alpha, reg_lambda) + min_split_gain."
    optimisation_notes = "Leaf-wise growth capped by num_leaves; n_estimators by early stopping on 10% of TRAIN; rest by TPE."
    uses_early_stopping = True
    iterative = True
    nominal_encoding = "ordinal"
    scale_inputs = False

    def is_available(self):
        return (lgb is not None, "lightgbm is not installed")

    def default_params(self) -> dict:
        return {"learning_rate": 0.1, "num_leaves": 63, "min_child_samples": 20, "subsample": 0.8,
                "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.01, "min_split_gain": 0.0}

    def hyperparameter_docs(self) -> dict:
        return {"learning_rate": "Shrinkage per tree.", "num_leaves": "Leaves per tree (main capacity control for leaf-wise growth).",
                "min_child_samples": "Minimum rows per leaf; larger = smoother.", "subsample": "Row fraction per tree (bagging).",
                "colsample_bytree": "Feature fraction per tree.", "reg_lambda": "L2 penalty.", "reg_alpha": "L1 penalty.",
                "min_split_gain": "Minimum gain to split."}

    def search_space(self, trial) -> dict:
        return {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 8, 255, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0)}

    def build_estimator(self, params, seed):
        return lgb.LGBMClassifier(objective="binary", n_estimators=1500, subsample_freq=1, n_jobs=-1,
                                  random_state=seed, verbosity=-1, **params)

    def fit_estimator(self, est, X, y, X_es=None, y_es=None, trial=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # lightgbm>=4.7 deprecates eval_set in favour of eval_X/eval_y
            est.fit(X, y, eval_set=[(X, y), (X_es, y_es)], eval_names=["fit", "es"], eval_metric="binary_logloss",
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        return est

    def training_curve(self, est):
        log = est.evals_result_
        return pd.DataFrame({"iteration": range(len(log["fit"]["binary_logloss"])),
                             "train_loss": log["fit"]["binary_logloss"], "val_loss": log["es"]["binary_logloss"]})

    def native_importance(self, est, names):
        return pd.Series(est.booster_.feature_importance(importance_type="gain"), index=names)

    def check_assumptions(self, ctx):
        return [A.check_finite_inputs(ctx, blocking=False), A.check_both_classes_present(ctx), A.check_class_balance(ctx),
                A.check_sample_size(ctx), A.check_duplicates(ctx), A.check_single_feature_leakage(ctx),
                A.check_train_validation_shift(ctx)]
