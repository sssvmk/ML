"""XGBoost: second-order gradient boosted decision trees.

Model      F(x) = sum_t eta * f_t(x), each tree f_t fitted to the gradient/Hessian of the loss at the current F
Loss       binary logistic loss  L = -[y log p + (1-y) log(1-p)],  p = sigmoid(F)
           + tree complexity penalty  gamma*T + (lambda/2)*sum(leaf weights^2) + alpha*sum|leaf weights|
Stopping   rounds added until log loss on a held-out 10% of TRAIN stops improving for 50 rounds (validation stays untouched)
Why        usually the strongest family on tabular data; native handling of interactions and monotone transforms
"""
from __future__ import annotations

import pandas as pd

from .. import assumptions as A
from .base import AlgorithmModule

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


class Algorithm(AlgorithmModule):
    name = "xgboost_gradient_boosted_trees"
    display_name = "XGBoost"
    family = "boosting"
    description = "Additive ensemble of shallow trees fitted sequentially to the loss gradient, with explicit regularisation."
    complexity_rank = 4
    loss_function = ("Logistic loss + gamma*T + (lambda/2)||w||2^2 + alpha||w||1 (T = leaves, w = leaf weights); "
                     "optimised with second-order (Newton) boosting.")
    optimisation_notes = "Histogram tree method; n_estimators chosen by early stopping; other hyper-parameters by TPE."
    uses_early_stopping = True
    iterative = True
    nominal_encoding = "ordinal"
    scale_inputs = False

    def is_available(self):
        return (xgb is not None, "xgboost is not installed")

    def default_params(self) -> dict:
        return {"learning_rate": 0.1, "max_depth": 6, "min_child_weight": 3.0, "subsample": 0.8,
                "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.01, "gamma": 0.0}

    def hyperparameter_docs(self) -> dict:
        return {"learning_rate": "Shrinkage per tree (eta). Lower = needs more trees but generalises better.",
                "max_depth": "Tree depth = interaction order the model can express.",
                "min_child_weight": "Minimum Hessian mass in a leaf; larger = more conservative.",
                "subsample": "Row fraction per tree (stochastic boosting).",
                "colsample_bytree": "Feature fraction per tree.",
                "reg_lambda": "L2 penalty on leaf weights.", "reg_alpha": "L1 penalty on leaf weights.",
                "gamma": "Minimum loss reduction required to make a split."}

    def search_space(self, trial) -> dict:
        return {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0)}

    def build_estimator(self, params, seed):
        return xgb.XGBClassifier(n_estimators=1500, early_stopping_rounds=50, eval_metric="logloss",
                                 tree_method="hist", objective="binary:logistic", n_jobs=-1, random_state=seed,
                                 verbosity=0, **params)

    def fit_estimator(self, est, X, y, X_es=None, y_es=None, trial=None):
        est.fit(X, y, eval_set=[(X, y), (X_es, y_es)], verbose=False)
        return est

    def training_curve(self, est):
        log = est.evals_result()
        return pd.DataFrame({"iteration": range(len(log["validation_0"]["logloss"])),
                             "train_loss": log["validation_0"]["logloss"], "val_loss": log["validation_1"]["logloss"]})

    def native_importance(self, est, names):
        return pd.Series(est.feature_importances_, index=names)

    def check_assumptions(self, ctx):
        return [A.check_finite_inputs(ctx, blocking=False), A.check_both_classes_present(ctx), A.check_class_balance(ctx),
                A.check_sample_size(ctx), A.check_duplicates(ctx), A.check_single_feature_leakage(ctx),
                A.check_train_validation_shift(ctx)]
