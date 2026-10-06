"""Random forest: bagged, de-correlated decision trees.

Model      average of B trees, each grown on a bootstrap sample with a random feature subset at every split
Loss       per-split impurity reduction (Gini by default); the ensemble reports the mean leaf class frequency
Why        robust, almost no scaling/encoding needs, captures interactions; typically strong with little tuning
Limits     cannot extrapolate beyond the training range, probabilities are less sharp than boosting's,
           deep trees on ~500k rows are memory hungry (hence the depth/leaf caps in the search space)
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .. import assumptions as A
from .base import AlgorithmModule


class Algorithm(AlgorithmModule):
    name = "random_forest_bagged_trees"
    display_name = "Random forest"
    family = "bagging"
    description = "Bagged ensemble of decorrelated trees; low variance, handles non-linearities and interactions automatically."
    complexity_rank = 3
    loss_function = "Gini impurity (or entropy) minimised greedily at each split; ensemble probability = mean leaf frequency."
    optimisation_notes = "Not gradient-based: trees are grown greedily. Hyper-parameters tuned by TPE on a training subsample."
    nominal_encoding = "ordinal"
    scale_inputs = False

    def default_params(self) -> dict:
        return {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 5, "max_features": "sqrt",
                "max_samples": 0.8, "criterion": "gini"}

    def hyperparameter_docs(self) -> dict:
        return {"n_estimators": "Number of trees (more = lower variance, diminishing returns).",
                "max_depth": "Maximum tree depth (capacity; capped for memory).",
                "min_samples_leaf": "Smallest allowed leaf; larger = smoother, less overfit.",
                "max_features": "Features considered at each split; smaller = more decorrelated trees.",
                "max_samples": "Fraction of rows in each bootstrap sample.",
                "criterion": "Split quality measure."}

    def search_space(self, trial) -> dict:
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 6, 24),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 40, log=True),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", "0.5"]),
                "max_samples": trial.suggest_float("max_samples", 0.4, 1.0),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"])}

    def build_estimator(self, params, seed):
        mf = params["max_features"]
        mf = float(mf) if mf not in ("sqrt", "log2") else mf
        return RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                      min_samples_leaf=params["min_samples_leaf"], max_features=mf,
                                      max_samples=params["max_samples"], criterion=params["criterion"],
                                      n_jobs=-1, random_state=seed)

    def native_importance(self, est, names):
        return pd.Series(est.feature_importances_, index=names)

    def check_assumptions(self, ctx):
        # Trees are non-parametric: the relevant checks are data quality, leakage, shift and range coverage.
        cont = ctx.feature_set.continuous
        tr, va = ctx.X_df[cont], ctx.X_val_df[cont] if ctx.X_val_df is not None else None
        extra = None
        if va is not None and cont:
            lo, hi = tr.min(), tr.max()
            outside = float(((va < lo) | (va > hi)).any(axis=1).mean())
            extra = A._r("validation_inside_training_range", outside < 0.01, outside,
                         f"{outside:.2%} of validation rows have a continuous value outside the training range",
                         "Forests predict a constant outside the training range (no extrapolation).",
                         "Winsorise or choose a model that extrapolates if this share is material.")
        return [A.check_finite_inputs(ctx), A.check_both_classes_present(ctx), A.check_class_balance(ctx),
                A.check_sample_size(ctx), A.check_duplicates(ctx), A.check_single_feature_leakage(ctx),
                A.check_train_validation_shift(ctx), extra]
