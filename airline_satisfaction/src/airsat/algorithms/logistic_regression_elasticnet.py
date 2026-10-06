"""Logistic regression with elastic-net penalty: the interpretable linear baseline.

Model      p(satisfied | x) = sigmoid(b + w.x)
Loss       mean binary cross-entropy (negative log-likelihood)  +  (1/C) * [ a*|w|_1 + (1-a)/2*|w|_2^2 ]   (a = l1_ratio)
Optimiser  SAGA (stochastic average gradient; supports L1/L2/elastic-net and scales to ~10^6 rows)
Needs      one-hot nominals + standardised numerics, linear log-odds, no perfect collinearity/separation,
           independent rows -> all checked in check_assumptions().
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression

from .. import assumptions as A
from .base import AlgorithmModule

_NEW_API = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)


class Algorithm(AlgorithmModule):
    name = "logistic_regression_elasticnet"
    display_name = "Logistic regression (elastic-net)"
    family = "linear"
    description = ("Generalised linear model for the log-odds of satisfaction. Fast, calibrated, interpretable; "
                   "cannot learn interactions or non-linearities unless they are engineered into the features.")
    complexity_rank = 1
    loss_function = ("Regularised binary cross-entropy: -(1/n) sum[y log p + (1-y) log(1-p)] + (1/C)[l1_ratio*||w||1 + "
                     "(1-l1_ratio)/2*||w||2^2]")
    optimisation_notes = "SAGA solver, convex objective -> a unique optimum; tuned by TPE over C, l1_ratio, class_weight."
    nominal_encoding = "onehot"
    scale_inputs = True

    def default_params(self) -> dict:
        return {"C": 1.0, "l1_ratio": 0.5, "class_weight": "none"}

    def hyperparameter_docs(self) -> dict:
        return {"C": "Inverse regularisation strength (smaller = stronger shrinkage). Log-uniform 1e-3..1e2.",
                "l1_ratio": "Mix of L1 (sparsity, 1.0) and L2 (shrinkage, 0.0).",
                "class_weight": "'balanced' re-weights classes; mostly irrelevant when classes are balanced."}

    def search_space(self, trial) -> dict:
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                "class_weight": trial.suggest_categorical("class_weight", ["none", "balanced"])}

    def build_estimator(self, params, seed):
        kwargs = dict(C=params["C"], solver="saga", max_iter=300, tol=1e-3, random_state=seed,
                      class_weight=None if params.get("class_weight", "none") == "none" else "balanced")
        if _NEW_API:
            kwargs["l1_ratio"] = params["l1_ratio"]
        else:  # scikit-learn < 1.8
            kwargs.update(penalty="elasticnet", l1_ratio=params["l1_ratio"])
        return LogisticRegression(**kwargs)

    def fit_estimator(self, est, X, y, X_es=None, y_es=None, trial=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # saga ConvergenceWarning at tol=1e-3 is expected and harmless
            est.fit(X, y)
        return est

    def native_importance(self, est, names):
        return pd.Series(np.abs(est.coef_[0]), index=names)

    def check_assumptions(self, ctx):
        checks = [A.check_finite_inputs(ctx), A.check_both_classes_present(ctx), A.check_class_balance(ctx),
                  A.check_sample_size(ctx), A.check_duplicates(ctx), A.check_single_feature_leakage(ctx),
                  A.check_train_validation_shift(ctx), A.check_events_per_variable(ctx), A.check_multicollinearity(ctx),
                  A.check_linearity_of_logit(ctx), A.check_no_separation(ctx), A.check_residual_independence(ctx)]
        return checks
