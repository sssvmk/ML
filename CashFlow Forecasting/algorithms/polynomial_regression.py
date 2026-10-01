from __future__ import annotations
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from .base import EligibilityResult
from ._sklearn_template import SklearnLagModule, build_lag_features
from .utils import endogenous_series, exogenous_frame


class PolynomialRegressionModule(SklearnLagModule):
    """
    Polynomial Regression (PRD §3.2 #9). Eligibility: the EXPANDED
    feature matrix (lags + exog, after polynomial expansion) must have
    full column rank.
    """

    name = "polynomial_regression"
    has_eligibility_condition = True
    default_n_lags = 3

    def __init__(self, hyperparameters=None):
        super().__init__(hyperparameters)
        self.degree = int(self.hyperparameters.get("degree", 2))

    def _make_estimator(self):
        return make_pipeline(PolynomialFeatures(degree=self.degree, include_bias=False), LinearRegression())

    def _effective_feature_count(self, n_raw_features: int) -> int:
        # matches PolynomialFeatures(include_bias=False) output width
        from math import comb
        return comb(n_raw_features + self.degree, self.degree) - 1

    def required_observations(self, n_exog: int = 0) -> int:
        return 10 * self._effective_feature_count(self.n_lags + n_exog)  # PRD v10 §3.2 row #9, no floor

    def hyperparameter_search_space(self) -> dict:
        return {
            "n_lags": {"type": "int", "low": 1, "high": 6},
            "degree": {"type": "int", "low": 1, "high": 3},
        }

    def _check_eligibility_impl(self, segment_df) -> EligibilityResult:
        series = endogenous_series(segment_df)
        exog = exogenous_frame(segment_df)
        feats = build_lag_features(series, exog, self.n_lags).dropna()
        if feats.empty:
            return EligibilityResult(False, "no complete rows after lagging")
        X = feats.drop(columns=["y"]).to_numpy()
        Xp = PolynomialFeatures(degree=self.degree, include_bias=False).fit_transform(X)
        if Xp.shape[0] < Xp.shape[1]:
            return EligibilityResult(False, "fewer observations than expanded features")
        rank = np.linalg.matrix_rank(Xp)
        if rank < Xp.shape[1]:
            return EligibilityResult(False, f"expanded feature matrix rank {rank} < {Xp.shape[1]} columns")
        return EligibilityResult(True, "expanded feature matrix has full column rank")
