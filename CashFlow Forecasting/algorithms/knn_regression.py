from __future__ import annotations
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor

from .base import EligibilityResult
from ._sklearn_template import SklearnLagModule
from .utils import endogenous_series


class KNNRegressionModule(SklearnLagModule):
    """
    KNN Regression (PRD §3.2 #19). Eligibility: no significant trend --
    KNN has no extrapolation mechanism, so a trending series would be
    forecast as flat/wrong beyond the training range. Checked via an OLS
    slope-significance test on the endogenous series.
    """

    name = "knn_regression"
    has_eligibility_condition = False  # G-15: PRD row #19 says 'None'; the N >= max(200, 20*dims*k) floor still gates it
    default_n_lags = 5
    trend_alpha = 0.05

    def __init__(self, hyperparameters=None):
        super().__init__(hyperparameters)
        self.n_neighbors = int(self.hyperparameters.get("n_neighbors", 5))

    def _make_estimator(self):
        return KNeighborsRegressor(n_neighbors=self.n_neighbors)

    def required_observations(self, n_exog: int = 0) -> int:
        dims = self.n_lags + n_exog
        return max(200, 20 * dims * self.n_neighbors)  # PRD v10 §3.2 row #19

    def hyperparameter_search_space(self) -> dict:
        return {
            "n_neighbors": {"type": "int", "low": 2, "high": 15},
            "n_lags": {"type": "int", "low": 3, "high": 14},
        }

    def _check_eligibility_impl(self, segment_df) -> EligibilityResult:
        series = endogenous_series(segment_df)
        t = range(len(series))
        slope, intercept, r, p, se = stats.linregress(list(t), series.to_numpy())
        if p < self.trend_alpha:
            return EligibilityResult(False, f"significant trend detected (OLS slope p={p:.4f} < {self.trend_alpha})")
        return EligibilityResult(True, f"no significant trend (OLS slope p={p:.4f})")
