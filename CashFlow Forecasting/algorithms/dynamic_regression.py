from __future__ import annotations
from ._statsmodels_template import StatsmodelsARIMALikeModule


class DynamicRegressionModule(StatsmodelsARIMALikeModule):
    """
    Dynamic Regression (PRD §3.2 #4): regression with ARMA(p,q) errors,
    implemented as SARIMAX with exogenous regressors and no seasonal
    component -- mathematically the same estimation problem as ARIMAX
    with exog, but the ARMA order defaults to no differencing (the
    series is assumed level-stationary, not the differenced series).
    Eligibility: series stationary (the ARMA-error term requires it).
    """

    name = "dynamic_regression"
    has_eligibility_condition = True
    default_order = (1, 0, 1)
    default_seasonal_order = (0, 0, 0, 0)
    min_observations_floor = 50  # PRD v10 §3.2 row #4: N >= max(50, 10*(k+p+q+1))
