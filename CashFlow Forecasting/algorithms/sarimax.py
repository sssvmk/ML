from __future__ import annotations
from ._statsmodels_template import StatsmodelsARIMALikeModule


class SARIMAXModule(StatsmodelsARIMALikeModule):
    """SARIMAX (PRD §3.2 #2): same stationarity eligibility as ARIMAX,
    extended with a seasonal order."""

    name = "sarimax"
    has_eligibility_condition = True
    default_order = (1, 1, 1)
    default_seasonal_order = (1, 0, 1, 7)  # weekly seasonality by default
    min_observations_floor = 100  # PRD v10 §3.2 row #2: N >= max(100, 10*(p+q+P+Q+k+1))
