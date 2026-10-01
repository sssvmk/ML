"""Acceptance checks for the non-neural gaps (matrix 'Gaps & Deviations' tab)."""
import numpy as np
import pytest
from adapters.synthetic import SyntheticAdapter
from algorithms.arimax import ARIMAXModule
from algorithms.sarimax import SARIMAXModule
from algorithms.dynamic_regression import DynamicRegressionModule
from algorithms.svr import SVRModule
from algorithms.knn_regression import KNNRegressionModule
from algorithms.ridge import RidgeModule
from algorithms.xgboost_module import XGBoostModule
from algorithms.lightgbm_module import LightGBMModule
from algorithms.catboost_module import CatBoostModule
from algorithms.prophet_model import ProphetModule
from backtest import rolling_backtest
from search import random_then_adaptive_search
from contract import validate_contract, Dataset


def head(d, n_dates):
    """first n_dates calendar days of a long-format contract frame (rows are sorted by dataset, so iloc would slice datasets)"""
    keep = sorted(d["date"].unique())[:n_dates]
    return d[d["date"].isin(keep)]


@pytest.fixture(scope="module")
def df():
    return SyntheticAdapter().extract(n_days=900)


# G-05 / G-04 / G-06 / G-02 ----------------------------------------------------------------
def test_g04_joint_window_and_hyperparameters(df):
    sr = random_then_adaptive_search(RidgeModule, {"n_lags": 3, "alpha": 1.0}, df, window=800, horizon=4,
                                     budget=10, seed=1, window_bounds=(1, 800), n_exog=2, step=60)
    assert sr.best_window is not None
    windows = {t.window for t in sr.trials if t.window}
    assert len(windows) > 3                      # window really varied
    assert sr.best_window != 800 or len(windows) > 1
    # every sampled window respects the sampled configuration's own required_observations
    for t in sr.trials:
        if t.window:
            assert t.window >= RidgeModule({**{"n_lags": 3, "alpha": 1.0}, **t.hyperparameters}).required_observations(2)


def test_g05_fold_level_pruning_aborts_midway(df):
    # a clearly bad configuration must terminate after its first folds, with the reason recorded
    seen = []

    def pruner(n_ok, running):
        seen.append(n_ok)
        return "test: clearly inferior" if n_ok >= 2 else None

    r = rolling_backtest(lambda: RidgeModule({"n_lags": 1}), df, 300, 4, step=20, pruner=pruner)
    assert r.pruned and r.prune_reason == "test: clearly inferior"
    assert r.n_folds == 2 and max(seen) == 2


def test_g05_search_records_fold_level_pruned_status(df):
    sr = random_then_adaptive_search(RidgeModule, {"n_lags": 3}, df, window=600, horizon=4, budget=14, seed=3,
                                     window_bounds=(1, 600), n_exog=2, step=30,
                                     pruner_cfg={"n_startup_trials": 2, "n_warmup_folds": 1, "margin": 1.05})
    reasons = [t.reason for t in sr.trials if t.status == "pruned"]
    assert any("fold-level" in (x or "") for x in reasons), reasons


def test_g06_seasonal_terms_searched_with_constraints():
    space = SARIMAXModule({}).hyperparameter_search_space()
    assert {"P", "D", "Q", "m"} <= set(space)
    assert {"P", "D", "Q", "m"} <= set(DynamicRegressionModule({}).hyperparameter_search_space())
    m = SARIMAXModule({"p": 3, "d": 0, "q": 3, "P": 2, "D": 0, "Q": 2, "m": 7})
    k = 2
    assert m.required_observations(k) == max(100, 10 * (3 + 3 + 2 + 2 + k + 1))     # N >= max(100, 10*(p+q+P+Q+k+1))
    assert not m.is_valid_config(available_observations=m.required_observations(k) - 1, n_exog=k)[0]
    assert m.is_valid_config(available_observations=m.required_observations(k), n_exog=k)[0]
    assert SARIMAXModule({"p": 1, "d": 0, "q": 1, "P": 0, "D": 0, "Q": 0, "m": 30}).seasonal_order == (0, 0, 0, 0)
    assert not SARIMAXModule({"p": 0, "d": 0, "q": 0, "P": 1, "D": 1, "Q": 1, "m": 30}).is_valid_config(80, 0)[0]


# G-07 charts / G-08 diagnostics / G-10 gap ------------------------------------------------
def test_g07_evaluate_populates_charts(df, tmp_path):
    hist = head(df, 890)
    m = RidgeModule({"n_lags": 7}); m.train(hist)
    fc = m.infer(hist, 4)
    actual = df[(df["series_role"] == "endogenous") & (df["date"].isin(fc["date"]))][["date", "value"]]
    assert len(actual) == 4
    ev = m.evaluate(fc, actual, chart_dir=tmp_path)
    assert {"forecast_vs_actual", "residuals"} <= set(ev.charts)
    for p in ev.charts.values():
        assert (tmp_path / p.split("/")[-1]).exists()


@pytest.mark.parametrize("cls,hp", [(ARIMAXModule, {"order": [1, 0, 1]}), (RidgeModule, {}), (LightGBMModule, {}),
                                    (ProphetModule, {})])
def test_g08_diagnose_ran_with_ljung_box(df, cls, hp):
    m = cls(hp); m.train(head(df, 300))
    d = m.diagnose()
    assert d.ran and "ljung_box" in d.details
    v = next(iter(d.details["ljung_box"].values()))
    assert set(v) == {"statistic", "p_value"}


def test_g10_train_validation_gap_present(df):
    r = rolling_backtest(lambda: RidgeModule({"n_lags": 7}), df, 500, 4, step=100)
    assert r.overfit["folds_with_train_metrics"] >= 1
    for k in ("train_mase", "validation_mase", "gap_mase"):
        assert r.overfit[k] is not None
    assert r.overfit["gap_mase"] == pytest.approx(r.overfit["validation_mase"] - r.overfit["train_mase"], abs=1e-9)


# G-09 early stopping -----------------------------------------------------------------------
@pytest.mark.parametrize("cls,extra", [(XGBoostModule, {"n_estimators": 500}), (LightGBMModule, {"n_estimators": 500}),
                                       (CatBoostModule, {"iterations": 500})])
def test_g09_gbm_early_stopping_best_iteration_below_max(df, cls, extra):
    m = cls({"learning_rate": 0.3, **extra}); m.train(df)
    es = m._fitted_model["early_stopping"]
    assert es["used"] and es["best_iteration"] < es["n_estimators"] and es["validation_rows"] >= 10


# G-11 SVR compute ceiling ------------------------------------------------------------------
def test_g11_svr_ceiling_policy_is_explicit(df):
    n = len(df["date"].unique())
    assert SVRModule({}).check_eligibility(df).eligible                       # unset: not enforced
    assert not SVRModule({"compute_ceiling": 300, "ceiling_action": "ineligible"}).check_eligibility(df).eligible
    m = SVRModule({"compute_ceiling": 300, "ceiling_action": "limit_window"}); m.train(df)
    assert len(m._resid) <= 300                                               # trained on the most recent 300 only
    with pytest.raises(ValueError):                                           # ceiling without an action is refused, never guessed
        SVRModule({"compute_ceiling": 300}).check_eligibility(df)


def test_g11_svr_ceiling_configured_value():
    from config import load_config
    hp = load_config("config.json")["algorithms"]["svr"]["hyperparameters"]
    assert hp["compute_ceiling"] == 10000 and hp["ceiling_action"] == "limit_window"    # your decision
    over = SyntheticAdapter().extract(n_days=10050)
    m = SVRModule({**hp, "n_lags": 5}); m.train(over)
    assert len(m._resid) <= 10000


# G-12 Prophet --------------------------------------------------------------------------------
def test_g12_prophet_is_real_library_with_changepoints(df):
    m = ProphetModule({"changepoint_prior_scale": 0.2}); m.train(head(df, 200))
    from prophet import Prophet
    assert isinstance(m._prophet, Prophet)
    assert m._prophet.changepoint_prior_scale == 0.2 and len(m.changepoints) > 0
    assert m._fitted_model["trend"] == "piecewise-linear"


# G-13 synthetic labels -----------------------------------------------------------------------
def test_g13_synthetic_uses_prd_datasets_for_ar_and_ap():
    segs = SyntheticAdapter().extract_all_processes(n_days=120)
    ar, ap = segs["1000-INR-inflow"], segs["1000-INR-outflow"]
    endo = lambda d: set(d[d.series_role == "endogenous"].dataset)
    exo = lambda d: set(d[d.series_role == "exogenous"].dataset)
    assert endo(ar) == {"Bank_Inflow_EBS"} and exo(ar) == {"AR_Actual_Cleared", "AR_Expected_Unpaid"}
    assert endo(ap) == {"Bank_Outflow_EBS"} and exo(ap) == {"AP_Actual_Cleared", "AP_Expected_Unpaid"}
    assert validate_contract(ar).ok and validate_contract(ap).ok


# G-15 SVR / KNN eligibility ------------------------------------------------------------------
def test_g15_svr_knn_no_necessary_condition_but_floors_remain():
    from algorithms.catalog import ALGORITHM_CATALOG, load_class
    flagged = sorted(n for r in [ALGORITHM_CATALOG] for n in r if load_class(n).has_eligibility_condition)
    assert len(flagged) == 13, flagged
    assert not SVRModule.has_eligibility_condition and not KNNRegressionModule.has_eligibility_condition
    assert SVRModule({}).required_observations() == 100
    assert KNNRegressionModule({"n_lags": 5, "n_neighbors": 5}).required_observations(2) == max(200, 20 * 7 * 5)


# G-14 nested lineage ---------------------------------------------------------------------------
def test_g14_lineage_is_a_nested_object_and_validator_enforces_it(df):
    from contract import CONTRACT_COLUMNS, LINEAGE_FIELDS, lineage_field, upgrade_flat_lineage
    assert "lineage" in CONTRACT_COLUMNS
    assert not set(LINEAGE_FIELDS) & set(CONTRACT_COLUMNS)                  # no flat provenance columns
    assert set(df["lineage"].iloc[0]) == set(LINEAGE_FIELDS)
    assert set(lineage_field(df, "source_adapter")) == {"synthetic"}
    assert validate_contract(df).ok
    broken = df.copy()
    broken.at[0, "lineage"] = {"source_adapter": "x"}                        # incomplete object
    r = validate_contract(broken)
    assert not r.ok and any("lineage" in e for e in r.errors)
    flat = df.copy()
    for f in LINEAGE_FIELDS:
        flat[f] = lineage_field(df, f)
    flat = flat.drop(columns=["lineage"])
    assert not validate_contract(flat).ok                                    # flat form is not silently accepted
    assert validate_contract(upgrade_flat_lineage(flat)).ok                  # but has an explicit migration
