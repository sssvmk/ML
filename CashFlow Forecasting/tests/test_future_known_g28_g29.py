"""G-28 TFT / G-29 TiDE and the future-known-covariate infrastructure (decision: inferred from STRUCTURE only)."""
import numpy as np
import pandas as pd
import pytest
import torch.nn as nn
from adapters.synthetic import SyntheticAdapter
from algorithms.base import AlgorithmModule
from algorithms.seasonal_naive import SeasonalNaiveModule
from algorithms.tft import TFTModule
from algorithms.tide import TiDEModule
from algorithms.utils import future_known_datasets, future_length
from backtest import rolling_backtest
from pooling import assemble_pool

H = 4
FAST_TFT = {"n_lags": 28, "max_steps": 20, "val_check_steps": 10, "horizon": H, "hidden_size": 16, "n_head": 2}
FAST_TIDE = {"n_lags": 28, "max_steps": 20, "val_check_steps": 10, "horizon": H, "hidden_size": 32}


@pytest.fixture(scope="module")
def plain():
    return SyntheticAdapter().extract(n_days=900)


@pytest.fixture(scope="module")
def fk():
    return SyntheticAdapter().extract(n_days=900, future_days=8)


def _pool(future_days=8, n_days=1100):
    spec = [("1000", "INR", "AR", 1), ("1000", "INR", "AP", 2), ("2000", "EUR", "AR", 3), ("2000", "JPY", "AP", 4)]
    out = {}
    for cc, ccy, proc, seed in spec:
        d = SyntheticAdapter().extract(company_code=cc, currency=ccy, process=proc, n_days=n_days, seed=seed, future_days=future_days)
        out[d["segment_id"].iloc[0]] = d
    return out


# ------------------------------------------------------------------------------------- detection is structural only
def test_detection_uses_dates_only_never_names(fk):
    assert future_known_datasets(fk) == ["AR_Expected_Unpaid"] and future_length(fk, "AR_Expected_Unpaid") == 8
    renamed = fk.copy()
    renamed["dataset"] = renamed["dataset"].map(lambda d: {"AR_Expected_Unpaid": "zzz_anything", "AR_Actual_Cleared": "aaa_other",
                                                          "Bank_Inflow_EBS": "target"}[d])
    assert future_known_datasets(renamed) == ["zzz_anything"]                 # same answer under arbitrary names
    assert future_known_datasets(fk, min_future=9) == []                       # needs enough values
    assert future_known_datasets(SyntheticAdapter().extract(n_days=200)) == []


# ------------------------------------------------------------------------------------- backtest: no leakage, fk through test window
class _Spy(SeasonalNaiveModule):
    seen: list = []

    def train(self, segment_df):
        type(self).seen.append(segment_df)
        super().train(segment_df)


def test_backtest_hands_future_known_through_the_test_window_and_nothing_else(fk):
    _Spy.seen = []
    r = rolling_backtest(lambda: _Spy({"season": 7}), fk, window=300, horizon=H, step=200)
    assert r.n_folds >= 2 and len(_Spy.seen) == r.n_folds
    for tdf in _Spy.seen:
        endo_end = tdf.loc[tdf["series_role"] == "endogenous", "date"].max()
        ex = tdf[tdf["series_role"] == "exogenous"]
        hist = ex[ex["dataset"] == "AR_Actual_Cleared"]
        known = ex[ex["dataset"] == "AR_Expected_Unpaid"]
        assert hist["date"].max() <= endo_end                                  # historical exogenous: cut at the training end
        after = known[known["date"] > endo_end]["date"]
        assert len(after) == H and after.max() == endo_end + pd.Timedelta(days=H)   # future-known: exactly through the test window


def test_backtest_folds_are_counted_on_target_dates_only(plain, fk):
    a = rolling_backtest(lambda: SeasonalNaiveModule({"season": 7}), plain, window=300, horizon=H, step=100)
    b = rolling_backtest(lambda: SeasonalNaiveModule({"season": 7}), fk, window=300, horizon=H, step=100)
    assert a.n_folds == b.n_folds                                              # future-only dates do not create folds


def test_in_sample_check_keeps_future_known_for_the_checked_periods(fk):
    m = TiDEModule(dict(FAST_TIDE)); m.train(fk)
    hist = fk[fk["date"] <= fk[fk["series_role"] == "endogenous"]["date"].max()]
    assert m.in_sample_forecast_check(fk, H) is not None                       # would be None if the covariates were cut


# ------------------------------------------------------------------------------------- eligibility is derived
@pytest.mark.parametrize("cls,hp", [(TFTModule, FAST_TFT), (TiDEModule, FAST_TIDE)])
def test_eligibility_is_derived_from_the_data(cls, hp, plain, fk):
    m = cls(dict(hp))
    assert not hasattr(m, "has_future_known_exog")                              # the manual flag is gone
    r = m.check_eligibility(plain)
    assert not r.eligible and "no future-known covariate series" in r.reason
    cal = cls({**hp, "use_calendar": True})
    assert not cal.check_eligibility(plain).eligible                           # calendar features alone do not qualify
    short = SyntheticAdapter().extract(n_days=900, future_days=H - 1)
    assert not m.check_eligibility(short).eligible                             # fewer future values than the horizon
    ok = m.check_eligibility(fk)
    assert ok.eligible and "1 future-known" in ok.reason
    assert m.required_observations() == max(1000, 28 + H + 500)


# ------------------------------------------------------------------------------------- TFT (G-28)
@pytest.fixture(scope="module")
def tft(fk):
    m = TFTModule(dict(FAST_TFT)); m.train(fk); return m


def test_tft_architecture_components(tft):
    names = {type(x).__name__ for x in tft.network().modules()}
    assert {"VariableSelectionNetwork", "GRN", "GLU", "InterpretableMultiHeadAttention", "TemporalFusionDecoder",
            "TemporalCovariateEncoder", "TFTEmbedding"} <= names
    assert sum(type(x).__name__ == "VariableSelectionNetwork" for x in tft.network().modules()) >= 2   # history + future
    assert any(isinstance(x, nn.LSTM) for x in tft.network().modules())
    assert "fk_0" in tft.network().hparams["futr_exog_list"]


def test_tft_uses_future_known_values_and_disabling_them_changes_the_graph(tft, fk):
    base = tft.infer(fk, H)["forecast"].to_numpy()
    edited = fk.copy()
    m = (edited["dataset"] == "AR_Expected_Unpaid") & (edited["date"] > edited[edited["series_role"] == "endogenous"]["date"].max())
    edited.loc[m, "value"] = edited.loc[m, "value"] * 5 + 100
    assert not np.allclose(base, tft.infer(edited, H)["forecast"].to_numpy())  # future covariates influence the forecast
    off = TFTModule({**FAST_TFT, "use_future_known": False}); off.train(fk)
    assert "fk_0" not in (off.network().hparams["futr_exog_list"] or []) and off._fk_cols == []       # not wired in ...
    assert np.allclose(off.infer(fk, H)["forecast"], off.infer(edited, H)["forecast"])                 # ... so editing them changes nothing


def test_tft_weights_are_retrievable_and_quantiles_are_monotone(tft, fk):
    imp = tft.variable_importance(fk)
    assert imp and all(len(v) > 0 for v in imp.values())
    attn = tft.attention(fk)
    assert attn is not None and len(attn) > 0
    q = tft.infer_quantiles(fk)
    assert list(q.columns) == ["date", "q10", "q50", "q90"] and len(q) == H
    assert (q["q10"] <= q["q50"]).all() and (q["q50"] <= q["q90"]).all()
    assert np.allclose(q["q50"], tft.infer(fk, H)["forecast"])                 # P50 is the point forecast


@pytest.mark.parametrize("h", [1, 4, 8])
def test_tft_horizons_and_the_limit_of_the_covariates(tft, fk, h):
    fc = tft.infer(fk, h)
    assert len(fc) == h and np.isfinite(fc["forecast"]).all()
    last = pd.Timestamp(fk[fk["series_role"] == "endogenous"]["date"].max())
    assert list(fc["date"]) == list(pd.date_range(last + pd.Timedelta(days=1), periods=h, freq="D"))
    with pytest.raises(ValueError, match="future-known"):
        tft.infer(fk, 12)                                                      # 8 covariate values cannot serve 12 periods


def test_tft_save_load_identical(tft, fk, tmp_path):
    tft.save(tmp_path / "v" / "model.pkl")
    l = TFTModule({}); l.load(tmp_path / "v" / "model.pkl")
    assert np.allclose(tft.infer(fk, H)["forecast"], l.infer(fk, H)["forecast"], rtol=1e-4)
    with pytest.raises(ValueError, match="absent"):
        l.infer(fk[fk["dataset"] != "AR_Expected_Unpaid"], H)                  # the covariate series it was trained with is missing


# ------------------------------------------------------------------------------------- TiDE (G-29)
def test_tide_components_and_future_covariate_conditioning(fk):
    m = TiDEModule(dict(FAST_TIDE)); m.train(fk)
    net = m.network()
    for part in ("dense_encoder", "dense_decoder", "temporal_decoder", "global_skip", "futr_exog_projection"):
        assert hasattr(net, part), part                                        # encoder / decoder / temporal decoder / linear skip
    assert type(net.loss).__name__ == "MSE"
    assert net.temporal_decoder.layernorm is False and net.dense_encoder[0].layernorm is True   # only the degenerate final LayerNorm is off
    base = m.infer(fk, H)["forecast"].to_numpy()
    edited = fk.copy()
    mask = (edited["dataset"] == "AR_Expected_Unpaid") & (edited["date"] > edited[edited["series_role"] == "endogenous"]["date"].max())
    edited.loc[mask, "value"] = edited.loc[mask, "value"] * 5 + 100
    assert not np.allclose(base, m.infer(edited, H)["forecast"].to_numpy())    # horizon-step outputs see the future covariates
    off = TiDEModule({**FAST_TIDE, "use_future_known": False}); off.train(fk)
    assert not hasattr(off.network(), "futr_exog_projection")


# ------------------------------------------------------------------------------------- pooled (both directions, all currencies)
def test_pooled_tft_maps_differently_named_covariates_positionally():
    segs = _pool()
    b = assemble_pool(segs)
    m = TFTModule(dict(FAST_TFT))
    assert m.check_pooled_eligibility(b).eligible
    m.train_pooled(b)
    assert m._fk_cols == ["fk_0"] and m._fk_positional                         # AR_Expected_Unpaid and AP_Expected_Unpaid -> one channel
    assert any(type(x).__name__ == "StaticCovariateEncoder" for x in m.network().modules())   # entity/currency/direction
    for sid, df in segs.items():
        fc = m.infer(df, H)
        level = df[df["series_role"] == "endogenous"]["value"].tail(28).mean()
        assert len(fc) == H and 0.5 < fc["forecast"].mean() / level < 1.6, sid
    # a pool whose series disagree on the number of covariates is refused, with the reason
    mixed = dict(segs)
    k = list(mixed)[0]
    mixed[k] = mixed[k][~((mixed[k]["dataset"] == "AR_Expected_Unpaid") & (mixed[k]["date"] > mixed[k][mixed[k]["series_role"] == "endogenous"]["date"].max()))]
    r = TFTModule(dict(FAST_TFT)).check_pooled_eligibility(assemble_pool(mixed))
    assert not r.eligible and "no future-known" in r.reason


def test_pooled_save_load_and_unseen_series():
    segs = _pool()
    m = TiDEModule(dict(FAST_TIDE)); m.train_pooled(assemble_pool(segs))
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        m.save(pathlib.Path(d) / "p" / "model.pkl")
        l = TiDEModule({}); l.load(pathlib.Path(d) / "p" / "model.pkl")
    df = next(iter(segs.values()))
    assert np.allclose(m.infer(df, H)["forecast"], l.infer(df, H)["forecast"], rtol=1e-4)
    new = df.copy(); new["segment_id"] = "9999-INR-inflow"
    assert len(l.infer(new, H)) == H


# ------------------------------------------------------------------------------------- orchestrator paths
def _orch(tmp_path, keep, hp):
    from config import load_config, build_candidates
    from orchestrator import Orchestrator
    from registry import ModelRegistry
    cfg = load_config("config.json")
    cfg["algorithms"] = {k: v for k, v in cfg["algorithms"].items() if k in keep}
    for k in keep:
        cfg["algorithms"][k]["hyperparameters"].update(hp)
    reg = ModelRegistry(tmp_path / "reg")
    return Orchestrator(reg, tmp_path / "logs", build_candidates(cfg), max_parallel_workers=1, holdout_periods=20,
                        window_search_enabled=False, backtest_step=100, hyperparameter_search={"enabled": False},
                        mlflow_config=None), reg


def test_holdout_and_windows_are_not_shifted_by_future_only_dates(tmp_path, plain, fk):
    orch, _ = _orch(tmp_path, ["tide"], {"max_steps": 15})
    sel_p, _ = orch._split_holdout(plain)
    sel_f, _ = orch._split_holdout(fk)
    assert sel_p.loc[sel_p["series_role"] == "endogenous", "date"].max() == sel_f.loc[sel_f["series_role"] == "endogenous", "date"].max()


def test_single_segment_pipeline_runs_tide_with_future_known_and_scores_the_holdout(tmp_path):
    fk = SyntheticAdapter().extract(n_days=1100, future_days=8)      # above the 1000-observation floor of TFT / TiDE
    orch, reg = _orch(tmp_path, ["tide"], {"max_steps": 15, "val_check_steps": 15})
    seg = fk["segment_id"].iloc[0]
    e = orch.full_train(seg, fk, horizon=H, rule_version="r1")
    log = e.elimination_log
    assert not log["eliminated_error"], log                                     # ran, did not error
    if e.algorithm_name == "tide":
        assert e.holdout_metrics and "error" not in e.holdout_metrics           # holdout scored with the held-out covariates
        assert len(orch.daily_infer(seg, fk, H)) == H


def test_pooled_pipeline_runs_tide_with_future_known(tmp_path):
    orch, reg = _orch(tmp_path, ["tide"], {"max_steps": 15, "val_check_steps": 15})
    segs = _pool(n_days=1100)
    entries = orch.full_train_pooled(segs, horizon=H, rule_version="r1")
    assert set(entries) == set(segs)
    for e in entries.values():
        assert not e.elimination_log["eliminated_error"], e.elimination_log
