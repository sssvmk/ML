"""Frequency is config-driven, not hard-coded (this request). config.json -> orchestration.frequency (default "D",
unchanged behaviour) flows into every module as hyperparameters["frequency"] / self.frequency, and every date-generating
call site in the codebase (future_dates, next_period_after, reindexing, the vendored NeuralForecast/GluonTS freq
arguments) honours it instead of assuming daily."""
import pandas as pd
import pytest
from adapters.synthetic import SyntheticAdapter
from algorithms.utils import future_dates, next_period_after
from algorithms.arimax import ARIMAXModule
from algorithms.tcn import TCNModule
from algorithms.rnn import RNNModule
from algorithms.wavenet import WaveNetModule
from algorithms.etsformer import ETSformerModule


def test_future_dates_and_next_period_after_generalise_beyond_daily():
    last = pd.Timestamp("2024-01-01")
    assert list(future_dates(last, 3)) == list(pd.date_range("2024-01-02", periods=3, freq="D"))         # default unchanged
    assert list(future_dates(last, 3, freq="W")) == list(pd.date_range("2024-01-07", periods=3, freq="W"))
    assert list(future_dates(last, 2, freq="ME")) == list(pd.date_range("2024-01-31", periods=2, freq="ME"))
    assert next_period_after(last, "D") == pd.Timestamp("2024-01-02")
    assert next_period_after(last, "W") == pd.Timestamp("2024-01-07")


def test_synthetic_adapter_frequency_is_configurable():
    daily = SyntheticAdapter().extract(n_days=60, freq="D")
    weekly = SyntheticAdapter().extract(n_days=60, freq="W")
    dd = sorted(daily["date"].unique())
    wd = sorted(weekly["date"].unique())
    assert (pd.Timestamp(dd[1]) - pd.Timestamp(dd[0])).days == 1
    assert (pd.Timestamp(wd[1]) - pd.Timestamp(wd[0])).days == 7


def test_default_frequency_is_daily_everywhere_unless_configured():
    assert ARIMAXModule({}).frequency == "D" and TCNModule({}).frequency == "D"


@pytest.mark.parametrize("cls,hp", [
    (ARIMAXModule, {}),                                                    # statsmodels family, via central future_dates
    (TCNModule, {"n_lags": 10, "max_epochs": 3}),                          # native torch family, via torch_base.py
    (RNNModule, {"n_lags": 10, "max_epochs": 1, "input_size": 10}),        # NeuralForecast adapter family, via nf_adapter.py
])
def test_weekly_frequency_end_to_end(cls, hp):
    df = SyntheticAdapter().extract(n_days=150, freq="W")
    m = cls({**hp, "frequency": "W", "horizon": 4})
    m.train(df)
    fc = m.infer(df, 4)
    last = pd.Timestamp(sorted(df["date"].unique())[-1])
    assert list(fc["date"]) == list(pd.date_range(last, periods=5, freq="W")[1:])   # weekly-spaced, not daily


def test_wavenet_and_etsformer_honour_configured_frequency():
    df = SyntheticAdapter().extract(n_days=200, freq="W")
    w = WaveNetModule({"frequency": "W", "horizon": 4, "max_epochs": 1, "dilation_depth": 2, "num_stacks": 1})
    w.train(df)
    fc = w.infer(df, 4)
    last = pd.Timestamp(sorted(df["date"].unique())[-1])
    assert list(fc["date"]) == list(pd.date_range(last, periods=5, freq="W")[1:])

    e = ETSformerModule({"frequency": "W", "horizon": 4, "n_lags": 12, "max_epochs": 1})
    e.train(df)
    dec = e.decompose(df)
    assert list(dec["date"]) == list(pd.date_range(last, periods=5, freq="W")[1:])
