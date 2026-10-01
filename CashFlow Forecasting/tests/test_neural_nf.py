"""G-41 adapter + G-20/G-21/G-22 (RNN, LSTM, GRU) acceptance checks (matrix C-1..C-6 and per-gap checks)."""
import sys
import numpy as np
import pandas as pd
import pytest
import torch.nn as nn
from adapters.synthetic import SyntheticAdapter
from algorithms.rnn import RNNModule
from algorithms.lstm import LSTMModule
from algorithms.gru import GRUModule

FAST = {"n_lags": 14, "max_steps": 60, "horizon": 4, "val_check_steps": 20}


@pytest.fixture(scope="module")
def df():
    return SyntheticAdapter().extract(n_days=900)


@pytest.fixture(scope="module")
def trained(df):
    out = {}
    for cls in (RNNModule, LSTMModule, GRUModule):
        m = cls(dict(FAST))
        m.train(df)
        out[cls.__name__] = m
    return out


def test_s2_vendored_neuralforecast_imports_without_ray():
    from algorithms._vendor import ensure_vendored_neuralforecast, VENDOR_ROOT
    nfm = ensure_vendored_neuralforecast()
    assert VENDOR_ROOT.resolve() in __import__("pathlib").Path(nfm.__file__).resolve().parents
    if "ray" in sys.modules:
        pytest.skip("ray is installed here; the no-ray path cannot be shown in this environment")
    import importlib.util
    assert importlib.util.find_spec("ray") is None   # and the import above still succeeded


@pytest.mark.parametrize("cls,torch_cls", [(RNNModule, nn.RNN), (LSTMModule, nn.LSTM), (GRUModule, nn.GRU)])
def test_real_recurrent_architecture_not_sklearn(trained, cls, torch_cls):
    m = trained[cls.__name__]
    net = m.network()
    assert any(type(x) is torch_cls for x in net.modules()), f"no {torch_cls.__name__} inside the network"
    assert "sklearn" not in type(net).__module__ and not hasattr(m, "_model")


def test_architecture_hyperparameters_change_the_network(df):
    small = RNNModule({**FAST, "max_steps": 5, "encoder_hidden_size": 16}); small.train(df)
    big = RNNModule({**FAST, "max_steps": 5, "encoder_hidden_size": 64}); big.train(df)
    assert big.n_parameters() > small.n_parameters()
    deep = LSTMModule({**FAST, "max_steps": 5, "encoder_n_layers": 3}); deep.train(df)
    shallow = LSTMModule({**FAST, "max_steps": 5, "encoder_n_layers": 1}); shallow.train(df)
    assert deep.n_parameters() > shallow.n_parameters()


def test_gru_has_three_quarters_of_the_lstm_recurrent_parameters(df):
    cfg = {**FAST, "max_steps": 5, "encoder_hidden_size": 32, "encoder_n_layers": 2}
    l, g = LSTMModule(cfg), GRUModule(cfg); l.train(df); g.train(df)
    count = lambda m, t: sum(p.numel() for x in m.network().modules() if type(x) is t for p in x.parameters())
    assert count(g, nn.GRU) * 4 == count(l, nn.LSTM) * 3     # 3 gates' worth vs 4 gates' worth


@pytest.mark.parametrize("name", ["RNNModule", "LSTMModule", "GRUModule"])
def test_horizons_1_4_30_return_exactly_that_many_dated_rows(trained, df, name):
    m = trained[name]
    last = pd.Timestamp(df["date"].max())
    for h in (1, 4, 30):
        fc = m.infer(df, h)
        assert len(fc) == h and np.isfinite(fc["forecast"]).all()
        assert list(fc["date"]) == list(pd.date_range(last + pd.Timedelta(days=1), periods=h, freq="D"))


def test_forecast_depends_on_the_ordered_recent_sequence(trained, df):
    m = trained["RNNModule"]
    base = m.infer(df, 4)["forecast"].to_numpy()
    shuffled = df.copy()
    endo = shuffled["series_role"] == "endogenous"
    tail_dates = sorted(shuffled.loc[endo, "date"].unique())[-14:]
    idx = shuffled.index[endo & shuffled["date"].isin(tail_dates)]
    shuffled.loc[idx, "value"] = np.random.default_rng(0).permutation(shuffled.loc[idx, "value"].to_numpy())
    assert not np.allclose(base, m.infer(shuffled, 4)["forecast"].to_numpy())      # order matters


def test_no_leakage_values_older_than_the_window_do_not_matter(trained, df):
    m = trained["LSTMModule"]
    base = m.infer(df, 4)["forecast"].to_numpy()
    edited = df.copy()
    first = sorted(edited["date"].unique())[0]
    edited.loc[(edited["date"] == first) & (edited["series_role"] == "endogenous"), "value"] += 1e6   # far outside the lookback
    assert np.allclose(base, m.infer(edited, 4)["forecast"].to_numpy(), rtol=1e-5)


def test_same_seed_reproducible_and_save_load_identical(df, tmp_path):
    a, b = GRUModule({**FAST, "max_steps": 30}), GRUModule({**FAST, "max_steps": 30})
    a.train(df); b.train(df)
    assert np.allclose(a.infer(df, 4)["forecast"], b.infer(df, 4)["forecast"])
    a.save(tmp_path / "v1" / "model.pkl")
    c = GRUModule({}); c.load(tmp_path / "v1" / "model.pkl")
    assert np.allclose(a.infer(df, 4)["forecast"], c.infer(df, 4)["forecast"])     # loaded without any training data


def test_early_stopping_restores_best_checkpoint_and_clipping_is_configured(df):
    m = RNNModule({**FAST, "max_steps": 300, "early_stop_patience_steps": 2})
    m.train(df)
    es = m._fitted_model["early_stopping"]
    assert es["used"] and es["restored_best_checkpoint"] and es["stopped_step"] < 300
    assert es["best_step"] <= es["stopped_step"]
    assert m.network().trainer_kwargs.get("gradient_clip_val") == 1.0               # recurrent family: BPTT clipping


def test_required_observations_and_eligibility_match_prd_rows():
    for cls in (RNNModule, LSTMModule, GRUModule):
        m = cls({"n_lags": 20, "horizon": 4})
        assert m.required_observations() == 20 + 4 + 500        # N >= lookback + horizon + 500 (no extra floor)
        assert not cls.has_eligibility_condition


def test_search_space_parameters_all_change_the_built_network(df):
    space = RNNModule({}).hyperparameter_search_space()
    assert {"n_lags", "encoder_hidden_size", "encoder_n_layers", "encoder_dropout", "learning_rate"} <= set(space)
    net = RNNModule({**FAST, "encoder_hidden_size": 16, "encoder_n_layers": 2, "encoder_dropout": 0.2})._build_model([], False, [])
    assert net.hparams["encoder_hidden_size"] == 16 and net.hparams["encoder_n_layers"] == 2 and net.hparams["encoder_dropout"] == 0.2
    one = RNNModule({**FAST, "encoder_n_layers": 1, "encoder_dropout": 0.2})._build_model([], False, [])
    assert one.hparams["encoder_dropout"] == 0.0            # inter-layer dropout is inert with one layer, so it is not applied


def test_calendar_gaps_are_not_silently_filled(df):
    gappy = df[~((df["date"] == df["date"].sort_values().unique()[100]))]
    with pytest.raises(ValueError, match="gap_policy"):
        RNNModule({**FAST, "max_steps": 2}).train(gappy)
    RNNModule({**FAST, "max_steps": 2, "gap_policy": "ffill"}).train(gappy)        # explicit choice works


def test_smoke_backtest_rank_registry_daily_infer(tmp_path):
    from registry import ModelRegistry
    from orchestrator import Orchestrator
    from config import load_config, build_candidates
    cfg = load_config("config.json")
    cfg["algorithms"] = {k: v for k, v in cfg["algorithms"].items() if k == "rnn"}
    cfg["algorithms"]["rnn"]["hyperparameters"].update({"max_steps": 40, "val_check_steps": 20})
    reg = ModelRegistry(tmp_path / "reg")
    orch = Orchestrator(reg, tmp_path / "logs", build_candidates(cfg), max_parallel_workers=1, holdout_periods=20,
                        window_search_enabled=False, backtest_step=150, hyperparameter_search={"enabled": False}, mlflow_config=None)
    df = SyntheticAdapter().extract(n_days=900)
    seg = df["segment_id"].iloc[0]
    entry = orch.full_train(seg, df, horizon=4, rule_version="r1")
    cand = entry.elimination_log
    assert "rnn" in (cand["eliminated_mase"] + cand["eliminated_bias"] + cand["eliminated_error"] + [n for n, _ in cand["ranked"]]), cand
    assert not cand["eliminated_error"], "the RNN candidate must run (not error) through the full pipeline"
    fc = orch.daily_infer(seg, df, 4) if entry.algorithm_name == "rnn" else None
    if fc is not None:
        assert len(fc) == 4
