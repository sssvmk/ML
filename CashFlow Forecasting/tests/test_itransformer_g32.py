"""G-32 iTransformer acceptance: variates are tokens (endogenous + one per exogenous series), attention runs across variates."""
import numpy as np
import pandas as pd
import pytest
from adapters.synthetic import SyntheticAdapter
from algorithms.itransformer import ITransformerModule, TARGET_ID

FAST = {"n_lags": 24, "max_steps": 12, "val_check_steps": 6, "horizon": 4, "hidden_size": 32, "d_ff": 64}


@pytest.fixture(scope="module")
def df():
    return SyntheticAdapter().extract(n_days=900)      # 1 endogenous + 2 exogenous datasets


@pytest.fixture(scope="module")
def m(df):
    x = ITransformerModule(dict(FAST)); x.train(df); return x


def _token_shapes(mod, df):
    seen = []
    layer = next(x for x in mod.network().modules() if type(x).__name__ == "TransEncoderLayer")
    h = layer.register_forward_pre_hook(lambda mm, a, kw: seen.append((a[0] if a else kw["x"]).shape), with_kwargs=True)
    mod.infer(df, 4)
    h.remove()
    return seen


def test_not_a_stand_in_and_inverted_architecture(m):
    net = m.network()
    assert type(net).__name__ == "iTransformer" and "sklearn" not in type(net).__module__
    names = {type(x).__name__ for x in net.modules()}
    assert {"DataEmbedding_inverted", "TransEncoder", "AttentionLayer"} <= names


def test_each_variate_is_one_token_and_exogenous_adds_a_token(m, df):
    assert m.n_variates == 3 and m.network().hparams["n_series"] == 3
    assert _token_shapes(m, df)[0][1] == 3                         # attention runs over 3 variate tokens
    endo_only = df[df["series_role"] == "endogenous"]
    one = ITransformerModule(dict(FAST)); one.train(endo_only)
    assert one.n_variates == 1 and _token_shapes(one, endo_only)[0][1] == 1
    ex_off = ITransformerModule({**FAST, "use_exog": False}); ex_off.train(df)
    assert ex_off.n_variates == 1                                  # exogenous can be switched off explicitly


def test_forecast_is_for_the_target_and_uses_the_exogenous_variates(m, df):
    base = m.infer(df, 4)["forecast"].to_numpy()
    edited = df.copy()
    ex = edited["series_role"] == "exogenous"
    tail = sorted(edited["date"].unique())[-24:]
    idx = edited.index[ex & edited["date"].isin(tail)]
    # change the SHAPE of the exogenous window (a pure rescale is removed by the model's per-variate series normalisation)
    edited.loc[idx, "value"] = np.random.default_rng(1).permutation(edited.loc[idx, "value"].to_numpy())
    assert not np.allclose(base, m.infer(edited, 4)["forecast"].to_numpy())   # other variates influence the target forecast


@pytest.mark.parametrize("h", [1, 4, 30])
def test_horizons_exact_dated_rows(m, df, h):
    fc = m.infer(df, h)
    last = pd.Timestamp(df["date"].max())
    assert len(fc) == h and np.isfinite(fc["forecast"]).all()
    assert list(fc["date"]) == list(pd.date_range(last + pd.Timedelta(days=1), periods=h, freq="D"))


def test_save_load_and_exog_mismatch_is_refused(m, df, tmp_path):
    m.save(tmp_path / "v" / "model.pkl")
    l = ITransformerModule({}); l.load(tmp_path / "v" / "model.pkl")
    assert np.allclose(m.infer(df, 4)["forecast"], l.infer(df, 4)["forecast"], rtol=1e-4)
    with pytest.raises(ValueError, match="absent"):
        l.infer(df[df["series_role"] == "endogenous"], 4)          # a variate seen in training is missing


def test_required_observations_and_diagnostics_are_honest(m):
    assert ITransformerModule({"n_lags": 24, "horizon": 4}).required_observations() == 1000
    d = m.diagnose()
    assert d.ran is False and "residuals" in d.details["reason"]   # the library has no in-sample prediction for multivariate models
