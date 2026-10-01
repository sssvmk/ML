"""Architecture acceptance checks for the NeuralForecast-backed models N-BEATS, N-HiTS, TimesNet, Informer,
Autoformer, FEDformer (G-31, G-33..G-38). Each asserts something specific to the published architecture, not just
that a forecast comes out."""
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from adapters.synthetic import SyntheticAdapter
from algorithms.nbeats import NBEATSModule
from algorithms.nhits import NHiTSModule
from algorithms.timesnet import TimesNetModule
from algorithms.informer import InformerModule
from algorithms.autoformer import AutoformerModule
from algorithms.fedformer import FEDformerModule

FAST = {"max_steps": 12, "val_check_steps": 6, "horizon": 4}


@pytest.fixture(scope="module")
def df():
    return SyntheticAdapter().extract(n_days=900)


def _count(m, cls_name):
    return sum(1 for x in m.network().modules() if type(x).__name__ == cls_name)


# ---------------------------------------------------------------- shared contract checks
CASES = [
    (NBEATSModule, {"n_lags": 12}), (NHiTSModule, {"n_lags": 14}),
    (TimesNetModule, {"n_lags": 24, "hidden_size": 16, "conv_hidden_size": 16}),
    (InformerModule, {"n_lags": 24, "hidden_size": 32}), (AutoformerModule, {"n_lags": 24, "hidden_size": 32}),
    (FEDformerModule, {"n_lags": 24, "hidden_size": 32}),
]


@pytest.mark.parametrize("cls,hp", CASES, ids=[c.__name__ for c, _ in CASES])
def test_contract_horizons_saveload_and_not_a_stand_in(cls, hp, df, tmp_path):
    m = cls({**FAST, **hp}); m.train(df)
    assert "sklearn" not in type(m.network()).__module__ and not getattr(cls, "is_simplified_stand_in", False)
    last = pd.Timestamp(df["date"].max())
    for h in (1, 4, 30):
        fc = m.infer(df, h)
        assert len(fc) == h and np.isfinite(fc["forecast"]).all()
        assert list(fc["date"]) == list(pd.date_range(last + pd.Timedelta(days=1), periods=h, freq="D"))
    m.save(tmp_path / "m" / "model.pkl")
    l = cls({}); l.load(tmp_path / "m" / "model.pkl")
    assert np.allclose(m.infer(df, 4)["forecast"], l.infer(df, 4)["forecast"], rtol=1e-4)


@pytest.mark.parametrize("cls,expected", [(NBEATSModule, 12 + 4 + 500), (NHiTSModule, 14 + 4 + 500)])
def test_lookback_plus_horizon_plus_500_family(cls, expected):
    m = cls({"n_lags": expected - 504, "horizon": 4})
    assert m.required_observations() == expected and m.context_family_floor == 0


@pytest.mark.parametrize("cls", [TimesNetModule, InformerModule, AutoformerModule, FEDformerModule])
def test_context_family_has_the_1000_floor(cls):
    assert cls({"n_lags": 24, "horizon": 4}).required_observations() == 1000           # max(1000, 24 + 4 + 500)
    assert cls({"n_lags": 600, "horizon": 4}).required_observations() == 1104          # formula wins above the floor


# ---------------------------------------------------------------- G-37 N-BEATS
def test_g37_nbeats_double_residual_stacking_and_forecast_is_sum_of_blocks(df):
    m = NBEATSModule({**FAST, "n_lags": 12, "scaler_type": "identity", "stack_types": ["identity", "trend", "seasonality"]})
    m.train(df)
    net = m.network()
    blocks = list(net.blocks)
    ins, outs = [], []
    hooks = [b.register_forward_pre_hook(lambda mod, a, kw: ins.append(kw["insample_y"].detach().clone()), with_kwargs=True) for b in blocks]
    hooks += [b.register_forward_hook(lambda mod, a, o: outs.append((o[0].detach().clone(), o[1].detach().clone()))) for b in blocks]
    fc = m.infer(df, 4)
    for h in hooks:
        h.remove()
    assert len(ins) == len(blocks) == len(outs)
    for l in range(len(blocks) - 1):                                      # x_{l+1} = x_l - backcast_l
        assert torch.allclose(ins[l + 1], ins[l] - outs[l][0], atol=1e-4)
    level = float(df[df.series_role == "endogenous"].sort_values("date")["value"].iloc[-1])
    total = level + sum(o[1].squeeze(-1)[0] for o in outs).numpy()          # naive-1 level + sum of block forecasts
    assert np.allclose(fc["forecast"].to_numpy(), total, rtol=1e-4)
    kinds = {type(b.basis).__name__ for b in blocks}
    assert {"IdentityBasis", "TrendBasis", "SeasonalityBasis"} <= kinds     # interpretable stacks are separable


def test_g37_nbeats_direct_multi_horizon_and_lookback_multiple(df):
    space = NBEATSModule({"horizon": 4}).hyperparameter_search_space()
    assert (space["n_lags"]["low"], space["n_lags"]["high"]) == (8, 28)     # lookback = k*H, k in 2..7
    a = NBEATSModule({**FAST, "n_lags": 12, "mlp_width": 32, "mlp_layers": 2}); a.train(df)
    b = NBEATSModule({**FAST, "n_lags": 12, "mlp_width": 128, "mlp_layers": 4}); b.train(df)
    assert b.n_parameters() > a.n_parameters()
    assert NBEATSModule({"stack_types": ["trend"], "horizon": 4})._model_kwargs()["n_blocks"] == [1]


# ---------------------------------------------------------------- G-38 N-HiTS
def test_g38_nhits_multirate_pooling_and_interpolated_output(df):
    m = NHiTSModule({**FAST, "n_lags": 16, "n_pool_kernel_size": [4, 2, 1], "n_freq_downsample": [4, 2, 1]}); m.train(df)
    blocks = list(m.network().blocks)
    assert [b.pooling_layer.kernel_size for b in blocks] == [4, 2, 1]        # different input resolution per stack
    same = NHiTSModule({**FAST, "n_lags": 16, "n_pool_kernel_size": [1, 1, 1], "n_freq_downsample": [1, 1, 1]}); same.train(df)
    assert same.n_parameters() != m.n_parameters()                           # the hierarchy changes the network
    seen = []
    hs = [b.basis.register_forward_hook(lambda mod, a, o: seen.append((a[0].shape[-1], o[1].shape[1]))) for b in blocks]
    m.infer(df, 4)
    for h in hs:
        h.remove()
    theta_sizes = [t for t, _ in seen]
    assert all(out_len == 4 for _, out_len in seen)                          # every block emits H outputs ...
    assert theta_sizes[0] < theta_sizes[-1] or len(set(theta_sizes)) > 1     # ... from a shorter theta at coarse stacks


# ---------------------------------------------------------------- G-31 TimesNet
def test_g31_timesnet_2d_inception_and_data_driven_periods(df):
    from algorithms._vendor import ensure_vendored_neuralforecast
    ensure_vendored_neuralforecast()
    from neuralforecast.models.timesnet import FFT_for_Period
    m = TimesNetModule({**FAST, "n_lags": 24, "hidden_size": 16, "conv_hidden_size": 16, "encoder_layers": 2, "top_k": 2}); m.train(df)
    assert _count(m, "TimesBlock") == 2 and any(isinstance(x, nn.Conv2d) for x in m.network().modules())
    t = np.arange(96)
    p = lambda per: FFT_for_Period(torch.tensor(np.sin(2 * np.pi * t / per)[None, :, None], dtype=torch.float32), k=1)[0][0]
    assert p(8) != p(24)                                                     # periods follow the signal's seasonality
    bigger = TimesNetModule({**FAST, "n_lags": 24, "hidden_size": 16, "conv_hidden_size": 16, "encoder_layers": 3, "top_k": 2}); bigger.train(df)
    assert bigger.n_parameters() > m.n_parameters()


# ---------------------------------------------------------------- G-33 Informer
def test_g33_informer_probsparse_and_distilling(df):
    m = InformerModule({**FAST, "n_lags": 24, "hidden_size": 32, "encoder_layers": 3, "distil": True}); m.train(df)
    assert _count(m, "ProbAttention") >= 1
    assert _count(m, "ConvLayer") == 2                                       # distilling between 3 encoder layers
    nd = InformerModule({**FAST, "n_lags": 24, "hidden_size": 32, "encoder_layers": 3, "distil": False}); nd.train(df)
    assert _count(nd, "ConvLayer") == 0 and nd.n_parameters() < m.n_parameters()
    assert m.infer(df, 4).shape[0] == 4                                      # single-pass multi-step decode


# ---------------------------------------------------------------- G-34 Autoformer
def test_g34_autoformer_decomposition_and_autocorrelation(df):
    m = AutoformerModule({**FAST, "n_lags": 24, "hidden_size": 32, "encoder_layers": 2, "decoder_layers": 1}); m.train(df)
    assert _count(m, "AutoCorrelation") >= 3 and _count(m, "SeriesDecomp") >= 4     # decomposition inside every layer
    assert _count(m, "TransEncoderLayer") == 0                               # no vanilla self-attention encoder
    deeper = AutoformerModule({**FAST, "n_lags": 24, "hidden_size": 32, "encoder_layers": 3, "decoder_layers": 1}); deeper.train(df)
    assert _count(deeper, "SeriesDecomp") > _count(m, "SeriesDecomp")
    from algorithms._vendor import ensure_vendored_neuralforecast
    ensure_vendored_neuralforecast()
    import inspect
    from neuralforecast.models import autoformer
    assert "fft" in inspect.getsource(autoformer.AutoCorrelation).lower()    # FFT-based lag correlation


# ---------------------------------------------------------------- G-35 FEDformer
def test_g35_fedformer_complex_weights_on_selected_modes_only(df):
    m = FEDformerModule({**FAST, "n_lags": 24, "hidden_size": 32, "modes": 6, "mode_select": "low"}); m.train(df)
    fbs = [x for x in m.network().modules() if type(x).__name__ == "FourierBlock"]
    assert fbs and all(any(p.is_complex() for p in fb.parameters()) for fb in fbs)
    assert all(len(fb.index) <= 6 for fb in fbs)                             # only M modes carry weights
    few = FEDformerModule({**FAST, "n_lags": 24, "hidden_size": 32, "modes": 2, "mode_select": "low"}); few.train(df)
    assert few.n_parameters() < m.n_parameters()
    # modes are clipped to what the sequence can carry instead of failing
    assert FEDformerModule({**FAST, "n_lags": 24, "hidden_size": 32, "modes": 64})._model_kwargs()["modes"] == 12
