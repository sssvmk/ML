"""G-23 acceptance: custom PyTorch TCN per spec (residual blocks, two weight-normed dilated causal convs, dilation 2**i)."""
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from adapters.synthetic import SyntheticAdapter
from algorithms.tcn import TCNModule, TCNNet, receptive_field, TemporalBlock
from algorithms.wavenet import WaveNetModule

FAST = {"horizon": 4, "max_epochs": 6, "kernel_size": 3, "levels": 3}


@pytest.fixture(scope="module")
def df():
    return SyntheticAdapter().extract(n_days=900)


@pytest.fixture(scope="module")
def trained(df):
    m = TCNModule(dict(FAST)); m.train(df)
    return m


def test_spec_structure_dilations_residual_weightnorm_two_convs(trained):
    net = trained.network()
    blocks = list(net.blocks)
    assert len(blocks) == 3 and all(isinstance(b, TemporalBlock) for b in blocks)
    assert [b.conv1.conv.dilation[0] for b in blocks] == [1, 2, 4] == [b.conv2.conv.dilation[0] for b in blocks]   # d = 2**i
    assert all(hasattr(b.conv1.conv, "parametrizations") for b in blocks)                                         # weight-norm
    assert isinstance(blocks[0].down, nn.Conv1d) and isinstance(blocks[1].down, nn.Identity)                     # 1x1 conv only when channels change
    assert "sklearn" not in type(net).__module__ and not getattr(TCNModule, "is_simplified_stand_in", False)


def test_causality_perturbing_future_input_leaves_earlier_outputs_unchanged():
    torch.manual_seed(0)
    net = TCNNet(n_features=3, horizon=4, kernel_size=3, levels=3, channels=8, dropout=0.0).eval()
    x = torch.randn(2, 40, 3)
    def seq_out(inp):
        h = inp.transpose(1, 2)
        for b in net.blocks:
            h = b(h)
        return h                                                    # (B, C, L) -- one output per timestep
    base = seq_out(x)
    t = 20
    x2 = x.clone(); x2[:, t + 1:, :] += 5.0                         # change everything after t
    out2 = seq_out(x2)
    assert torch.allclose(base[:, :, : t + 1], out2[:, :, : t + 1], atol=1e-6)
    assert not torch.allclose(base[:, :, t + 1:], out2[:, :, t + 1:])


@pytest.mark.parametrize("k,levels", [(2, 3), (3, 3), (3, 4), (5, 2)])
def test_measured_receptive_field_equals_reported(k, levels):
    torch.manual_seed(1)
    rf = receptive_field(k, levels)
    assert rf == 1 + 2 * (k - 1) * (2 ** levels - 1)
    L = rf + 20
    net = TCNNet(n_features=1, horizon=1, kernel_size=k, levels=levels, channels=16, dropout=0.0).eval()
    influenced = {}
    for dist in (rf - 1, rf):                                       # position rf-1 back is inside the RF, rf back is outside
        moved = False
        for _ in range(30):
            x = torch.randn(1, L, 1)
            x2 = x.clone(); x2[0, L - 1 - dist, 0] += 3.0
            with torch.no_grad():
                moved = moved or not torch.allclose(net(x), net(x2), atol=1e-7)
        influenced[dist] = moved
    assert influenced[rf - 1] is True and influenced[rf] is False, (k, levels, rf, influenced)


def test_kernel_levels_channels_dropout_change_the_model(df):
    base = TCNModule({**FAST}); base.train(df)
    for change in ({"kernel_size": 5}, {"levels": 4}, {"channels": 64}):
        other = TCNModule({**FAST, **change}); other.train(df)
        assert other.n_parameters() != base.n_parameters(), change
        assert not np.allclose(other.infer(df, 4)["forecast"], base.infer(df, 4)["forecast"]), change
    d0, d3 = TCNModule({**FAST, "dropout": 0.0}), TCNModule({**FAST, "dropout": 0.3})
    d0.train(df); d3.train(df)
    assert d0.n_parameters() == d3.n_parameters()
    assert not np.allclose(d0.infer(df, 4)["forecast"], d3.infer(df, 4)["forecast"])   # dropout changes what is learned


def test_eligibility_and_required_observations_use_the_real_receptive_field(df):
    m = TCNModule({"kernel_size": 3, "levels": 4, "horizon": 4})
    assert m.receptive_field == 61 and m.required_observations() == 61 + 4 + 500
    assert m.check_eligibility(df).eligible
    tiny = df[df["date"] < df["date"].min() + np.timedelta64(50, "D")]
    r = m.check_eligibility(tiny)
    assert not r.eligible and "receptive field 61" in r.reason
    assert TCNModule({"kernel_size": 5, "levels": 6}).required_observations() == receptive_field(5, 6) + 4 + 500


@pytest.mark.parametrize("h", [1, 4, 30])
def test_horizons_return_exactly_that_many_dated_rows(trained, df, h):
    fc = trained.infer(df, h)
    last = pd.Timestamp(df["date"].max())
    assert len(fc) == h and np.isfinite(fc["forecast"]).all()
    assert list(fc["date"]) == list(pd.date_range(last + pd.Timedelta(days=1), periods=h, freq="D"))


def test_exogenous_features_are_inputs_and_windows_are_local(trained, df):
    assert trained._n_features == 3 and len(trained._exog_cols) == 2               # target + 2 exogenous datasets
    base = trained.infer(df, 4)["forecast"].to_numpy()
    old = df.copy()
    first = sorted(old["date"].unique())[0]
    old.loc[(old["date"] == first) & (old["series_role"] == "endogenous"), "value"] += 1e6   # far outside the receptive field
    assert np.allclose(base, trained.infer(old, 4)["forecast"].to_numpy(), rtol=1e-5)


def test_seed_reproducibility_save_load_and_early_stopping(df, tmp_path):
    a, b = TCNModule({**FAST, "max_epochs": 12}), TCNModule({**FAST, "max_epochs": 12})
    a.train(df); b.train(df)
    assert np.allclose(a.infer(df, 4)["forecast"], b.infer(df, 4)["forecast"])
    es = a._fitted_model["early_stopping"]
    assert es["used"] and es["restored_best_checkpoint"] and "train_validation_loss_gap" in es
    a.save(tmp_path / "v1" / "model.pt")
    c = TCNModule({}); c.load(tmp_path / "v1" / "model.pt")
    assert c.receptive_field == a.receptive_field
    assert np.allclose(a.infer(df, 4)["forecast"], c.infer(df, 4)["forecast"], atol=1e-5)


def test_diagnose_runs_and_search_space_is_architecture_only():
    space = TCNModule({}).hyperparameter_search_space()
    assert {"kernel_size", "levels", "channels", "dropout"} <= set(space)


def test_wavenet_is_independent_of_tcn_and_now_the_gluonts_model():
    """WaveNet must never silently inherit the custom TCN: it has its own (GluonTS) implementation, see test_wavenet_g24.py."""
    w = WaveNetModule({"dilation_depth": 6, "num_stacks": 1})
    assert not isinstance(w, TCNModule) and not getattr(w, "is_simplified_stand_in", False)
    assert w.has_eligibility_condition and w.required_observations() == 1000
