"""G-24 WaveNet acceptance (route: GluonTS PyTorch WaveNet behind the adapter, decision D-4)."""
import numpy as np
import pandas as pd
import pytest
import torch
from adapters.synthetic import SyntheticAdapter
from algorithms.wavenet import WaveNetModule, receptive_field
from algorithms.tcn import TCNModule

FAST = {"horizon": 4, "dilation_depth": 5, "max_epochs": 2, "num_batches_per_epoch": 10, "num_bins": 128}


@pytest.fixture(scope="module")
def df():
    return SyntheticAdapter().extract(n_days=900)


@pytest.fixture(scope="module")
def m(df):
    x = WaveNetModule(dict(FAST)); x.train(df); return x


def _layers(net):
    return [x for x in net.modules() if type(x).__name__ == "CausalDilatedResidualLayer"]


def test_is_the_gluonts_wavenet_and_not_the_tcn_or_a_stand_in(m):
    net = m.network()
    assert type(net).__name__ == "WaveNet" and type(net).__module__.startswith("gluonts.torch.model.wavenet")
    assert not isinstance(m, TCNModule) and not getattr(WaveNetModule, "is_simplified_stand_in", False)
    assert "sklearn" not in type(net).__module__


def test_dilations_cycle_1_2_4_and_receptive_field_matches_the_real_network(m):
    net = m.network()
    assert net.dilations == [1, 2, 4, 8, 16] and len(_layers(net)) == 5
    assert net.receptive_field == m.receptive_field == receptive_field(5, 1) == 32      # the module reports the network's own RF
    two = WaveNetModule({**FAST, "num_stacks": 2, "max_epochs": 1, "num_batches_per_epoch": 2}); two.train(SyntheticAdapter().extract(n_days=600))
    assert two.network().dilations == [1, 2, 4, 8, 16] * 2 and two.network().receptive_field == two.receptive_field == 63


def test_receptive_field_grows_exponentially_with_layers_per_cycle():
    rf = [receptive_field(d, 1) for d in range(1, 8)]
    assert rf == [2, 4, 8, 16, 32, 64, 128]                                             # doubles with every added dilation layer
    assert receptive_field(6, 3) == 1 + 3 * 63


def test_gated_activation_tanh_times_sigmoid_with_residual_and_skip_convs(m):
    layer = _layers(m.network())[0]
    x = torch.randn(2, next(p for p in layer.conv_tanh.parameters()).shape[1], 40)
    shift = (layer.kernel_size - 1) * layer.dilation
    with torch.no_grad():
        skip, out = layer(x)                                                              # forward returns (skip, residual-out)
        gated = layer.conv_sigmoid(x) * layer.conv_tanh(x)                                 # sigmoid(conv) * tanh(conv)
        assert torch.allclose(skip, layer.conv_skip(gated), atol=1e-5)                    # skip = 1x1 conv of the gated activation
        assert torch.allclose(out, layer.conv_residual(gated) + x[..., shift:], atol=1e-5)  # residual = 1x1 conv + input (time-aligned)
    assert isinstance(layer.conv_sigmoid[-1], torch.nn.Sigmoid) and isinstance(layer.conv_tanh[-1], torch.nn.Tanh)


def test_causality_future_input_does_not_change_earlier_outputs(m):
    layer = _layers(m.network())[3]                                                      # dilation 8
    shift = (layer.kernel_size - 1) * layer.dilation                                     # output j is aligned with input time j + shift
    c = next(p for p in layer.conv_tanh.parameters()).shape[1]
    x = torch.randn(1, c, 60); y = x.clone(); y[:, :, 41:] += 5.0                        # change everything after t = 40
    with torch.no_grad():
        (sa, ra), (sb, rb) = layer(x), layer(y)
    keep = 40 - shift + 1                                                                # outputs whose aligned time is <= 40
    assert keep > 0
    assert torch.allclose(ra[:, :, :keep], rb[:, :, :keep], atol=1e-6) and torch.allclose(sa[:, :, :keep], sb[:, :, :keep], atol=1e-6)
    assert not torch.allclose(ra[:, :, keep:], rb[:, :, keep:])                          # ... but later outputs do change


def test_eligibility_and_required_observations_use_the_real_receptive_field(df):
    w = WaveNetModule({"horizon": 4, "dilation_depth": 6, "num_stacks": 1})
    assert w.receptive_field == 64 and w.has_eligibility_condition
    assert w.required_observations() == 1000                                              # max(1000, 64 + 4 + 500)
    assert WaveNetModule({"horizon": 4, "dilation_depth": 9, "num_stacks": 3}).required_observations() == 1 + 3 * 511 + 4 + 500
    assert w.check_eligibility(df).eligible
    tiny = df[df["date"] < df["date"].min() + np.timedelta64(50, "D")]
    assert not w.check_eligibility(tiny).eligible


def test_architecture_hyperparameters_change_the_network(df):
    small = WaveNetModule({**FAST, "num_residual_channels": 8, "max_epochs": 1, "num_batches_per_epoch": 2}); small.train(df)
    big = WaveNetModule({**FAST, "num_residual_channels": 32, "max_epochs": 1, "num_batches_per_epoch": 2}); big.train(df)
    assert big.n_parameters() > small.n_parameters()


@pytest.mark.parametrize("h", [1, 4, 30])
def test_horizons_exact_dated_rows(m, df, h):
    fc = m.infer(df, h)
    assert len(fc) == h and np.isfinite(fc["forecast"]).all()
    assert list(fc["date"]) == list(pd.date_range(pd.Timestamp(df["date"].max()) + pd.Timedelta(days=1), periods=h, freq="D"))


def test_sample_based_quantiles_are_ordered(m, df):
    q = m.infer_quantiles(df)
    assert len(q) == 4 and (q["q10"] <= q["q50"]).all() and (q["q50"] <= q["q90"]).all()
    assert not np.allclose(q["q10"], q["q90"])                                            # a real distribution, not a point


def test_same_seed_reproducible_and_save_load_identical(df, tmp_path):
    a, b = WaveNetModule(dict(FAST)), WaveNetModule(dict(FAST))
    a.train(df); b.train(df)
    assert np.allclose(a.infer(df, 4)["forecast"], b.infer(df, 4)["forecast"])
    a.save(tmp_path / "v" / "model.pkl")
    l = WaveNetModule({}); l.load(tmp_path / "v" / "model.pkl")                           # no training data
    assert l.receptive_field == a.receptive_field and np.allclose(a.infer(df, 4)["forecast"], l.infer(df, 4)["forecast"])


def test_early_stopping_uses_a_chronological_validation_tail_and_leaves_no_files(df):
    import os
    w = WaveNetModule({**FAST, "max_epochs": 1, "num_batches_per_epoch": 2}); w.train(df)
    es = w._fitted_model["early_stopping"]
    assert es["used"] and es["validation_rows"] >= 8 and es["restored_best_checkpoint"]
    assert not any(d in os.listdir(".") for d in ("lightning_logs", "checkpoints"))      # trainer output goes to a temp dir


def test_gaps_are_not_silently_filled(df):
    gappy = df[df["date"] != sorted(df["date"].unique())[100]]
    with pytest.raises(ValueError, match="gap_policy"):
        WaveNetModule({**FAST, "max_epochs": 1, "num_batches_per_epoch": 1}).train(gappy)
