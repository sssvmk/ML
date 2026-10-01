"""G-25 DeepAR acceptance: global AR network, distribution output + NLL loss, sample/quantile forecasts, >= 2 series in one
fit, eligibility derived from the pool actually supplied."""
import sys
import numpy as np
import pandas as pd
import pytest
import torch.nn as nn
sys.path.insert(0, "tests")
from test_g03_pooling import _segments
from pooling import assemble_pool
from algorithms.deepar import DeepARModule

FAST = {"n_lags": 28, "max_steps": 40, "val_check_steps": 20, "horizon": 4}


@pytest.fixture(scope="module")
def segs():
    return _segments()


@pytest.fixture(scope="module")
def m(segs):
    x = DeepARModule(dict(FAST)); x.train_pooled(assemble_pool(segs)); return x


def test_eligibility_is_derived_from_the_pool_supplied(segs):
    b = assemble_pool(segs)
    assert DeepARModule({}).check_pooled_eligibility(b).eligible
    one = assemble_pool({k: segs[k] for k in list(segs)[:1]})
    assert not DeepARModule({}).check_pooled_eligibility(one).eligible                     # one series is not a pool
    small = assemble_pool({k: v[v["date"] < v["date"].min() + np.timedelta64(100, "D")] for k, v in list(segs.items())[:2]})
    assert "1000" in DeepARModule({}).check_pooled_eligibility(small).reason               # pooled N >= 1,000, from the data
    assert not DeepARModule({}).check_eligibility(next(iter(segs.values()))).eligible      # single-segment path
    assert not hasattr(DeepARModule({}), "pooled_series_count")                            # the manual flag is gone
    with pytest.raises(RuntimeError, match="pooled"):
        DeepARModule({}).train(next(iter(segs.values())))


def test_autoregressive_lstm_with_distribution_output_and_nll_loss(m):
    net = m.network()
    assert type(net).__name__ == "DeepAR" and any(type(x) is nn.LSTM for x in net.modules())
    assert type(net.loss).__name__ == "DistributionLoss" and net.loss.distribution == "StudentT"   # NLL objective, heavy-tailed
    assert net.hparams["stat_exog_list"] and all("=" in c for c in net.hparams["stat_exog_list"])  # entity/currency/direction conditioning
    normal = DeepARModule({**FAST, "distribution": "Normal"})._build_model([], False, [])
    assert normal.loss.distribution == "Normal"


def test_one_fit_serves_every_series_at_its_own_scale_without_fx(m, segs):
    assert m._fitted_model["pooled"]["n_series"] == 4
    for sid, df in segs.items():
        fc = m.infer(df, 4)
        level = df[df["series_role"] == "endogenous"]["value"].tail(28).mean()
        assert len(fc) == 4 and 0.6 < fc["forecast"].mean() / level < 1.4, sid


def test_quantiles_are_ordered_and_form_an_interval(m, segs):
    q = m.infer_quantiles(next(iter(segs.values())))
    assert len(q) == 4 and {"median", "lo-80", "hi-80", "lo-90", "hi-90"} <= set(q.columns)
    assert (q["lo-90"] <= q["lo-80"]).all() and (q["lo-80"] <= q["median"]).all()
    assert (q["median"] <= q["hi-80"]).all() and (q["hi-80"] <= q["hi-90"]).all()
    assert ((q["hi-90"] - q["lo-90"]) > 0).all()


@pytest.mark.parametrize("h", [1, 4, 30])
def test_horizons(m, segs, h):
    df = next(iter(segs.values()))
    fc = m.infer(df, h)
    assert len(fc) == h and np.isfinite(fc["forecast"]).all()
    assert list(fc["date"]) == list(pd.date_range(pd.Timestamp(df["date"].max()) + pd.Timedelta(days=1), periods=h, freq="D"))


def test_save_load_and_unseen_series(m, segs, tmp_path):
    m.save(tmp_path / "pool" / "model.pkl")
    l = DeepARModule({}); l.load(tmp_path / "pool" / "model.pkl")
    df = next(iter(segs.values()))
    assert np.allclose(m.infer(df, 4)["forecast"], l.infer(df, 4)["forecast"], rtol=1e-4)   # loaded, no training data
    new = df.copy(); new["segment_id"] = "9999-INR-inflow"                                    # unseen series id, known categories
    assert len(l.infer(new, 4)) == 4
    alien = new.copy(); alien["currency"] = "CHF"                                             # unseen category cannot be embedded
    with pytest.raises(ValueError, match="unseen category"):
        l.infer(alien, 4)
    changed = df.copy(); changed["currency"] = "CHF"                                          # known series, attributes now differ
    with pytest.raises(ValueError, match="changed since pooled training"):
        l.infer(changed, 4)


def test_diagnostics_run_for_the_pooled_fit(m):
    d = m.diagnose()
    assert d.ran and "ljung_box" in d.details
