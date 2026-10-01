"""G-36 ETSformer acceptance + D-12 vendoring rules (pinned commit, notice retained, unmodified, isolated, compatibility)."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from adapters.synthetic import SyntheticAdapter
from algorithms.etsformer import ETSformerModule
from algorithms._vendor_etsformer import VENDOR_DIR, PKG_DIR, PKG_NAME

FAST = {"n_lags": 48, "max_epochs": 3, "horizon": 4, "d_model": 16, "n_heads": 2}


@pytest.fixture(scope="module")
def df():
    return SyntheticAdapter().extract(n_days=900)


@pytest.fixture(scope="module")
def m(df):
    x = ETSformerModule(dict(FAST)); x.train(df); return x


# ---------------------------------------------------------------- D-12 vendoring rules
def test_vendored_files_are_unmodified_and_pinned():
    rec = json.loads((VENDOR_DIR / "FILES.sha256.json").read_text())
    assert rec["commit"] == "082555c3638d80dcc7655fc5f316b5a18fd93867"
    for name, h in rec["files"].items():
        assert hashlib.sha256((PKG_DIR / name).read_bytes()).hexdigest() == h, f"{name} differs from the pinned upstream file"
    assert "082555c3638d80dcc7655fc5f316b5a18fd93867" in (VENDOR_DIR / "PROVENANCE.md").read_text()


def test_bsd3_notice_is_retained_in_the_vendored_directory():
    lic = (VENDOR_DIR / "LICENSE.txt").read_text()
    for needle in ("Copyright (c) 2022, Salesforce.com, Inc.", "Redistributions of source code must retain",
                   "Neither the name of Salesforce.com"):
        assert needle in lic
    assert "Salesforce" in (VENDOR_DIR / "NOTICE.md").read_text()


def test_oss_scan_exists_and_flags_nothing_unreviewed():
    scan = (VENDOR_DIR / "OSS_SCAN.md").read_text()
    assert "Autoformer" in scan and "Informer2020" in scan and "identical to pristine" in scan
    assert "NO" not in [l.split("|")[3].strip() for l in scan.splitlines() if l.startswith("| `") and l.count("|") == 5 and "sha256" not in l and "`..." in l]
    assert "Flagged for human review (containment >= 0.15): none" in scan


def test_vendored_tree_is_reachable_only_through_a_private_module_name():
    from algorithms._vendor_etsformer import load_etsformer
    load_etsformer()                                                # the loader is lazy: trigger it
    assert PKG_NAME in sys.modules
    assert "models" not in sys.modules                              # the upstream top-level name `models` is never claimed


def test_compatibility_on_the_target_python(m):
    # unmodified upstream code, written for Python 3.8 / torch 1.11, trains and runs here (matrix G-36 requirement)
    assert m._fitted_model and m.n_parameters() > 0 and np.isfinite(m.infer(SyntheticAdapter().extract(n_days=900), 4)["forecast"]).all()


# ---------------------------------------------------------------- G-36 architecture
def test_is_the_official_model_not_a_stand_in(m):
    net = m.network()
    assert type(net.model).__name__ == "ETSformer" and type(net.model).__module__.startswith(PKG_NAME)
    names = {type(x).__name__ for x in net.modules()}
    assert {"ExponentialSmoothing", "GrowthLayer", "FourierLayer", "LevelLayer", "DampingLayer"} <= names
    assert not getattr(ETSformerModule, "is_simplified_stand_in", False) and "sklearn" not in type(net).__module__


def test_learned_smoothing_and_damping_factors_are_in_the_open_unit_interval(m):
    es = [x for x in m.network().modules() if type(x).__name__ == "ExponentialSmoothing"]
    assert es
    for x in es:
        a = x.weight.detach()
        assert bool(((a > 0) & (a < 1)).all())
    for x in (x for x in m.network().modules() if type(x).__name__ == "DampingLayer"):
        d = x.damping_factor.detach()
        assert bool(((d > 0) & (d < 1)).all())


def test_components_are_returned_separately_and_sum_to_the_forecast(m, df):
    d = m.decompose(df)
    assert list(d.columns) == ["date", "level", "growth", "season", "forecast"] and len(d) == 4
    assert np.allclose(d["level"] + d["growth"] + d["season"], d["forecast"], atol=0.05)
    assert np.allclose(d["forecast"], m.infer(df, 4)["forecast"], atol=0.05)
    assert d["season"].abs().sum() > 0 and d["growth"].abs().sum() > 0            # not degenerate components


def test_top_k_frequencies_and_layers_change_the_network(df):
    a = ETSformerModule({**FAST, "K": 1, "e_layers": 1}); a.train(df)
    b = ETSformerModule({**FAST, "K": 4, "e_layers": 3}); b.train(df)
    assert b.n_parameters() > a.n_parameters()
    ks = lambda mod: {x.k for x in mod.network().modules() if type(x).__name__ == "FourierLayer"}
    assert ks(a) == {1} and ks(b) == {4}
    assert ETSformerModule({"n_lags": 24, "K": 50})._configs(1).K == 10             # K cannot exceed the window's frequencies
    with pytest.raises(ValueError, match="divisible"):
        ETSformerModule({"d_model": 30, "n_heads": 4}).build_network(1)


def test_training_augmentation_is_reference_behaviour_and_seeded(df):
    a, b = ETSformerModule({**FAST, "std": 0.2}), ETSformerModule({**FAST, "std": 0.2})
    a.train(df); b.train(df)
    assert np.allclose(a.infer(df, 4)["forecast"], b.infer(df, 4)["forecast"])     # same seed -> same model despite the noise


# ---------------------------------------------------------------- module contract
@pytest.mark.parametrize("h", [1, 4, 30])
def test_horizons(m, df, h):
    fc = m.infer(df, h)
    assert len(fc) == h and np.isfinite(fc["forecast"]).all()
    assert list(fc["date"]) == list(pd.date_range(pd.Timestamp(df["date"].max()) + pd.Timedelta(days=1), periods=h, freq="D"))


def test_save_load_without_training_data(m, df, tmp_path):
    m.save(tmp_path / "v" / "model.pkl")
    l = ETSformerModule({}); l.load(tmp_path / "v" / "model.pkl")
    assert np.allclose(m.infer(df, 4)["forecast"], l.infer(df, 4)["forecast"], atol=1e-3)


def test_required_observations_and_no_eligibility_condition():
    assert not ETSformerModule.has_eligibility_condition
    assert ETSformerModule({"n_lags": 48, "horizon": 4}).required_observations() == 1000
    assert ETSformerModule({"n_lags": 800, "horizon": 4}).required_observations() == 1304
    assert {"n_lags", "d_model", "e_layers", "K", "dropout", "std"} <= set(ETSformerModule({}).hyperparameter_search_space())
