"""G-39 TimesFM-3 adapter: zero-shot, path-only weights, non-commercial licence guard, eligibility per decision D-13.
The real `timesfm3` code path is exercised with a TINY random checkpoint in the standard directory layout (the real weights are
not available offline); accuracy of the real weights is therefore not tested here."""
import hashlib
import numpy as np
import pandas as pd
import pytest
import torch
from adapters.synthetic import SyntheticAdapter
from algorithms.prebuilt import PrebuiltModelUnavailable, resolve_prebuilt
from algorithms.timesfm3 import TimesFM3Module, _CACHE

FAST = {"horizon": 4, "context_length": 128}


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from timesfm3 import ModelConfig, ResidualBlockConfig, StackedTransformersConfig, TransformerConfig
    from timesfm3.torch.timesfm3_forecaster import _make_torch_model
    torch.manual_seed(0)
    res = ResidualBlockConfig(hidden_dims=32, output_dims=32, use_bias=False, activation="relu")
    tr = StackedTransformersConfig(num_layers=1, transformer=TransformerConfig(model_dims=32, hidden_dims=32, num_heads=2, attention_norm="rms",
        feedforward_norm="rms", qk_norm="rms", use_rope_seq=True, use_rope_var=True, use_bias=False, ff_activation="relu", deterministic=True))
    model = _make_torch_model(ModelConfig(residual_block_config=res, transformer_config=tr, device="cpu"))
    d = tmp_path_factory.mktemp("tiny_timesfm3")
    model.save_pretrained(str(d))
    return d


def _entry(ckpt, **over):
    e = {"enabled": True, "path": str(ckpt), "fine_tune": False, "expected_fingerprint": None, "licence_id": "timesfm-non-commercial-v1.0",
         "use_scope": "non_commercial", "non_commercial_use_acknowledged": True, "allowed_environments": ["dev", "test"]}
    e.update(over)
    return e


def _mod(ckpt, env="dev", **over):
    entry = over.pop("entry", _entry(ckpt))
    return TimesFM3Module({**FAST, **over, "prebuilt": entry, "environment": env})


@pytest.fixture(scope="module")
def df():
    return SyntheticAdapter().extract(n_days=900)


# ------------------------------------------------------------------------------------- availability and licence guard
def test_unavailable_until_enabled_pathed_and_present_and_the_reason_names_the_key(df, tmp_path):
    for entry, needle in [(None, "missing"), ({"enabled": False}, "enabled is false"),
                          ({"enabled": True, "path": None}, "path is not set"),
                          ({"enabled": True, "path": str(tmp_path / "nope")}, "does not exist")]:
        with pytest.raises(PrebuiltModelUnavailable, match=needle) as e:
            TimesFM3Module({**FAST, "prebuilt": entry, "environment": "dev"}).train(df)
        assert "prebuilt_models.timesfm3" in str(e.value)


def test_licence_guard_needs_acknowledgement_and_a_permitted_environment(ckpt, df):
    with pytest.raises(PrebuiltModelUnavailable, match="NON-COMMERCIAL"):
        _mod(ckpt, entry=_entry(ckpt, non_commercial_use_acknowledged=False)).train(df)
    with pytest.raises(PrebuiltModelUnavailable, match="production"):
        _mod(ckpt, env="production").train(df)
    with pytest.raises(PrebuiltModelUnavailable, match="production"):
        _mod(ckpt, env=None).train(df)                                             # an unknown environment is not assumed harmless
    _mod(ckpt, env="dev").train(df)                                                # acknowledged + permitted environment: allowed
    resolve_prebuilt({"enabled": True, "path": str(ckpt), "licence_id": "Apache-2.0"}, "other", default_fine_tune=False)   # no restriction on other weights


def test_fingerprint_is_recorded_and_a_mismatch_is_refused(ckpt, df):
    m = _mod(ckpt); m.train(df)
    fp = m._fitted_model["prebuilt"]["fingerprint"]
    assert len(fp) == 64 and m._fitted_model["prebuilt"]["use_scope"] == "non_commercial"
    with pytest.raises(PrebuiltModelUnavailable, match="fingerprint"):
        _mod(ckpt, entry=_entry(ckpt, expected_fingerprint="0" * 64)).train(df)
    _mod(ckpt, entry=_entry(ckpt, expected_fingerprint=fp)).train(df)


# ------------------------------------------------------------------------------------- zero-shot, offline
def test_train_fits_nothing_and_the_weights_are_untouched(ckpt, df):
    m = _mod(ckpt); m.train(df)
    before = {k: v.clone() for k, v in m._model.model.state_dict().items()}
    m.train(df); m.infer(df, 4)
    assert all(torch.equal(before[k], v) for k, v in m._model.model.state_dict().items())
    es = m._fitted_model["early_stopping"]
    assert m._fitted_model["zero_shot"] and es["used"] is False and "zero-shot" in es["note"]
    assert m.n_parameters() == 42848


def test_weights_are_read_only_from_the_path_never_downloaded(ckpt, df, monkeypatch):
    import huggingface_hub
    def boom(*a, **k):
        raise AssertionError("a download was attempted")
    for name in ("hf_hub_download", "snapshot_download"):
        monkeypatch.setattr(huggingface_hub, name, boom, raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _CACHE.clear()
    m = _mod(ckpt); m.train(df)
    assert len(m.infer(df, 4)) == 4


def test_the_loaded_weights_are_shared_between_modules(ckpt, df):
    a, b = _mod(ckpt), _mod(ckpt)
    a.train(df); b.train(df)
    assert a._model is b._model                                                     # a backtest builds a module per fold: load once


# ------------------------------------------------------------------------------------- eligibility (D-13)
def test_eligibility_is_only_context_plus_horizon_with_no_1000_floor(ckpt, df):
    m = _mod(ckpt)
    assert m.has_eligibility_condition and m.required_observations() == 128 + 4
    assert TimesFM3Module({"horizon": 4}).required_observations() == 512 + 4        # default context length 512
    assert TimesFM3Module({"horizon": 4, "context_length": 1024}).required_observations() == 1028
    assert m.check_eligibility(df).eligible
    exact = df[df["date"] < df["date"].min() + np.timedelta64(132, "D")]
    short = df[df["date"] < df["date"].min() + np.timedelta64(131, "D")]
    assert m.check_eligibility(exact).eligible                                      # exactly context + horizon is enough
    r = m.check_eligibility(short)
    assert not r.eligible and "128" in r.reason and "horizon 4" in r.reason


# ------------------------------------------------------------------------------------- inference
@pytest.mark.parametrize("h", [1, 4, 30])
def test_horizons_exact_dated_rows(ckpt, df, h):
    m = _mod(ckpt); m.train(df)
    fc = m.infer(df, h)
    assert len(fc) == h and np.isfinite(fc["forecast"]).all()
    assert list(fc["date"]) == list(pd.date_range(pd.Timestamp(df["date"].max()) + pd.Timedelta(days=1), periods=h, freq="D"))


def test_uses_only_the_last_context_values(ckpt, df):
    m = _mod(ckpt); m.train(df)
    base = m.infer(df, 4)["forecast"].to_numpy()
    old = df.copy()
    first = sorted(old["date"].unique())[:50]
    old.loc[old["date"].isin(first) & (old["series_role"] == "endogenous"), "value"] += 1e6      # far older than the 128-period context
    assert np.allclose(base, m.infer(old, 4)["forecast"].to_numpy())
    recent = df.copy()
    last = sorted(recent["date"].unique())[-5:]
    recent.loc[recent["date"].isin(last) & (recent["series_role"] == "endogenous"), "value"] *= 3
    assert not np.allclose(base, m.infer(recent, 4)["forecast"].to_numpy())
    with pytest.raises(ValueError, match="shorter than the context"):
        m.infer(df[df["date"] < df["date"].min() + np.timedelta64(60, "D")], 4)


def test_quantiles_are_available_and_ordered(ckpt, df):
    m = _mod(ckpt); m.train(df)
    q = m.infer_quantiles(df, 6)
    cols = [f"q{i}0" for i in range(1, 10)]
    assert list(q.columns) == ["date"] + cols and len(q) == 6
    assert (np.diff(q[cols].to_numpy(), axis=1) >= -1e-9).all()                      # monotone across quantiles


def test_is_the_timesfm3_library_model_and_forecasts_the_target_alone(ckpt, df):
    m = _mod(ckpt); m.train(df)
    assert type(m._model).__name__ == "TimesFM3Forecaster" and type(m._model.model).__name__ == "TimesFM3Torch"
    assert not m.describe_contract()["requires_exogenous"]


# ------------------------------------------------------------------------------------- save / load
def test_save_load_stores_a_reference_not_the_weights(ckpt, df, tmp_path):
    m = _mod(ckpt); m.train(df)
    m.save(tmp_path / "v" / "model.pkl")
    assert (tmp_path / "v" / "model.pkl").stat().st_size < 50_000                     # no weights inside the artifact
    l = TimesFM3Module({}); l.load(tmp_path / "v" / "model.pkl")
    assert np.allclose(m.infer(df, 4)["forecast"], l.infer(df, 4)["forecast"])
    # the configured path is resolved again on load: a missing checkpoint fails loudly and names the key
    import shutil
    moved = tmp_path / "gone"
    shutil.copytree(ckpt, moved)
    m2 = _mod(ckpt, entry=_entry(ckpt, path=str(moved))); m2.train(df)
    m2.save(tmp_path / "w" / "model.pkl")
    shutil.rmtree(moved)
    with pytest.raises(PrebuiltModelUnavailable, match="prebuilt_models.timesfm3"):
        TimesFM3Module({}).load(tmp_path / "w" / "model.pkl")


# ------------------------------------------------------------------------------------- pipeline
def _orch(tmp_path, entry):
    from config import load_config, build_candidates
    from orchestrator import Orchestrator
    from registry import ModelRegistry
    cfg = load_config("config.json")
    cfg["algorithms"] = {"timesfm3": {"enabled": True, "hyperparameters": {"context_length": 128}}}
    cfg["prebuilt_models"]["timesfm3"] = entry
    return Orchestrator(ModelRegistry(tmp_path / "reg"), tmp_path / "logs", build_candidates(cfg), max_parallel_workers=1, holdout_periods=20,
                        window_search_enabled=False, backtest_step=100, hyperparameter_search={"enabled": False}, mlflow_config=None)


def test_unavailable_model_is_visible_in_the_elimination_log_not_silent(tmp_path, df):
    orch = _orch(tmp_path, {"enabled": False, "path": None})
    e = orch.full_train(df["segment_id"].iloc[0], df, horizon=4, rule_version="r1")
    assert "timesfm3" in e.elimination_log["eliminated_error"]


def test_available_model_runs_through_backtest_ranking_and_daily_inference(tmp_path, ckpt, df):
    orch = _orch(tmp_path, _entry(ckpt))
    seg = df["segment_id"].iloc[0]
    e = orch.full_train(seg, df, horizon=4, rule_version="r1")
    assert "timesfm3" not in e.elimination_log["eliminated_error"], e.elimination_log
    if e.algorithm_name == "timesfm3":
        assert len(orch.daily_infer(seg, df, 4)) == 4
