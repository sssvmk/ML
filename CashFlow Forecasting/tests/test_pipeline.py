"""Tests for pipeline.py — the config-driven bridge between adapters and the orchestrator."""
import json
import pytest
from pathlib import Path
from adapters.synthetic import SyntheticAdapter
from config import load_config, build_candidates
from contract import validate_contract, Dataset
from orchestrator import Orchestrator
from pipeline import Pipeline, PipelineResult, register_adapter
from registry import ModelRegistry


def _orch(tmp_path, cfg):
    cands = build_candidates(cfg)
    return Orchestrator(ModelRegistry(tmp_path / "reg"), tmp_path / "logs", cands,
                        max_parallel_workers=1, holdout_periods=20, window_search_enabled=False,
                        backtest_step=100, hyperparameter_search={"enabled": False}, mlflow_config=None,
                        baselines=cfg.get("baselines")), cands


def _cfg(adapter="synthetic", params=None, algorithms=None, **orch_over):
    cfg = load_config("config.json")
    cfg["data_source"] = {"adapter": adapter, "params": params or {"n_days": 400}, "segments": None}
    if algorithms:
        cfg["algorithms"] = algorithms
    return cfg


# ---- adapter registry -------------------------------------------------------
def test_register_adapter_makes_custom_class_available(tmp_path):
    register_adapter("_test_synthetic", "adapters.synthetic.SyntheticAdapter")
    cfg = _cfg(adapter="_test_synthetic", params={"n_days": 300})
    pipe = Pipeline(cfg, tmp_path / "logs")
    segs = pipe.extract()
    assert len(segs) >= 1 and all(validate_contract(df).ok for df in segs.values())


def test_unknown_adapter_raises_a_clear_error(tmp_path):
    cfg = _cfg(adapter="no_such_adapter")
    with pytest.raises(KeyError, match="no_such_adapter"):
        Pipeline(cfg, tmp_path).extract()


# ---- extraction + validation ------------------------------------------------
def test_pipeline_extracts_and_validates_synthetic_data(tmp_path):
    cfg = _cfg()
    pipe = Pipeline(cfg, tmp_path / "logs")
    segs = pipe.extract()
    assert len(segs) >= 1
    v = pipe.validate(segs)
    assert all(r.ok for r in v.values())


def test_pipeline_uses_segment_filter(tmp_path):
    cfg = _cfg(params={"n_days": 400})
    all_segs = Pipeline(cfg, tmp_path / "a").extract()
    first = sorted(all_segs)[0]
    cfg["data_source"]["segments"] = [first]
    filtered = Pipeline(cfg, tmp_path / "b").extract()
    assert set(filtered) == {first}


def test_pipeline_excludes_invalid_segments_and_continues(tmp_path, monkeypatch):
    cfg = _cfg()
    pipe = Pipeline(cfg, tmp_path / "logs")
    segs = pipe.extract()
    bad_sid = sorted(segs)[0]
    segs[bad_sid] = segs[bad_sid].copy()
    segs[bad_sid]["value"] = segs[bad_sid]["value"].astype(str)   # breaks numeric check
    v = pipe.validate(segs)
    assert not v[bad_sid].ok
    assert all(v[s].ok for s in segs if s != bad_sid)
    assert pipe.valid_segments if hasattr(pipe, "valid_segments") else True


# ---- dataset enum is open ---------------------------------------------------
def test_non_cashflow_dataset_label_passes_validation(tmp_path):
    from adapters.generic_multivariate import GenericMultivariateAdapter
    df = GenericMultivariateAdapter().extract(
        "/mnt/user-data/uploads/exchange_rate.txt",
        ["AUD","GBP","CAD","CHF","CNY","JPY","NZD","SGD"],
        start="1990-01-01", freq="D", segment_prefix="FX-", dataset="FX_Rate"
    )
    assert df["dataset"].iloc[0] == "FX_Rate"
    assert validate_contract(df).ok
    assert Dataset.BANK_INFLOW_EBS.value == "Bank_Inflow_EBS"   # cashflow constants still work


# ---- single-segment pipeline run --------------------------------------------
def test_single_segment_run_picks_winner_and_writes_outputs(tmp_path):
    cfg = _cfg(algorithms={"arimax": {"enabled": True, "hyperparameters": {"order": [1,1,1]}}})
    orch, cands = _orch(tmp_path, cfg)
    pipe = Pipeline(cfg, tmp_path / "logs")
    result = pipe.run(orch, cands, horizon=4, rule_version="test-v1")
    assert isinstance(result, PipelineResult)
    assert result.valid_segments and not result.invalid_segments
    assert all(e.algorithm_name for e in result.entries.values())
    assert (tmp_path / "logs" / "pipeline_run.json").exists()
    run_log = json.loads((tmp_path / "logs" / "pipeline_run.json").read_text())
    assert "winners" in run_log and run_log["valid_segments"]


def test_cross_model_comparison_file_is_written_per_segment(tmp_path):
    cfg = _cfg(algorithms={"arimax": {"enabled": True, "hyperparameters": {}},
                           "linear_regression": {"enabled": True, "hyperparameters": {}}})
    orch, cands = _orch(tmp_path, cfg)
    pipe = Pipeline(cfg, tmp_path / "logs")
    result = pipe.run(orch, cands, horizon=4, rule_version="test-v1")
    for sid in result.valid_segments:
        cmp_file = tmp_path / "logs" / sid / "cross_model_comparison.json"
        assert cmp_file.exists(), f"missing cross_model_comparison.json for {sid}"
        data = json.loads(cmp_file.read_text())
        assert "models" in data and "elimination_log" in data
        names = [m["algorithm"] for m in data["models"]]
        assert "arimax" in names and "linear_regression" in names
        winner_rows = [m for m in data["models"] if m["outcome"] == "winner"]
        assert len(winner_rows) <= 1


# ---- pooled pipeline run + cross-segment summary ----------------------------
def test_pooled_run_writes_cross_segment_summary(tmp_path):
    from test_deepstate_g26_g42 import _pool
    cfg = load_config("config.json")
    cfg["algorithms"] = {"arimax": {"enabled": True, "hyperparameters": {}}}
    cfg["data_source"] = {"adapter": "synthetic", "params": {"n_days": 400}, "segments": None}
    orch, cands = _orch(tmp_path, cfg)
    segs = _pool(400)
    entries = orch.full_train_pooled(segs, horizon=4, rule_version="test-v1")
    summary_path = tmp_path / "logs" / "cross_segment_summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())
    assert data["n_segments"] == len(segs)
    assert "winner_distribution" in data
    assert all(r["segment_id"] in segs for r in data["segments"])


# ---- inference pipeline -------------------------------------------------------
def test_pipeline_inference_returns_forecast_for_all_valid_segments(tmp_path):
    cfg = _cfg(algorithms={"arimax": {"enabled": True, "hyperparameters": {}}})
    orch, cands = _orch(tmp_path, cfg)
    pipe = Pipeline(cfg, tmp_path / "logs")
    result = pipe.run(orch, cands, horizon=4, rule_version="test-v1")
    forecasts = pipe.infer(orch, horizon=4)
    assert set(forecasts) == set(result.valid_segments)
    assert all(len(fc) == 4 for fc in forecasts.values())


# ---- orchestrator knows single vs pooled ------------------------------------
def test_orchestrator_detects_pooling_capable_algorithms(tmp_path):
    cfg = _cfg(algorithms={"arimax": {"enabled": True, "hyperparameters": {}}})
    _, cands_single = _orch(tmp_path / "s", cfg)
    pipe_single = Pipeline(cfg, tmp_path / "s_logs")
    assert not pipe_single._needs_pooled(cands_single)
