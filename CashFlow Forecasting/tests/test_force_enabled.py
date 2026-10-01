"""force_enabled (this request): every algorithm, and the elimination-gate/challenger/fallback baselines, are config-driven.
A `config.json -> algorithms.<id>.force_enabled: true` makes the orchestrator pick and run exactly that model -- training
it (or, for a purely pretrained model, just resolving/loading it) and installing it as the Production winner
unconditionally, bypassing candidate competition/ranking and, for a custom model, G-42 admission too."""
import json
import pytest
from adapters.synthetic import SyntheticAdapter
from config import build_candidates, load_config
from orchestrator import Orchestrator
from registry import ModelRegistry


def _cfg(**algo_overrides):
    cfg = load_config("config.json")
    cfg["algorithms"] = {"arimax": dict(cfg["algorithms"]["arimax"]), "linear_regression": dict(cfg["algorithms"]["linear_regression"])}
    for algo, over in algo_overrides.items():
        cfg["algorithms"][algo] = over
    return cfg


def _orch(tmp_path, cfg, **kw):
    return Orchestrator(ModelRegistry(tmp_path / "reg"), tmp_path / "logs", build_candidates(cfg), max_parallel_workers=1,
                        holdout_periods=20, window_search_enabled=False, backtest_step=100,
                        hyperparameter_search={"enabled": False}, mlflow_config=None, baselines=cfg.get("baselines"), **kw)


def test_force_enabled_tags_the_factory():
    cfg = _cfg(linear_regression={"enabled": True, "force_enabled": True, "hyperparameters": {}})
    cands = build_candidates(cfg)
    assert cands["linear_regression"].force_enabled is True and cands["arimax"].force_enabled is False


def test_forced_single_segment_model_wins_regardless_of_backtest(tmp_path):
    cfg = _cfg(linear_regression={"enabled": True, "force_enabled": True, "hyperparameters": {}})
    orch = _orch(tmp_path, cfg)
    df = SyntheticAdapter().extract(n_days=400)
    entry = orch.full_train(df["segment_id"].iloc[0], df, horizon=4, rule_version="r1")
    assert entry.algorithm_name == "linear_regression"
    assert entry.elimination_log["forced_override"] is True and "force_enabled=true" in entry.elimination_log["reason"]
    assert entry.metrics and "mase" in entry.metrics                       # still has real backtest metrics for the record


def test_more_than_one_forced_algorithm_is_refused(tmp_path):
    cfg = _cfg(linear_regression={"enabled": True, "force_enabled": True, "hyperparameters": {}},
               arimax={"enabled": True, "force_enabled": True, "hyperparameters": {}})
    orch = _orch(tmp_path, cfg)
    df = SyntheticAdapter().extract(n_days=400)
    with pytest.raises(ValueError, match="more than one algorithm has force_enabled"):
        orch.full_train(df["segment_id"].iloc[0], df, horizon=4, rule_version="r1")


def test_forcing_a_pooling_capable_algorithm_through_single_segment_full_train_is_refused(tmp_path):
    cfg = _cfg(deepvar={"enabled": True, "force_enabled": True, "hyperparameters": {"context_length": 40, "max_epochs": 3, "steps_per_epoch": 5, "batch_size": 16}})
    orch = _orch(tmp_path, cfg)
    df = SyntheticAdapter().extract(n_days=400)
    with pytest.raises(ValueError, match="full_train_pooled"):
        orch.full_train(df["segment_id"].iloc[0], df, horizon=4, rule_version="r1")


def test_forced_deepvar_bypasses_admission_and_wins_every_pooled_segment(tmp_path):
    """The real point of this request: DeepVAR, currently NOT admitted (G-42), can still be forced into production."""
    from admission import is_admitted
    from algorithms.deepvar import DeepVARModule
    cfg = load_config("config.json")
    assert not is_admitted("deepvar", DeepVARModule, cfg)[0]                # confirm it's genuinely not admitted right now
    cfg = _cfg(deepvar={"enabled": True, "force_enabled": True,
                        "hyperparameters": {"context_length": 40, "max_epochs": 6, "steps_per_epoch": 8, "batch_size": 16, "hidden_size": 16}})
    assert "deepvar" in build_candidates(cfg)                               # force_enabled bypassed the admission exclusion
    orch = _orch(tmp_path, cfg)
    segs = {sid: seg for sid, seg in list(__import__("tests.test_deepstate_g26_g42", fromlist=["_pool"])._pool(500).items())}
    entries = orch.full_train_pooled(segs, horizon=4, rule_version="r1")
    assert set(entries) == set(segs)
    for sid, e in entries.items():
        assert e.algorithm_name == "deepvar" and e.elimination_log["forced_override"] is True
        assert "G-42 admission bypassed" in e.elimination_log["reason"]
        assert e.pooled_group and sid in e.pooled_segments
    fc = orch.daily_infer_pool(segs, 4)
    assert all(len(fc[sid]) == 4 for sid in segs)


def test_baselines_are_config_driven_and_fall_back_to_todays_defaults(tmp_path):
    cfg = _cfg(linear_regression={"enabled": True, "hyperparameters": {}})
    cfg["baselines"] = {"elimination_gate": {"season": 14}, "challengers": {"naive_1": {"enabled": False}, "seasonal_naive_30": {"enabled": True, "season": 60}},
                        "fallback": {"combine": "rolling_mean_median", "lookback": 21}}
    orch = _orch(tmp_path, cfg)
    assert orch.baseline_factory().season == 14
    assert set(orch.challenger_factories) == {"seasonal_naive_30_challenger"}
    assert orch.challenger_factories["seasonal_naive_30_challenger"]().season == 60
    assert orch._fallback_hp == {"combine": "rolling_mean_median", "lookback": 21}
    default_orch = _orch(tmp_path / "d", _cfg(linear_regression={"enabled": True, "hyperparameters": {}}))
    assert default_orch.baseline_factory().season == 7                      # unset -> today's original hardcoded defaults
    assert set(default_orch.challenger_factories) == {"naive_1_challenger", "seasonal_naive_30_challenger"}
    assert default_orch._fallback_hp == {"combine": "rolling_mean_median", "lookback": 14}
