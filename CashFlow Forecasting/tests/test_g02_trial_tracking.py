"""G-02 acceptance: a 12-trial search yields 12 child runs under one parent, each with the §3.3.10 fields."""
import pytest
from adapters.synthetic import SyntheticAdapter
from algorithms.ridge import RidgeModule
from mlflow_logging import MLflowLogger
from search import random_then_adaptive_search


def test_g02_twelve_trials_twelve_child_runs_with_all_fields(tmp_path):
    df = SyntheticAdapter().extract(n_days=600)
    tracker = MLflowLogger({"mlflow": {"enabled": True, "tracking_uri": f"sqlite:///{tmp_path}/mlflow.db",
                                       "experiment_name": "g02"}})
    if not tracker.active:
        pytest.skip("mlflow unavailable")
    sr = random_then_adaptive_search(RidgeModule, {"n_lags": 3, "alpha": 1.0}, df, window=500, horizon=4, budget=12,
                                     seed=5, window_bounds=(1, 500), n_exog=2, step=40, tracker=tracker,
                                     context={"segment_id": "seg", "rule_version": "r1"})
    assert len(sr.trials) == 12
    import mlflow
    client = mlflow.tracking.MlflowClient(tracker.tracking_uri)
    exp = client.get_experiment_by_name("g02")
    runs = client.search_runs([exp.experiment_id], max_results=100)
    parent = [r for r in runs if r.data.tags.get("process") == "hyperparameter_search"]
    kids = [r for r in runs if r.data.tags.get("mlflow.parentRunId") == parent[0].info.run_id]
    assert len(parent) == 1 and len(kids) == 12
    need_params = {"algorithm", "algorithm_version", "search_space_version", "dataset_version_id", "code_version_id",
                   "bt.horizon", "window", "trial_status"}
    for k in kids:
        assert need_params <= set(k.data.params), need_params - set(k.data.params)
        assert k.data.tags["trial_status"] in {"completed", "pruned", "failed"}
        assert any(m.startswith("agg_") for m in k.data.metrics) or k.data.tags["trial_status"] == "failed"
    assert sum(1 for k in kids if k.data.tags.get("selected") == "true") == 1
