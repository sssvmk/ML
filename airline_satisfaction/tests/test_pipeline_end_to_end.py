import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from airsat.analysis.final_evaluation import HoldoutAlreadyUsed, evaluate_on_test
from airsat.bundle import ModelBundle
from airsat.predict import predict_file


def test_run_creates_all_artifacts(smoke_run):
    for rel in ["schema.json", "selection.json", "model_comparison.md", "test_evaluation.json", "champion/bundle.joblib",
                "reports/experiment_report.md", "reports/MODEL_CARD.md", "eda/eda_summary.json", "splits/population_report.json",
                "reports/agent_explanations/index.md", "algorithms/lightgbm_gradient_boosted_trees/plots/roc_pr.png",
                "algorithms/logistic_regression_elasticnet/assumption_checks.json"]:
        assert (smoke_run / rel).exists(), rel


def test_baseline_is_floor_and_models_learn(smoke_run):
    sel = json.load(open(smoke_run / "selection.json"))
    t = {r["name"]: r["val_primary"] for r in sel["table"]}
    assert abs(t["majority_class_baseline"] - 0.5) < 1e-9
    assert t["logistic_regression_elasticnet"] > 0.7
    assert sel["winner"] != "majority_class_baseline"


def test_test_set_is_locked_after_first_use(smoke_run, fast_cfg, csv_path):
    assert (smoke_run / "test_set_used.lock").exists()
    X = pd.read_csv(csv_path).drop(columns=["id", "satisfaction"])
    with pytest.raises(HoldoutAlreadyUsed):
        evaluate_on_test(smoke_run / "champion", X, np.zeros(len(X), dtype=int), fast_cfg, smoke_run)


def test_every_algorithm_documents_assumption_checks(smoke_run):
    for name in ["logistic_regression_elasticnet", "lightgbm_gradient_boosted_trees"]:
        r = json.load(open(smoke_run / "algorithms" / name / "result.json"))
        assert len(r["assumptions"]) >= 6 and r["loss_function"] and r["hyperparameter_docs"]
    lr = json.load(open(smoke_run / "algorithms" / "logistic_regression_elasticnet" / "result.json"))
    assert {"linearity_of_logit", "low_multicollinearity", "no_complete_separation"} <= {a["name"] for a in lr["assumptions"]}


def test_inference_on_unseen_file_matches_sample_format(smoke_run, fast_cfg, csv_path, tmp_path):
    out = predict_file(str(Path(csv_path).parent / "unseen.csv"), str(smoke_run / "champion"), str(tmp_path / "pred.csv"), fast_cfg, drift=True)
    assert list(out.columns) == ["id", "satisfaction"] and len(out) == 600
    assert set(out["satisfaction"]) <= {"TRUE", "FALSE"}
    assert (tmp_path / "pred.drift.json").exists()


def test_bundle_roundtrip_is_deterministic(smoke_run, csv_path):
    X = pd.read_csv(csv_path).drop(columns=["id", "satisfaction"]).head(300)
    a = ModelBundle.load(smoke_run / "champion").predict_proba(X)
    b = ModelBundle.load(smoke_run / "champion").predict_proba(X.iloc[::-1])[::-1]
    assert np.allclose(a, b, atol=1e-9)


def test_bundle_rejects_missing_columns(smoke_run, csv_path):
    from airsat.validation import InputValidationError
    X = pd.read_csv(csv_path).drop(columns=["id", "satisfaction", "Age"]).head(5)
    with pytest.raises(InputValidationError):
        ModelBundle.load(smoke_run / "champion").predict_proba(X)
