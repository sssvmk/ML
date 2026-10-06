import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from airsat.config import load_config  # noqa: E402
from airsat.synthetic import make_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def csv_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    df = make_synthetic(4000, seed=11)
    df["satisfaction"] = df["satisfaction"].map({True: "TRUE", False: "FALSE"})
    p = d / "train.csv"
    df.to_csv(p, index=False)
    unseen = make_synthetic(600, seed=12, with_target=False)
    unseen.to_csv(d / "unseen.csv", index=False)
    return p


@pytest.fixture(scope="session")
def fast_cfg(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    return load_config(overrides=[f"project.output_dir={out}", "search.n_trials=2", "search.timeout_seconds=60",
                                  "cv_check.folds=2", "cv_check.subsample=2000", "diagnostics.permutation_importance_rows=1000",
                                  "selection.bootstrap_samples=100", "diagnostics.learning_curve_fractions=[0.3,1.0]"])


@pytest.fixture(scope="session")
def smoke_run(csv_path, fast_cfg):
    from airsat.run_pipeline import run

    return run(fast_cfg, str(csv_path), run_id="pytest_run",
               algorithms=["majority_class_baseline", "logistic_regression_elasticnet", "lightgbm_gradient_boosted_trees"],
               explain="offline")
