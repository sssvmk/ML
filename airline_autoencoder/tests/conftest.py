import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from aeclf.config import load_config  # noqa: E402
from aeclf.synthetic import make_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def csv_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    df = make_synthetic(4000, seed=11)
    df["satisfaction"] = df["satisfaction"].map({True: "TRUE", False: "FALSE"})
    df.to_csv(d / "train.csv", index=False)
    make_synthetic(500, seed=12, with_target=False).to_csv(d / "unseen.csv", index=False)
    return d / "train.csv"


@pytest.fixture(scope="session")
def fast_cfg(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    return load_config(overrides=[f"project.output_dir={out}", "autoencoder.kinds=[ae,dae]", "autoencoder.search.n_trials=1", "autoencoder.max_epochs=8",
                                  "classifier.search.n_trials=1", "classifier.max_epochs=8", "label_efficiency.fractions=[0.1,1.0]",
                                  "label_efficiency.seeds=[0]", "selection.bootstrap_samples=100"])


@pytest.fixture(scope="session")
def smoke_run(csv_path, fast_cfg):
    from aeclf.run_pipeline import run

    return run(fast_cfg, str(csv_path), run_id="pytest_run", explain="offline")
