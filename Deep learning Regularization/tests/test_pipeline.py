import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from regpipe.evaluate import metrics_from_logits
from regpipe.tuning import tune

ROOT = Path(__file__).resolve().parent.parent


def test_metrics_perfect_and_se():
    y = torch.arange(10).repeat(20)
    logits = torch.nn.functional.one_hot(y, 10).float() * 10
    m = metrics_from_logits(logits, y)
    assert m["accuracy"] == 1.0 and m["auc"] == pytest.approx(1.0) and m["accuracy_se"] == 0.0
    logits[:20] = logits[:20].roll(1, dims=1)
    m = metrics_from_logits(logits, y)
    p = m["accuracy"]
    assert m["accuracy_se"] == pytest.approx((p * (1 - p) / len(y)) ** 0.5)


def test_tuning_never_touches_the_test_set_and_evaluates_defaults_first(cfg, data, ctx):
    import copy
    import mlflow
    from regpipe.methods.l2_weight_decay import L2WeightDecay
    blind = copy.copy(data)
    blind.x_test = blind.y_test = None                     # any access would raise
    with mlflow.start_run():
        best, trials = tune(L2WeightDecay(), blind, cfg, ctx)
    assert trials.iloc[0]["params_alpha"] == pytest.approx(L2WeightDecay().defaults()["alpha"])
    assert len(trials) == cfg["tune"]["n_trials"] and "alpha" in best


def test_predictor_rejects_out_of_range_pixels(data, ctx, tmp_path):
    from regpipe.bundle import Predictor, save_bundle
    from regpipe.model import MLP
    save_bundle(tmp_path / "b", MLP([8]), {})
    with pytest.raises(ValueError):
        Predictor(tmp_path / "b").predict(np.full((1, 784), 255.0))


@pytest.mark.slow
def test_end_to_end_smoke(tmp_path):
    """Whole pipeline in a throwaway store: tune -> train -> test-once -> MLflow -> register -> fresh-process reload -> infer."""
    out = tmp_path / "res"
    r = subprocess.run([sys.executable, "run.py", "--smoke", "--out", str(out), "--methods",
                        "l2_weight_decay,dropout,bagging"], cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads((out / "results.json").read_text())
    assert [x["method"] for x in res["rows"]][0] == "baseline" and len(res["rows"]) == 4
    reg = json.loads((out / "registry.json").read_text())
    assert reg["reload_check"]["passed"] and reg["alias_state"] == "candidate"
    for name in ("baseline", "dropout", "bagging"):
        assert (out / "methods" / name / "loss_curve.png").exists()
        assert (out / "methods" / name / "bundle" / "model.pt").exists()
    # inference CLI on a saved bundle
    x = np.random.rand(3, 784).astype("float32")
    np.save(tmp_path / "x.npy", x)
    r = subprocess.run([sys.executable, "infer.py", "--model", str(out / "methods" / "dropout" / "bundle"),
                        "--npy", str(tmp_path / "x.npy")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    preds = json.loads(r.stdout)
    assert len(preds) == 3 and all(0 <= p["label"] <= 9 and abs(sum(p["probabilities"]) - 1) < 1e-4 for p in preds)
    # test set evaluated once per method
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{out / 'mlflow.db'}")
    runs = mlflow.search_runs(search_all_experiments=True, filter_string="tags.phase = 'method'")
    assert (runs["tags.test_set_evaluations"] == "1").all() and len(runs) == 4
