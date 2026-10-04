import copy
import json
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import pytest
import torch

from optpipe.evaluate import init_diagnostics, metrics_from_logits
from optpipe.methods import REGISTRY
from optpipe.tuning import tune

ROOT = Path(__file__).resolve().parent.parent


def test_metrics_perfect_and_standard_error():
    y = torch.arange(10).repeat(20)
    logits = torch.nn.functional.one_hot(y, 10).float() * 10
    m = metrics_from_logits(logits, y)
    assert m["accuracy"] == 1.0 and m["auc"] == pytest.approx(1.0) and m["accuracy_se"] == 0.0
    logits[:20] = logits[:20].roll(1, dims=1)
    m = metrics_from_logits(logits, y)
    p = m["accuracy"]
    assert m["accuracy_se"] == pytest.approx((p * (1 - p) / len(y)) ** 0.5)


def test_init_diagnostics_loss_near_ln_k_for_a_small_random_output_layer(ctx, data):
    from optpipe.model import MLP
    model = MLP([32, 16])
    with torch.no_grad():
        model.out.weight.mul_(1e-3)
    d = init_diagnostics(model, data.x_train, data.y_train, torch.device("cpu"))
    assert d["init_loss_over_lnK"] == pytest.approx(1.0, abs=0.02)
    assert len(d["init_act_std"]) == 2 and len(d["init_grad_norms"]) == 3


@pytest.mark.parametrize("name", ["adam", "lbfgs"])
def test_tuning_never_touches_the_test_set_and_evaluates_defaults_first(name, cfg, data, ctx):
    blind = copy.copy(data)
    blind.x_test = blind.y_test = None                                           # any access would raise
    m = REGISTRY[name]()
    with mlflow.start_run():
        best, trials = tune(m, blind, cfg, ctx)
    first = trials.iloc[0]
    for k, v in m.defaults().items():
        if f"params_{k}" in trials.columns and isinstance(v, (int, float)):
            assert first[f"params_{k}"] == pytest.approx(v), k
    assert len(trials) == cfg["tune"]["n_trials"] and {"l1", "l2"} <= set(best)


def test_diverging_trials_are_pruned_not_fatal(cfg, data, ctx):
    from optpipe.methods.sgd import MinibatchSGD

    class Exploding(MinibatchSGD):
        lr_range = (1e3, 1e4)
        def defaults(self):
            return {**super().defaults(), "lr": 5e3}

    with mlflow.start_run():
        best, trials = tune(Exploding(), data, cfg, ctx)                         # must return, falling back to defaults
    assert isinstance(best, dict)


def test_predictor_rejects_out_of_range_pixels(tmp_path):
    from optpipe.bundle import Predictor, save_bundle
    from optpipe.model import MLP
    save_bundle(tmp_path / "b", MLP([8]), {})
    with pytest.raises(ValueError):
        Predictor(tmp_path / "b").predict(np.full((1, 784), 255.0))


@pytest.mark.slow
def test_end_to_end_smoke(tmp_path):
    """tune -> train -> test-once -> MLflow -> register -> fresh-process reload -> inference CLI, in a throwaway store."""
    out = tmp_path / "res"
    r = subprocess.run([sys.executable, "run.py", "--smoke", "--out", str(out), "--methods",
                        "adam,lbfgs,design_aux_heads,bias_gate"], cwd=ROOT, capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stderr[-2500:]
    res = json.loads((out / "results.json").read_text())
    names = [x["method"] for x in res["rows"]]
    assert names[0] == "baseline" and len(names) == 5
    reg = json.loads((out / "registry.json").read_text())
    assert reg["reload_check"]["passed"] and reg["alias_state"] == "candidate"
    for n in names:
        assert (out / "methods" / n / "loss_curve.png").exists() and (out / "methods" / n / "bundle" / "model.pt").exists()
    bundle = json.loads((out / "methods" / "design_aux_heads" / "bundle" / "bundle.json").read_text())
    assert bundle["spec"]["aux_heads"] == []                                     # auxiliary heads were discarded
    assert all(x["passes"] > 0 for x in res["rows"])
    x = np.random.rand(3, 784).astype("float32")
    np.save(tmp_path / "x.npy", x)
    r = subprocess.run([sys.executable, "infer.py", "--model", str(out / "methods" / "lbfgs" / "bundle"),
                        "--npy", str(tmp_path / "x.npy")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    preds = json.loads(r.stdout)
    assert len(preds) == 3 and all(0 <= p["label"] <= 9 and abs(sum(p["probabilities"]) - 1) < 1e-4 for p in preds)
    mlflow.set_tracking_uri(f"sqlite:///{out / 'mlflow.db'}")
    runs = mlflow.search_runs(search_all_experiments=True, filter_string="tags.phase = 'method'")
    assert (runs["tags.test_set_evaluations"] == "1").all() and len(runs) == 5
    assert not (out / "methods" / "bfgs").exists()                               # only the requested methods ran
