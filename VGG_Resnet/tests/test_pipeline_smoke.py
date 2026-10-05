"""Tiny end-to-end run: train -> test -> register -> fresh-process reload -> promote -> predict."""
import json

import numpy as np
import pytest

from src.utils import ROOT, load_config


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # sqlite/artifact files land in tmp
    return {"tracking_uri": f"sqlite:///{tmp_path}/mlflow.db", "registered_model_name": "smoke-vgg"}


@pytest.mark.parametrize("model_overrides,explicit_name", [
    (["model.arch=vgg11", "model.width_mult=0.125", "model.head_hidden=64"], True),
    (["model.arch=resnet20", "model.width_mult=0.5"], False),   # registry name derived: resnet20-synthetic
])
def test_end_to_end(env, tmp_path, model_overrides, explicit_name):
    if not explicit_name:
        env = {**env, "registered_model_name": None}
    cfg = load_config(ROOT / "config" / "train.yaml", [
        "data.dataset=synthetic", "data.synthetic_size=400", "data.num_workers=0", "data.batch_size=32",
        *model_overrides,
        "training.max_epochs=4", "training.lr=0.05", "training.warmup_epochs=1", "training.device=cpu",
        "sanity.overfit_steps=200", "metric.target_value=0.02", "experiment_name=smoke",
    ])
    from src.train import run_training
    res = run_training(cfg, env)
    assert res["best_val_top1"] > 0.15  # synthetic classes are separable by colour; chance is 0.10
    run_dir = ROOT / "reports" / "runs" / res["run_id"]
    assert (run_dir / "metrics.csv").exists() and (run_dir / "diagnosis.json").exists()

    from src.evaluate import run_test_evaluation
    t = run_test_evaluation(cfg, env, res["run_id"])
    assert t["evaluation_count"] == 1
    with pytest.raises(RuntimeError):  # second look at the test set is refused
        run_test_evaluation(cfg, env, res["run_id"])

    from src.register import register_candidate
    reg = register_candidate(cfg, env, res["run_id"])
    assert reg["load_verified"] is True

    card = ROOT / "reports" / "MODEL_CARD.md"
    made_card = not card.exists()
    if made_card:
        card.write_text("# test card\n")
    try:
        from src.promote import run_gates
        rec = run_gates(cfg, env, reg["version"], approver="smoke-test")
        assert reg["name"] == ("smoke-vgg" if explicit_name else "resnet20-synthetic")
        assert rec["gates"]["test_set_used_exactly_once"] and rec["gates"]["load_verified"]
        assert rec["passed"], rec["gates"]
        bad = run_gates(cfg, env, reg["version"], approver=None)
        assert not bad["passed"]
    finally:
        if made_card:
            card.unlink()
        for f in (ROOT / "approvals").glob("*synthetic*"):
            f.unlink()
        for f in (ROOT / "approvals").glob("smoke-vgg*"):
            f.unlink()
        (ROOT / "approvals" / "promotion_log.jsonl").unlink(missing_ok=True)

    from src.serve import Predictor
    pred = Predictor(f"models:/{reg['name']}@champion", env["tracking_uri"])
    out = pred.predict([np.zeros((32, 32, 3), np.uint8), np.full((40, 50, 3), 200, np.uint8)], topk=3)
    assert len(out) == 2 and len(out[0]["top_k"]) == 3
    assert abs(sum(c["prob"] for c in pred.predict([np.zeros((32, 32, 3), np.uint8)], topk=100)[0]["top_k"]) - 1) < 1e-3
    json.dumps(out)
