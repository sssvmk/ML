"""Tiny end-to-end run on synthetic data: train -> test -> register -> fresh-process reload -> promote -> predict -> refit."""
import json

import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient
from PIL import Image

from src.utils import ROOT, load_config

OVERRIDES = [
    "data.dataset=synthetic", "data.synthetic_size=32", "data.val_fraction=0.25", "data.num_workers=0",
    "data.batch_size=4", "model.pretrained=none", "model.min_size=[96]", "model.max_size=128",
    "model.box_batch_size_per_image=32", "model.score_floor=0.0", "training.max_epochs=2", "training.warmup_iters=4",
    "training.device=cpu", "training.eval_train_subset=4", "training.ema.tau=5", "sanity.overfit_steps=40",
    "sanity.overfit_drop=0.8", "metric.target_value=0.0", "experiment_name=smoke",
]


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # sqlite / mlruns land in tmp
    return {"tracking_uri": f"sqlite:///{tmp_path}/mlflow.db", "registered_model_name": None}


def test_end_to_end_including_refit(env):
    cfg = load_config(ROOT / "config" / "train.yaml", OVERRIDES)
    from src.train import run_training
    res = run_training(cfg, env)
    run_dir = ROOT / "reports" / "runs" / res["run_id"]
    for f in ("metrics.csv", "diagnosis.json", "next_steps.json", "sanity.json", "qualitative_val.png",
              "reference_stats.json", "val_error_breakdown.json", "curves.png"):
        assert (run_dir / f).exists(), f
    assert json.loads((run_dir / "sanity.json").read_text())["passed"]
    assert isinstance(res["best_val_metric"], float)
    client = MlflowClient(tracking_uri=env["tracking_uri"])
    metrics = client.get_run(res["run_id"]).data.metrics
    assert {"train_loss", "val_loss", "val_map50", "train_loss_classifier", "lr", "best_val_metric",
            "best_epoch"} <= set(metrics)
    assert client.get_run(res["run_id"]).data.tags["sanity_passed"] == "true"

    from src.evaluate import run_test_evaluation
    t = run_test_evaluation(cfg, env, res["run_id"])
    assert t["evaluation_count"] == 1 and 0.0 <= t["test_map50_voc07"] <= 1.0
    with pytest.raises(RuntimeError):  # a second look at the test set is refused
        run_test_evaluation(cfg, env, res["run_id"])

    from src.register import register_candidate
    reg = register_candidate(cfg, env, res["run_id"])
    assert reg["load_verified"] is True and reg["name"].endswith("-synthetic")

    from src.promote import run_gates
    bad = run_gates(cfg, env, reg["version"], approver=None)
    assert not bad["passed"] and not bad["gates"]["approver_named"]
    rec = run_gates(cfg, env, reg["version"], approver="smoke-test")
    assert rec["passed"], rec["gates"]

    from src.serve import Predictor
    pred = Predictor(f"models:/{reg['name']}@champion", env["tracking_uri"])
    img_a = Image.fromarray(np.random.default_rng(0).integers(0, 255, (96, 96, 3), dtype=np.uint8))
    img_b = np.random.default_rng(1).integers(0, 255, (150, 200, 3), dtype=np.uint8)
    out = pred.predict([img_a, img_b], score_threshold=0.0, max_detections=7)
    assert len(out) == 2 and (out[0]["width"], out[1]["width"]) == (96, 200)
    assert all(len(o["detections"]) <= 7 for o in out) and out[0]["detections"]
    d = out[0]["detections"][0]
    assert set(d) == {"box", "score", "label_id", "label"} and 1 <= d["label_id"] <= 3 and d["label"].startswith("c")
    assert all(len(o["detections"]) == 0 for o in pred.predict([img_a], score_threshold=0.999))

    # refit on all of trainval for best_epoch + 1 epochs (early-stopping meta-algorithm)
    from src.refit import run_refit
    rf = run_refit(cfg, env, res["run_id"])
    rtags = client.get_run(rf["run_id"]).data.tags
    assert rtags["refit_of"] == res["run_id"] and rtags["phase"] == "refit" and rf["best_val_metric"] is None
    assert not [m for m in client.get_run(rf["run_id"]).data.metrics if m.startswith("val_")]
    run_test_evaluation(cfg, env, rf["run_id"])
    reg2 = register_candidate(cfg, env, rf["run_id"])
    assert reg2["load_verified"] and str(reg2["version"]) == "2"
    assert run_gates(cfg, env, reg2["version"], approver="smoke-test")["passed"]
    mlflow.set_tracking_uri(env["tracking_uri"])
    assert str(client.get_model_version_by_alias(reg["name"], "champion").version) == "2"
    for f in (ROOT / "approvals").glob("*synthetic*"):
        f.unlink()
    (ROOT / "approvals" / "promotion_log.jsonl").unlink(missing_ok=True)


def test_resume_continues_from_last_checkpoint(env):
    small = [o for o in OVERRIDES if not o.startswith(("training.max_epochs", "sanity."))]
    cfg1 = load_config(ROOT / "config" / "train.yaml", [*small, "training.max_epochs=1", "sanity.enabled=false",
                                                         "training.early_stopping.enabled=false"])
    from src.train import run_training
    r1 = run_training(cfg1, env)
    last = ROOT / "reports" / "runs" / r1["run_id"] / "checkpoints" / "last.pt"
    assert last.exists()
    cfg2 = load_config(ROOT / "config" / "train.yaml", [*small, "training.max_epochs=2", "sanity.enabled=false",
                                                         "training.early_stopping.enabled=false",
                                                         f"training.resume_from={last}"])
    r2 = run_training(cfg2, env)
    rows = (ROOT / "reports" / "runs" / r2["run_id"] / "metrics.csv").read_text().strip().splitlines()
    assert [r.split(",")[0] for r in rows] == ["epoch", "0", "1"]  # history carried over + the resumed epoch
    assert (ROOT / "reports" / "runs" / r2["run_id"] / "checkpoints" / "best.pt").exists()
    import torch
    s = torch.load(ROOT / "reports" / "runs" / r2["run_id"] / "checkpoints" / "last.pt", weights_only=False)
    assert s["epoch"] == 1 and s["lrc"]["it"] > 0 and s["ema"]["updates"] > 0
