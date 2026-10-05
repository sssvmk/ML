"""Stage-level training on a synthetic dataset with a tiny ResNet-18 DETR (CPU)."""
import csv
import json
import os
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pytest
import torch
from mlflow.tracking import MlflowClient
from PIL import Image
from smoke_over import SMOKE

from detr_voc.config import load_config
from detr_voc.env import setup_environment
from detr_voc.evaluate import run_test_evaluation
from detr_voc.infer import Predictor
from detr_voc.serve import package_model
from detr_voc.train import benchmark, run_training, setup_mlflow

PH = 'schedule.phases=[{"lr":0.0003,"epochs":2},{"lr":0.00003,"epochs":1}]'
BASE = [o for o in SMOKE if not o.startswith("schedule.phases")]


def world(tmp_path, *extra):
    paths = setup_environment(tmp_path / "dest")
    return load_config("voc", [*BASE, PH, *extra], dest=tmp_path / "dest"), paths


def rows_of(summary):
    return list(csv.DictReader(open(Path(summary["out_dir"]) / "metrics.csv")))


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    env, tdir = dict(os.environ), tempfile.tempdir
    cfg, paths = world(tmp_path_factory.mktemp("trained"), "schedule.overfit_gap_warn=-1.0")
    yield cfg, paths, run_training(cfg, paths)
    os.environ.clear()
    os.environ.update(env)
    tempfile.tempdir = tdir


def test_training_writes_every_artifact_inside_dest(trained):
    cfg, paths, s = trained
    run_dir = Path(s["out_dir"])
    for f in ("metrics.csv", "diagnosis.json", "next_steps.json", "sanity.json", "qualitative_val.png", "curves.png",
              "val_per_class_ap50.json", "val_error_breakdown.json", "class_names.json", "stage.json"):
        assert (run_dir / f).exists(), f
    assert (run_dir / "checkpoints" / "best.pt").exists() and (run_dir / "checkpoints" / "last.pt").exists()
    assert Path(s["stable_checkpoint"]) == paths["checkpoints"] / "voc_best.pt" and Path(s["stable_checkpoint"]).exists()
    assert run_dir.resolve().is_relative_to(paths["dest"].resolve()) and (paths["mlflow"] / "mlflow.db").exists()
    assert json.loads((run_dir / "sanity.json").read_text())["passed"] and json.loads((run_dir / "stage.json").read_text())["status"] == "complete"
    assert s["epochs"] == 3 and s["stop_reason"] == "schedule_complete" and s["diagnosis"]


def test_mlflow_captures_params_tags_and_metric_families(trained):
    cfg, paths, s = trained
    setup_mlflow(cfg, paths)
    run = MlflowClient().get_run(s["run_id"])
    m, t, p = run.data.metrics, run.data.tags, run.data.params
    assert {"train_loss", "val_loss", "val_metric", "val_map50", "train_metric", "train_ce", "val_bbox", "lr", "grad_norm", "best_val_metric",
            "final_val_map50", "final_val_map", "generalisation_gap", "total_epochs", "train_loss_epoch"} <= set(m)
    assert t["model"] == "DETR" and t["sanity_passed"] == "true" and t["test_evaluated"] == "false" and t["overfit_warning"] == "true"
    assert p["schedule.grad_clip"] == "0.1" and p["loss.eos_coef"] == "0.1" and p["model.num_queries"] == "10" and int(p["num_parameters"]) > 0
    assert [h.step for h in MlflowClient().get_metric_history(s["run_id"], "val_metric")] == [1, 2, 3]


def test_paper_learning_rates_per_phase_and_a_lower_backbone_rate(trained):
    cfg, paths, s = trained
    rows = rows_of(s)
    assert {"epoch", "train_loss", "val_loss", "train_metric", "val_metric", "lr", "phase"} <= set(rows[0])
    assert [int(r["phase"]) for r in rows] == [0, 0, 1] and [float(r["lr"]) for r in rows] == pytest.approx([3e-4, 3e-4, 3e-5])
    last = torch.load(Path(s["out_dir"]) / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    assert [g["lr"] for g in last["optimizer"]["param_groups"]] == pytest.approx([3e-5, 1e-6])      # transformer 1e-4 -> 1e-5 style, backbone 10x lower
    assert all(g["weight_decay"] == 1e-4 for g in last["optimizer"]["param_groups"]) and last["epoch"] == 3


def test_best_checkpoint_is_the_best_validation_epoch(trained):
    cfg, paths, s = trained
    st = torch.load(s["stable_checkpoint"], map_location="cpu", weights_only=False)
    assert {"state_dict", "cfg", "classes", "epoch", "val_metric", "ema", "stage", "num_foreground"} <= set(st)
    assert st["val_metric"] == pytest.approx(max(float(r["val_metric"]) for r in rows_of(s))) and st["ema"] is False


def test_early_stopping_takes_the_lower_lr_early_then_stops(tmp_path):
    cfg, paths = world(tmp_path, 'schedule.phases=[{"lr":0.0003,"epochs":20},{"lr":0.00003,"epochs":20}]',
                       "schedule.early_stopping.patience=1", "schedule.early_stopping.min_delta=10.0")
    s = run_training(cfg, paths)                    # min_delta=10: nothing after the first evaluation counts as an improvement
    assert s["stop_reason"] == "early_stop_plateau" and s["epochs"] == 3
    setup_mlflow(cfg, paths)
    assert [c.value for c in MlflowClient().get_metric_history(s["run_id"], "phase_change_epoch")] == [2.0]
    assert [int(r["phase"]) for r in rows_of(s)] == [0, 0, 1]


def test_without_advance_on_plateau_the_phase_runs_to_its_end(tmp_path):
    cfg, paths = world(tmp_path, 'schedule.phases=[{"lr":0.0003,"epochs":3},{"lr":0.00003,"epochs":3}]',
                       "schedule.early_stopping.patience=1", "schedule.early_stopping.min_delta=10.0",
                       "schedule.early_stopping.advance_phase_on_plateau=false")
    s = run_training(cfg, paths)
    assert [int(r["phase"]) for r in rows_of(s)] == [0, 0, 0, 1] and s["epochs"] == 4 and s["stop_reason"] == "early_stop_plateau"


def test_resume_continues_and_carries_the_history_forward(tmp_path):
    cfg1, paths = world(tmp_path, 'schedule.phases=[{"lr":0.0003,"epochs":1}]')
    s1 = run_training(cfg1, paths)
    cfg2 = load_config("voc", [*BASE, 'schedule.phases=[{"lr":0.0003,"epochs":2}]', "sanity.enabled=false",
                               f"resume_from={Path(s1['out_dir']) / 'checkpoints' / 'last.pt'}"], dest=paths["dest"])
    s2 = run_training(cfg2, paths)
    assert [int(r["epoch"]) for r in rows_of(s2)] == [1, 2] and s2["epochs"] == 2 and (Path(s2["out_dir"]) / "checkpoints" / "best.pt").exists()


def test_ema_option_saves_the_averaged_weights(tmp_path):
    cfg, paths = world(tmp_path, 'schedule.phases=[{"lr":0.0003,"epochs":2}]', "schedule.ema.enabled=true", "schedule.ema.tau=3",
                       "schedule.ema.decay=0.9")
    s = run_training(cfg, paths)
    assert torch.load(s["stable_checkpoint"], map_location="cpu", weights_only=False)["ema"] is True


def test_failed_sanity_check_stops_before_training(tmp_path):
    cfg, paths = world(tmp_path, "sanity.initial_loss_tol=-1.0")
    with pytest.raises(RuntimeError, match="sanity checks failed"):
        run_training(cfg, paths)
    assert not list(paths["checkpoints"].glob("*.pt"))


def test_test_once_then_package_and_serve(trained):
    cfg, paths, s = trained
    res = run_test_evaluation(cfg, paths, s["run_id"])
    assert res["evaluation_count"] == 1 and 0.0 <= res["test_map50"] <= 1.0
    with pytest.raises(RuntimeError, match="already evaluated"):
        run_test_evaluation(cfg, paths, s["run_id"])
    ex = Image.fromarray(np.random.default_rng(0).integers(0, 255, (96, 128, 3), dtype=np.uint8))
    pkg = package_model(cfg, paths, s["run_id"], ex, paths["dest"] / "models" / "m")
    assert pkg["load_verified"] is True
    for f in ("checkpoint.pt", "detr_pyfunc.py", "README.txt", "expected.json"):
        assert (paths["dest"] / "models" / "m" / f).exists()
    import base64
    import io
    import pandas as pd
    buf = io.BytesIO()
    ex.save(buf, "PNG")
    df = pd.DataFrame({"image_b64": [base64.b64encode(buf.getvalue()).decode()]})
    got = json.loads(mlflow.pyfunc.load_model(pkg["model_uri"]).predict(df, params={"score_threshold": 0.0})["detections"].iloc[0])
    want = Predictor(s["stable_checkpoint"]).predict([ex], score_threshold=0.0)[0]
    assert (got["width"], got["height"]) == (want["width"], want["height"]) == (128, 96) and len(got["detections"]) == len(want["detections"]) == 10
    assert np.allclose([d["box"] for d in got["detections"]], [d["box"] for d in want["detections"]], atol=0.05)


def test_predictor_validation_and_thresholds(trained):
    pred = Predictor(trained[2]["stable_checkpoint"])
    out = pred.predict([np.zeros((50, 60, 3), np.uint8)], score_threshold=0.0, max_detections=4)[0]
    assert (out["width"], out["height"]) == (60, 50) and len(out["detections"]) == 4
    scores = [d["score"] for d in out["detections"]]
    assert scores == sorted(scores, reverse=True) and all(0 <= d["box"][0] <= 60 and 1 <= d["label_id"] <= 3 for d in out["detections"])
    assert pred.predict([np.zeros((50, 60, 3), np.uint8)], score_threshold=1.01)[0]["detections"] == []
    with pytest.raises(ValueError, match="uint8"):
        pred.predict([np.zeros((50, 60, 3), np.float32)])


def test_benchmark_reports_speed_and_a_time_estimate(tmp_path):
    cfg, paths = world(tmp_path, "data.num_workers=0")
    out = benchmark(cfg, paths, steps=2, loader_batches=4)
    assert out["model_img_per_s"] > 0 and out["augmentation_img_per_s"] > 0 and out["bottleneck"] in ("model", "data loading")
    assert out["est_hours_if_run_to_the_end"] >= 0 and out["schedule_epochs_max"] == 3
