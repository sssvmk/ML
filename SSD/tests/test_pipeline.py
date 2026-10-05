"""End-to-end on the synthetic dataset with a tiny ResNet18 SSD (CPU): train -> test (once) -> register -> predict."""
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

from ssd_voc.config import load_config
from ssd_voc.env import setup_environment
from ssd_voc.evaluate import run_test_evaluation
from ssd_voc.serve import Predictor, register_candidate
from ssd_voc.train import resolve_init, run_training, setup_mlflow


def make_world(tmp_path, *extra, stage="voc"):
    paths = setup_environment(tmp_path / "dest")
    cfg = load_config(stage, [*SMOKE, *extra], dest=tmp_path / "dest")
    return cfg, paths


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("trained")
    env, tdir = dict(os.environ), tempfile.tempdir
    cfg, paths = make_world(tmp, 'schedule.phases=[{"lr":0.01,"iters":18},{"lr":0.001,"iters":12}]', "schedule.eval_every=6",
                            "schedule.overfit_gap_warn=-1.0")
    summary = run_training(cfg, paths)
    yield cfg, paths, summary
    os.environ.clear()
    os.environ.update(env)
    tempfile.tempdir = tdir


def test_training_produces_all_artifacts_inside_dest(trained):
    cfg, paths, s = trained
    run_dir = Path(s["out_dir"])
    for f in ("metrics.csv", "diagnosis.json", "next_steps.json", "sanity.json", "qualitative_val.png", "curves.png",
              "val_per_class_ap50.json", "val_error_breakdown.json", "class_names.json"):
        assert (run_dir / f).exists(), f
    assert (run_dir / "checkpoints" / "best.pt").exists() and (run_dir / "checkpoints" / "last.pt").exists()
    assert Path(s["stable_checkpoint"]) == paths["checkpoints"] / "voc_best.pt" and Path(s["stable_checkpoint"]).exists()
    assert run_dir.resolve().is_relative_to(paths["dest"].resolve())
    assert (paths["mlflow"] / "mlflow.db").exists() and any(paths["mlflow_artifacts"].iterdir())   # tracking + artifacts in dest
    assert Path(tempfile.gettempdir()).resolve() == paths["tmp"].resolve()
    assert json.loads((run_dir / "sanity.json").read_text())["passed"]
    assert s["stop_reason"] == "schedule_complete" and s["iterations"] == 30 and s["diagnosis"]


def test_mlflow_captures_params_tags_and_every_metric_family(trained):
    cfg, paths, s = trained
    setup_mlflow(cfg, paths)
    run = MlflowClient().get_run(s["run_id"])
    m, t, p = run.data.metrics, run.data.tags, run.data.params
    assert {"train_loss", "val_loss", "val_metric", "val_map50", "train_metric", "lr", "train_conf", "val_loc", "grad_norm",
            "best_val_metric", "final_val_map50", "final_val_map", "generalisation_gap", "total_iterations"} <= set(m)
    assert t["stage"] == "voc" and t["sanity_passed"] == "true" and t["test_evaluated"] == "false"
    assert t["overfit_warning"] == "true"                  # threshold forced to -1 in this fixture: the guard triggers
    assert t["stop_reason"] == "schedule_complete" and "synthetic" in t["data_version"] and str(paths["dest"]) == t["dest"]
    assert p["schedule.weight_decay"] == "0.0005" and p["loss.neg_pos_ratio"] == "3" and int(p["num_priors"]) > 0
    hist = MlflowClient().get_metric_history(s["run_id"], "val_metric")
    assert len(hist) == 5 and [h.step for h in hist] == [6, 12, 18, 24, 30]


def test_metrics_csv_has_the_diagnosis_columns_and_a_phase_change(trained):
    rows = list(csv.DictReader(open(Path(trained[2]["out_dir"]) / "metrics.csv")))
    assert {"epoch", "train_loss", "val_loss", "train_metric", "val_metric", "lr", "phase"} <= set(rows[0])
    assert [int(r["phase"]) for r in rows] == [0, 0, 0, 1, 1] and float(rows[-1]["lr"]) == pytest.approx(0.001)
    assert all(np.isfinite(float(r["val_loss"])) for r in rows)


def test_checkpoint_schema_and_best_is_the_best_validation_checkpoint(trained):
    cfg, paths, s = trained
    st = torch.load(s["stable_checkpoint"], map_location="cpu", weights_only=False)
    assert {"state_dict", "cfg", "classes", "iteration", "val_metric", "ema", "stage", "num_foreground"} <= set(st)
    rows = list(csv.DictReader(open(Path(s["out_dir"]) / "metrics.csv")))
    assert st["val_metric"] == pytest.approx(max(float(r["val_metric"]) for r in rows)) and st["ema"] is True


def test_early_stopping_takes_the_next_lr_step_then_stops_in_the_last_phase(tmp_path):
    cfg, paths = make_world(tmp_path, 'schedule.phases=[{"lr":0.01,"iters":60},{"lr":0.001,"iters":60}]', "schedule.eval_every=3",
                            "schedule.early_stopping.patience=1", "schedule.early_stopping.min_delta=10.0")
    s = run_training(cfg, paths)       # min_delta=10 means no evaluation after the first ever counts as an improvement
    assert s["stop_reason"] == "early_stop_plateau" and s["iterations"] < 120 and s["iterations"] == 9
    setup_mlflow(cfg, paths)
    changes = MlflowClient().get_metric_history(s["run_id"], "phase_change_iteration")
    assert [c.value for c in changes] == [6.0]                    # one early LR drop (plateau), then the stop
    rows = list(csv.DictReader(open(Path(s["out_dir"]) / "metrics.csv")))
    assert [int(r["phase"]) for r in rows] == [0, 0, 1] and (Path(s["out_dir"]) / "checkpoints" / "best.pt").exists()


def test_without_advance_on_plateau_the_phase_runs_to_its_end(tmp_path):
    cfg, paths = make_world(tmp_path, 'schedule.phases=[{"lr":0.01,"iters":12},{"lr":0.001,"iters":6}]', "schedule.eval_every=3",
                            "schedule.early_stopping.patience=1", "schedule.early_stopping.min_delta=10.0",
                            "schedule.early_stopping.advance_phase_on_plateau=false")
    s = run_training(cfg, paths)
    rows = list(csv.DictReader(open(Path(s["out_dir"]) / "metrics.csv")))
    assert [int(r["phase"]) for r in rows][:4] == [0, 0, 0, 0] and s["iterations"] > 12 and s["stop_reason"] == "early_stop_plateau"


def test_resume_continues_and_carries_the_history_and_best_checkpoint(tmp_path):
    cfg1, paths = make_world(tmp_path, 'schedule.phases=[{"lr":0.01,"iters":6}]', "schedule.eval_every=6")
    s1 = run_training(cfg1, paths)
    last = Path(s1["out_dir"]) / "checkpoints" / "last.pt"
    cfg2 = load_config("voc", [*SMOKE, 'schedule.phases=[{"lr":0.01,"iters":12}]', "schedule.eval_every=6", "sanity.enabled=false",
                               f"resume_from={last}"], dest=paths["dest"])
    s2 = run_training(cfg2, paths)
    rows = list(csv.DictReader(open(Path(s2["out_dir"]) / "metrics.csv")))
    assert [int(r["iteration"]) for r in rows] == [6, 12] and s2["iterations"] == 12
    assert (Path(s2["out_dir"]) / "checkpoints" / "best.pt").exists()


def test_fine_tuning_from_another_dataset_skips_class_heads(tmp_path):
    cfg1, paths = make_world(tmp_path, 'schedule.phases=[{"lr":0.01,"iters":6}]', "schedule.eval_every=6")
    s1 = run_training(cfg1, paths)
    cfg2 = load_config("voc", [*SMOKE, "data.synthetic.classes=5", 'schedule.phases=[{"lr":0.01,"iters":6}]', "schedule.eval_every=6",
                               f"init_from={s1['stable_checkpoint']}"], dest=paths["dest"])
    s2 = run_training(cfg2, paths)
    setup_mlflow(cfg2, paths)
    assert MlflowClient().get_run(s2["run_id"]).data.tags["init_from"] == s1["stable_checkpoint"]
    assert torch.load(s2["stable_checkpoint"], map_location="cpu", weights_only=False)["num_foreground"] == 5


def test_second_round_stage_resolves_the_coco_checkpoint_and_asks_for_it_when_missing(tmp_path):
    paths = setup_environment(tmp_path / "dest")
    cfg = load_config("voc_from_coco", dest=paths["dest"])
    with pytest.raises(FileNotFoundError, match="run the stage that produces it first"):
        resolve_init(cfg, paths)
    (paths["checkpoints"] / "coco_best.pt").write_bytes(b"x")
    assert resolve_init(cfg, paths) == paths["checkpoints"] / "coco_best.pt"
    long_cfg = load_config("voc_from_coco_long", dest=paths["dest"])
    with pytest.raises(FileNotFoundError, match="coco_long"):
        resolve_init(long_cfg, paths)                       # the long variant needs the long COCO model, not the short one
    (paths["checkpoints"] / "coco_long_best.pt").write_bytes(b"x")
    assert resolve_init(long_cfg, paths).name == "coco_long_best.pt"
    assert resolve_init(load_config("voc", dest=paths["dest"]), paths) is None


def test_failed_sanity_check_stops_before_any_training(tmp_path):
    cfg, paths = make_world(tmp_path, "sanity.initial_loss_tol=-1.0")
    with pytest.raises(RuntimeError, match="sanity checks failed"):
        run_training(cfg, paths)
    assert not list(paths["checkpoints"].glob("*.pt"))


def test_test_set_is_evaluated_exactly_once_then_registered_and_served(trained):
    cfg, paths, s = trained
    res = run_test_evaluation(cfg, paths, s["run_id"])
    assert res["evaluation_count"] == 1 and 0.0 <= res["test_map50"] <= 1.0 and "error_breakdown" in res
    with pytest.raises(RuntimeError, match="already evaluated"):
        run_test_evaluation(cfg, paths, s["run_id"])
    assert MlflowClient().get_run(s["run_id"]).data.metrics["test_map50"] == pytest.approx(res["test_map50"])
    ex = Image.fromarray(np.random.default_rng(0).integers(0, 255, (96, 128, 3), dtype=np.uint8))
    reg = register_candidate(cfg, paths, s["run_id"], ex, "ssd-test-model")
    assert reg["load_verified"] is True
    mv = MlflowClient().get_model_version_by_alias("ssd-test-model", "candidate")
    assert mv.tags["load_verified"] == "true" and mv.tags["test_metric"] != "not evaluated"
    # the registered pyfunc and the checkpoint predictor agree
    model = mlflow.pyfunc.load_model(reg["model_uri"])
    import base64
    import io
    buf = io.BytesIO()
    ex.save(buf, "PNG")
    import pandas as pd
    df = pd.DataFrame({"image_b64": [base64.b64encode(buf.getvalue()).decode()]})
    got = json.loads(model.predict(df, params={"score_threshold": 0.0})["detections"].iloc[0])
    want = Predictor(s["stable_checkpoint"]).predict([ex], score_threshold=0.0)[0]
    assert (got["width"], got["height"]) == (want["width"], want["height"]) == (128, 96) and len(got["detections"]) == len(want["detections"]) > 0
    assert np.allclose([d["box"] for d in got["detections"]], [d["box"] for d in want["detections"]], atol=0.05)
    high = json.loads(model.predict(df, params={"score_threshold": 0.999})["detections"].iloc[0])["detections"]
    assert len(high) <= len(got["detections"])


def test_predictor_input_validation_and_thresholds(trained):
    pred = Predictor(trained[2]["stable_checkpoint"])
    out = pred.predict([np.zeros((50, 60, 3), np.uint8)], score_threshold=0.0, max_detections=5)[0]
    assert (out["width"], out["height"]) == (60, 50) and len(out["detections"]) <= 5
    assert all(0 <= d["box"][0] <= 60 and 0 <= d["box"][1] <= 50 and 1 <= d["label_id"] <= 3 for d in out["detections"])
    assert pred.predict([np.zeros((50, 60, 3), np.uint8)], score_threshold=1.01)[0]["detections"] == []
    with pytest.raises(ValueError, match="uint8"):
        pred.predict([np.zeros((50, 60, 3), np.float32)])


def test_benchmark_reports_model_and_augmentation_throughput(tmp_path):
    from ssd_voc.train import benchmark
    cfg, paths = make_world(tmp_path, "data.num_workers=0")
    out = benchmark(cfg, paths, steps=2, loader_batches=4)
    assert out["model_img_per_s"] > 0 and out["augmentation_img_per_s"] > 0 and out["bottleneck"] in ("model", "data loading")
    assert out["est_minutes_if_run_to_the_end"] > 0 and out["schedule_iterations_max"] == 18
