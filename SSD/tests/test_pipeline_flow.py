"""The sequential pipeline end to end on a synthetic dataset (tiny ResNet-18 SSD, CPU): the five trainings, three tests, three saved
models, resume, skip, device modes."""
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient
from smoke_over import SMOKE

import ssd_voc.pipeline as pl
import ssd_voc.train as tr
from ssd_voc.env import setup_environment
from ssd_voc.pipeline import (STEPS, BackgroundPrep, check_device, choose_device_mode, device_overrides, find_resumable,
                              pipeline_summary, run_pipeline, run_step, start_pipeline, step_detection, step_load_datasets,
                              step_second_round)
from ssd_voc.train import setup_mlflow

PHASES = 'schedule.phases=[{"lr":0.01,"iters":8},{"lr":0.001,"iters":4}]'
TINY = [o for o in SMOKE if not o.startswith(("schedule.phases", "schedule.eval_every", "schedule.log_every"))] + [
    PHASES, "schedule.eval_every=4", "schedule.log_every=2", "sanity.overfit_steps=30", "mlflow.backend=sqlite"]
STAGE_OVER = {"coco": ["data.synthetic.classes=5"], "coco_long": ["data.synthetic.classes=5"]}   # COCO-like stages: 5 classes


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    saved = dict(os.environ), tempfile.tempdir
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    yield
    os.environ.clear()
    os.environ.update(saved[0])
    tempfile.tempdir = saved[1]


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    saved = dict(os.environ), tempfile.tempdir
    path = tmp_path_factory.mktemp("pipeline") / "ssd"
    summary = run_pipeline(path, "cpu", TINY, STAGE_OVER)
    yield path, summary
    os.environ.clear()
    os.environ.update(saved[0])
    tempfile.tempdir = saved[1]


def runs_of(path):
    paths = setup_environment(path)
    setup_mlflow(pl.load_config("voc", TINY, dest=path), paths)
    client = MlflowClient()
    exp = client.get_experiment_by_name("smoke")
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time ASC"])
    return client, runs


def test_the_five_steps_run_in_the_required_order(full_run):
    path, summary = full_run
    assert [r["step"] for r in summary["steps"]] == ["detection", "coco", "second_round", "coco_long", "long_final"]
    assert [r["stage"] for r in summary["steps"]] == ["voc", "coco", "voc_from_coco", "coco_long", "voc_from_coco_long"]
    _, runs = runs_of(path)
    labels = [r.data.tags["mlflow.runName"] for r in runs]
    assert labels == ["1_detection", "2_coco", "3_second_round", "4_coco_long", "5_long_final"]
    assert all(r.data.tags["stop_reason"] for r in runs) and all("generalisation_gap" in r.data.metrics for r in runs)


def test_each_step_starts_from_the_right_model(full_run):
    path, _ = full_run
    _, runs = runs_of(path)
    init = {r.data.tags["mlflow.runName"]: r.data.tags["init_from"] for r in runs}
    assert init["1_detection"] == "imagenet-pretrained backbone" and init["2_coco"] == "imagenet-pretrained backbone"
    assert init["3_second_round"].endswith(str(Path("checkpoints") / "coco_best.pt"))
    assert init["4_coco_long"].endswith(str(Path("models") / "02_final_second_round" / "checkpoint.pt"))   # the SAVED final model
    assert init["5_long_final"].endswith(str(Path("checkpoints") / "coco_long_best.pt"))
    params = {r.data.tags["mlflow.runName"]: r.data.params for r in runs}
    assert params["1_detection"]["aug.expansion"] == "False" and params["3_second_round"]["aug.expansion"] == "False"
    assert params["4_coco_long"]["aug.expansion"] == "True" and params["5_long_final"]["aug.expansion"] == "True"
    assert params["5_long_final"]["schedule.scale"] == "2.0" and params["1_detection"]["schedule.scale"] == "1.0"
    assert params["2_coco"]["data.num_foreground"] == "5" and params["3_second_round"]["data.num_foreground"] == "3"


def test_tests_run_on_voc_steps_only_and_each_model_is_saved_and_verified(full_run):
    path, summary = full_run
    rows = {r["step"]: r for r in summary["steps"]}
    for k in ("detection", "second_round", "long_final"):
        assert rows[k]["test_map50"] is not None and rows[k]["saved_model"]
    for k in ("coco", "coco_long"):
        assert rows[k]["test_map50"] is None and rows[k]["saved_model"] is None
    for name in ("01_detection_voc", "02_final_second_round", "03_final_longer_schedule"):
        d = path / "models" / name
        for f in ("checkpoint.pt", "torchscript.pt", "priors.npy", "meta.json", "ssd_pyfunc.py", "README.txt", "expected.json"):
            assert (d / f).exists(), (name, f)
    _, runs = runs_of(path)
    tests = {r.data.tags["mlflow.runName"]: r.data.tags["test_evaluation_count"] for r in runs if "test_evaluation_count" in r.data.tags}
    assert tests == {"1_detection": "1", "3_second_round": "1", "5_long_final": "1"}      # each model: the test set once


def test_everything_is_under_the_path_and_the_state_file_records_progress(full_run):
    path, _ = full_run
    state = json.loads((path / "pipeline_state.json").read_text())
    assert state["device_mode"] == "cpu" and all(state["stages"][k]["train_done"] for k in ("detection", "coco", "second_round", "coco_long", "long_final"))
    assert (path / "pipeline_summary.json").exists() and (path / "mlflow" / "mlflow.db").exists()
    assert (path / "checkpoints" / "voc_best.pt").exists() and (path / "checkpoints" / "coco_long_best.pt").exists()
    assert Path(tempfile.gettempdir()).resolve() == (path / "tmp").resolve()
    stray = [p for p in path.parent.iterdir() if p.name != "ssd"]
    assert not stray                                                                   # nothing was written next to `path`


def test_overfitting_guards_run_in_every_step(full_run):
    path, summary = full_run
    for r in summary["steps"]:
        assert r["train_minus_val"] is not None and r["overfit_warning"] in (True, False) and r["best_val_map50"] is not None
    for run_dir in (path / "runs").iterdir():
        assert (run_dir / "sanity.json").exists() and (run_dir / "diagnosis.json").exists() and (run_dir / "next_steps.json").exists()
        assert json.loads((run_dir / "sanity.json").read_text())["passed"]


def test_running_again_with_the_same_path_skips_everything(full_run):
    path, _ = full_run
    client, before = runs_of(path)
    out = run_pipeline(path, "cpu", TINY, STAGE_OVER)
    _, after = runs_of(path)
    assert len(after) == len(before) == 5 and [r["step"] for r in out["steps"]][-1] == "long_final"


def test_an_interrupted_step_resumes_instead_of_restarting(tmp_path, monkeypatch):
    path = tmp_path / "ssd"
    over = [o for o in TINY if not o.startswith("schedule.phases")] + ['schedule.phases=[{"lr":0.01,"iters":16}]',
                                                                       "schedule.early_stopping.enabled=false"]
    ctx = start_pipeline(path, "cpu", over, STAGE_OVER)
    step_load_datasets(ctx)
    real, calls = tr.evaluate_model, {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 5:                      # the third validation pass of the COCO step: simulate a cluster failure
            raise RuntimeError("cluster went away")
        return real(*a, **k)
    monkeypatch.setattr(tr, "evaluate_model", flaky)
    with pytest.raises(RuntimeError, match="cluster went away"):
        run_step(ctx, "coco")
    last = find_resumable(ctx.paths, "coco")
    assert last is not None and last.exists()
    monkeypatch.setattr(tr, "evaluate_model", real)
    ctx2 = start_pipeline(path, "cpu", over, STAGE_OVER)          # "run the notebook again with the same path"
    st = run_step(ctx2, "coco")
    assert st["train_done"] and st["summary"]["iterations"] == 16
    import csv
    rows = list(csv.DictReader(open(Path(st["summary"]["out_dir"]) / "metrics.csv")))
    assert [int(r["iteration"]) for r in rows] == [4, 8, 12, 16]        # the history before the failure was carried over
    old = json.loads((last.parent.parent / "stage.json").read_text())
    assert old["status"] == "superseded" and find_resumable(ctx2.paths, "coco") is None


def test_device_questions_and_answers(monkeypatch):
    assert [choose_device_mode(v) for v in ("cpu", "GPU", "combination", "1", "2", "3")] == ["cpu", "gpu", "combination", "cpu", "gpu", "combination"]
    with pytest.raises(ValueError, match="cpu, gpu or combination|device must be"):
        choose_device_mode("tpu")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit, match="cpu, gpu or combination"):
        choose_device_mode(None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    answers = iter(["7", "", "2"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert choose_device_mode(None) == "gpu"                           # invalid answers are asked again


def test_device_settings_and_a_missing_gpu_is_reported_not_hidden(tmp_path):
    cpu, gpu, both = device_overrides("cpu", 24), device_overrides("gpu", 24), device_overrides("combination", 24)
    assert cpu["device"] == "cpu" and cpu["schedule.amp"] == "none" and cpu["data.num_workers"] == 6
    assert gpu == both and gpu["device"] == "cuda" and gpu["schedule.amp"] == "bf16" and gpu["data.num_workers"] == 20
    assert device_overrides("gpu", 6)["data.num_workers"] == 2
    assert check_device("cpu") == "cpu"
    with pytest.raises(RuntimeError, match="sees no GPU"):
        check_device("gpu")
    with pytest.raises(RuntimeError, match="Machine Learning"):
        start_pipeline(tmp_path / "x", "combination")


def test_combination_mode_downloads_the_later_datasets_in_the_background_while_gpu_stage_trains(tmp_path, monkeypatch):
    calls = []

    def recorder(ctx, download=True, index=True):
        calls.append((threading.current_thread().name, download, index, time.time()))
        time.sleep(0.5)
    monkeypatch.setattr(pl, "prepare_second_round_data", recorder)
    monkeypatch.setattr(pl.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pl.torch.cuda, "get_device_name", lambda i=0: "Fake A100")
    over = [*TINY, "device=cpu", "schedule.amp=none"]                  # the 'GPU' is simulated: the tiny model still runs on CPU
    ctx = start_pipeline(tmp_path / "ssd", "combination", over, STAGE_OVER)
    t0 = time.time()
    step_load_datasets(ctx)
    assert time.time() - t0 < 0.45 and ctx.background is not None            # returned before the download finished
    step_detection(ctx)
    assert calls[0][:3] == ("ssd-background-prep", True, False)                 # started in a worker thread, download only
    run_step(ctx, "coco")
    assert ctx.background is None and calls[1][:3] == ("MainThread", True, True)  # joined; the index is built in the main thread
    assert len(calls) == 2


def test_sequential_modes_prepare_everything_before_training(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pl, "prepare_second_round_data", lambda ctx, download=True, index=True: calls.append((threading.current_thread().name, download, index)))
    ctx = start_pipeline(tmp_path / "ssd", "cpu", TINY, STAGE_OVER)
    step_load_datasets(ctx)
    assert calls == [("MainThread", True, True)] and ctx.background is None


def test_background_failure_is_raised_at_the_join():
    def boom():
        raise OSError("network down")
    bg = BackgroundPrep(boom)
    with pytest.raises(RuntimeError, match="background data preparation failed: network down"):
        bg.join()
    ok = BackgroundPrep(lambda: None)
    ok.join()
    assert ok.finished is not None and ok.error is None


def test_summary_and_step_table():
    assert [s["key"] for s in STEPS] == ["detection", "coco", "second_round", "coco_long", "long_final"]
    assert [s["test"] for s in STEPS] == [True, False, True, False, True] and [bool(s["save"]) for s in STEPS] == [True, False, True, False, True]
    assert [s["init"] for s in STEPS] == [None, None, "auto", "saved:second_round", "auto"]


def test_second_round_needs_nothing_but_the_first_coco_step(tmp_path):
    ctx = start_pipeline(tmp_path / "ssd", "cpu", TINY, STAGE_OVER)
    with pytest.raises(RuntimeError, match="saved final model of the second round"):
        ctx.cfg("coco_long")
    assert ctx.cfg("coco_long", need_init=False)["stage"] == "coco_long"
    assert pipeline_summary(ctx)["steps"][0]["saved_model"] is None and step_second_round is not None
