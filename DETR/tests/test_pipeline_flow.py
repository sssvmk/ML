"""The pipeline end to end (synthetic data, tiny ResNet-18 DETR, CPU) and the generated single-file script."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from smoke_over import SMOKE

import detr_voc.train as tr
from detr_voc.pipeline import (check_device, choose_device_mode, device_overrides, find_resumable, pipeline_summary, run_pipeline,
                               start_pipeline, step_load_datasets, step_train)

ROOT = Path(__file__).resolve().parent.parent
BASE = [o for o in SMOKE if not o.startswith("schedule.phases")]
TINY = [*BASE, 'schedule.phases=[{"lr":0.0003,"epochs":2},{"lr":0.00003,"epochs":1}]']


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
    path = tmp_path_factory.mktemp("pipeline") / "detr"
    summary = run_pipeline(path, "cpu", "voc", TINY)
    yield path, summary
    os.environ.clear()
    os.environ.update(saved[0])
    tempfile.tempdir = saved[1]


def test_pipeline_trains_tests_once_and_saves_a_verified_model(full_run):
    path, summary = full_run
    r = summary["result"]
    assert r["epochs"] == 3 and r["stop_reason"] == "schedule_complete" and r["test_map50"] is not None
    assert r["train_minus_val"] is not None and r["overfit_warning"] in (True, False) and r["best_val_map50"] is not None
    model_dir = Path(r["saved_model"])
    assert model_dir.name == "detr_synthetic" and (model_dir / "checkpoint.pt").exists() and (model_dir / "detr_pyfunc.py").exists()
    state = json.loads((path / "pipeline_state.json").read_text())
    st = state["stages"]["detr"]
    assert state["device_mode"] == "cpu" and st["train_done"] and st["test"]["evaluation_count"] == 1 and st["model_dir"] == str(model_dir)
    assert (path / "pipeline_summary.json").exists() and (path / "checkpoints" / "voc_best.pt").exists()


def test_everything_stays_under_the_path(full_run):
    path, _ = full_run
    assert Path(tempfile.gettempdir()).resolve() == (path / "tmp").resolve()
    assert not [p for p in path.parent.iterdir() if p.name != "detr"]                        # nothing next to `path`
    for run_dir in (path / "runs").iterdir():
        assert (run_dir / "sanity.json").exists() and (run_dir / "diagnosis.json").exists() and (run_dir / "next_steps.json").exists()


def test_running_again_with_the_same_path_skips_all_work(full_run):
    path, summary = full_run
    n_runs = len(list((path / "runs").iterdir()))
    out = run_pipeline(path, "cpu", "voc", TINY)
    assert len(list((path / "runs").iterdir())) == n_runs == 1 and out["result"]["saved_model"] == summary["result"]["saved_model"]


def test_an_interrupted_training_resumes_instead_of_restarting(tmp_path, monkeypatch):
    path = tmp_path / "detr"
    over = [*BASE, 'schedule.phases=[{"lr":0.0003,"epochs":4}]', "schedule.early_stopping.enabled=false"]
    ctx = start_pipeline(path, "cpu", "voc", over)
    step_load_datasets(ctx)
    real, calls = tr.evaluate_model, {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 5:                           # validation + train-subset pass of epochs 1 and 2 are 4 calls; the 5th is epoch 3
            raise RuntimeError("cluster went away")
        return real(*a, **k)
    monkeypatch.setattr(tr, "evaluate_model", flaky)
    with pytest.raises(RuntimeError, match="cluster went away"):
        step_train(ctx)
    last = find_resumable(ctx.paths, "voc")
    assert last is not None and last.exists()
    monkeypatch.setattr(tr, "evaluate_model", real)
    ctx2 = start_pipeline(path, "cpu", "voc", over)         # "run it again with the same path"
    st = step_train(ctx2)
    assert st["train_done"] and st["summary"]["epochs"] == 4
    import csv
    rows = list(csv.DictReader(open(Path(st["summary"]["out_dir"]) / "metrics.csv")))
    assert [int(r["epoch"]) for r in rows] == [1, 2, 3, 4]                          # history before the failure was carried over
    assert json.loads((last.parent.parent / "stage.json").read_text())["status"] == "superseded" and find_resumable(ctx2.paths, "voc") is None
    assert pipeline_summary(ctx2)["result"]["saved_model"] == st["model_dir"]


def test_device_question_and_answers(monkeypatch, tmp_path):
    assert [choose_device_mode(v) for v in ("cpu", "GPU", "1", "2")] == ["cpu", "gpu", "cpu", "gpu"]
    with pytest.raises(ValueError, match="device must be"):
        choose_device_mode("tpu")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit, match="cpu or gpu"):
        choose_device_mode(None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    answers = iter(["9", "", "2"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert choose_device_mode(None) == "gpu"                                            # invalid answers are asked again
    cpu, gpu = device_overrides("cpu", 24), device_overrides("gpu", 24)
    assert cpu["device"] == "cpu" and cpu["schedule.amp"] == "none" and cpu["data.num_workers"] == 6
    assert gpu["device"] == "cuda" and gpu["schedule.amp"] == "bf16" and gpu["data.num_workers"] == 20 and device_overrides("gpu", 6)["data.num_workers"] == 2
    assert check_device("cpu") == "cpu"
    with pytest.raises(RuntimeError, match="sees no GPU"):
        check_device("gpu")
    with pytest.raises(RuntimeError, match="Machine Learning"):
        start_pipeline(tmp_path / "x", "gpu")
    with pytest.raises(ValueError, match="unknown stage"):
        start_pipeline(tmp_path / "y", "cpu", "nope")


# ------------------------------------------------------------------ the generated single-file script
def load_script():
    spec = importlib.util.spec_from_file_location("detr_pipeline_script", ROOT / "dist" / "detr_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_checked_in_script_is_up_to_date(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    import build_script
    build_script.write_script(tmp_path / "s.py")
    assert (tmp_path / "s.py").read_text() == (ROOT / "dist" / "detr_pipeline.py").read_text(), "regenerate: python tools/build_script.py"


def test_single_file_script_cli_asks_and_never_guesses(tmp_path):
    mod = load_script()
    assert mod.main(["--help"]) == 0
    r = subprocess.run([sys.executable, str(ROOT / "dist" / "detr_pipeline.py")], capture_output=True, text=True)
    assert r.returncode == 0 and "PATH [cpu|gpu]" in r.stdout
    r = subprocess.run([sys.executable, str(ROOT / "dist" / "detr_pipeline.py"), str(tmp_path / "d")], capture_output=True, text=True,
                       stdin=subprocess.DEVNULL)
    assert r.returncode != 0 and "cpu or gpu" in (r.stderr + r.stdout)
    with pytest.raises(ValueError, match="device must be"):
        mod.main([str(tmp_path / "d"), "tpu"])


def test_single_file_script_runs_the_pipeline_and_its_saved_model_reloads(tmp_path):
    mod = load_script()
    out = mod.run_pipeline(tmp_path / "d", "cpu", "voc", [*BASE, 'schedule.phases=[{"lr":0.0003,"epochs":1}]'])
    r = out["result"]
    assert r["epochs"] == 1 and r["test_map50"] is not None and (Path(r["saved_model"]) / "detr_pyfunc.py").exists()
    # the saved pyfunc imports the SCRIPT as its code module (single-file mode) and was verified in a fresh process
    assert "from detr_pipeline import Predictor" in (Path(r["saved_model"]) / "detr_pyfunc.py").read_text()
