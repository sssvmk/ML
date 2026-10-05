"""The generated notebook and the generated single-file script ARE the package: run them as a user would."""
import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from smoke_over import SMOKE

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import build_notebook as bn  # noqa: E402

SEP = "# COMMAND ----------"
PHASES = 'schedule.phases=[{"lr":0.01,"iters":8},{"lr":0.001,"iters":4}]'
TINY = [o for o in SMOKE if not o.startswith(("schedule.phases", "schedule.eval_every", "schedule.log_every"))] + [
    PHASES, "schedule.eval_every=4", "schedule.log_every=2", "sanity.overfit_steps=30", "mlflow.backend=sqlite"]
STAGE_OVER = {"coco": ["data.synthetic.classes=5"], "coco_long": ["data.synthetic.classes=5"]}


class NotebookExit(Exception):
    pass


class FakeDbutils:
    def __init__(self, values):
        self.values, self.declared, self.exited = values, {}, None
        outer = self

        class Widgets:
            def text(self, name, default, label=""):
                outer.declared[name] = ("text", default, None)

            def dropdown(self, name, default, choices, label=""):
                assert default in choices
                outer.declared[name] = ("dropdown", default, choices)

            def multiselect(self, name, default, choices, label=""):
                outer.declared[name] = ("multiselect", default, choices)

            def get(self, name):
                return outer.values.get(name, outer.declared[name][1])

        class Notebook:
            def exit(self, value):
                outer.exited = value
                raise NotebookExit()

        self.widgets, self.notebook = Widgets(), Notebook()


def code_cells(py_path: Path) -> list[str]:
    src = py_path.read_text()
    assert src.startswith("# Databricks notebook source")
    out = []
    for c in src[len("# Databricks notebook source"):].split(SEP):
        lines = c.strip("\n").splitlines()
        if lines and all(ln.startswith("# MAGIC") for ln in lines):
            continue
        out.append("\n".join(lines))
    return out


def run_notebook(py_path, dbutils=None, test_hooks=True):
    ns: dict = {"__name__": "notebook"}
    if dbutils is not None:
        ns["dbutils"] = dbutils
    if test_hooks:
        ns["_TEST_OVERRIDES"], ns["_TEST_STAGE_OVERRIDES"] = TINY, STAGE_OVER
    for c in code_cells(py_path):
        exec(compile(c, str(py_path), "exec"), ns)
    return ns


def load_script():
    spec = importlib.util.spec_from_file_location("ssd_pipeline_script", ROOT / "dist" / "ssd_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    saved = dict(os.environ), tempfile.tempdir
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    yield
    os.environ.clear()
    os.environ.update(saved[0])
    tempfile.tempdir = saved[1]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("nb") / "ssd_nb"
    bn.write_notebook(out)
    return out


def test_notebook_structure_is_valid_databricks_source_in_the_required_order(built):
    py = built.with_suffix(".py").read_text()
    ast.parse(py)
    assert py.startswith("# Databricks notebook source") and "from ssd_voc" not in py and "import ssd_voc" not in py
    for cell in py.split(SEP):                                            # %md cells consist ONLY of MAGIC lines
        lines = [ln for ln in cell.strip("\n").splitlines() if ln.strip() and ln != "# Databricks notebook source"]
        if any(ln.startswith("# MAGIC") for ln in lines):
            assert all(ln.startswith("# MAGIC") for ln in lines)
    start = py.index("ctx = start_pipeline(params")                      # the step cells come after the library cells
    order = [py.index(s, start) for s in ("ctx = start_pipeline(params", "step_load_datasets(ctx)\nstep_estimate_time(ctx)",
                                          "step_detection(ctx)\n", "step_second_round(ctx)\n",
                                          "step_longer_schedule(ctx)\n", "summary = pipeline_summary(ctx)")]
    assert order == sorted(order) and len(set(order)) == 6                # datasets, detection, second round, longer, summary
    assert py.count("# [library]") == len(bn.MODULES)


def test_ipynb_is_valid_and_identical_in_content(built):
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(str(built.with_suffix(".ipynb")), as_version=4)
    nbformat.validate(nb)
    assert ["".join(c["source"]) for c in nb.cells if c["cell_type"] == "code"] == code_cells(built.with_suffix(".py"))
    assert nb.cells[0]["cell_type"] == "markdown" and "device" in "".join(nb.cells[0]["source"])


def test_checked_in_notebook_and_script_are_up_to_date(built, tmp_path):
    for suffix in (".py", ".ipynb"):
        assert (bn.NB_BASE.with_suffix(suffix)).read_text() == built.with_suffix(suffix).read_text(), \
            "regenerate: python tools/build_notebook.py"
    bn.write_script(tmp_path / "s.py")
    assert (tmp_path / "s.py").read_text() == bn.PY_OUT.read_text()


def test_notebook_runs_the_whole_sequence_with_only_path_and_device(built, tmp_path):
    db = FakeDbutils({"path": str(tmp_path / "ssd"), "device": "cpu"})
    ns = run_notebook(built.with_suffix(".py"), db)
    assert set(db.declared) == {"path", "device"}                                    # no other parameter exists
    assert db.declared["device"][2] == ["cpu", "gpu", "combination"] and db.declared["path"][0] == "text"
    s = ns["summary"]
    assert [r["step"] for r in s["steps"]] == ["detection", "coco", "second_round", "coco_long", "long_final"]
    assert all(r["train_minus_val"] is not None for r in s["steps"]) and s["device_mode"] == "cpu"
    for name in ("01_detection_voc", "02_final_second_round", "03_final_longer_schedule"):
        assert (tmp_path / "ssd" / "models" / name / "torchscript.pt").exists()
    for key in ("TORCH_HOME", "MPLCONFIGDIR", "TMPDIR"):
        assert Path(os.environ[key]).resolve().is_relative_to((tmp_path / "ssd").resolve())


def test_notebook_as_a_job_returns_the_summary_and_a_rerun_skips_all_work(built, tmp_path, monkeypatch):
    path = tmp_path / "ssd"
    run_notebook(built.with_suffix(".py"), FakeDbutils({"path": str(path), "device": "cpu"}))    # first full run
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "16.4")
    db = FakeDbutils({"path": str(path), "device": "cpu"})
    with pytest.raises(NotebookExit):
        run_notebook(built.with_suffix(".py"), db)                                                # rerun: all steps skipped
    out = json.loads(db.exited)
    assert out["steps"][-1]["step"] == "long_final" and out["steps"][-1]["saved_model"]


def test_notebook_asks_for_a_gpu_it_cannot_find(built, tmp_path):
    with pytest.raises(RuntimeError, match="sees no GPU"):
        run_notebook(built.with_suffix(".py"), FakeDbutils({"path": str(tmp_path / "ssd"), "device": "gpu"}))


def test_single_file_script_has_the_same_pipeline_and_a_two_argument_cli(tmp_path):
    mod = load_script()
    assert mod.main(["--help"]) == 0 and [s["key"] for s in mod.STEPS][-1] == "long_final"
    r = subprocess.run([sys.executable, str(ROOT / "dist" / "ssd_pipeline.py")], capture_output=True, text=True)
    assert r.returncode == 0 and "PATH [cpu|gpu|combination]" in r.stdout
    r = subprocess.run([sys.executable, str(ROOT / "dist" / "ssd_pipeline.py"), str(tmp_path / "ssd")], capture_output=True,
                       text=True, stdin=subprocess.DEVNULL)
    assert r.returncode != 0 and "cpu, gpu or combination" in (r.stderr + r.stdout)        # asks; never guesses silently
    with pytest.raises(ValueError, match="device must be"):
        mod.main([str(tmp_path / "ssd"), "tpu"])


def test_single_file_script_trains_tests_and_saves_the_detection_model(tmp_path):
    mod = load_script()
    ctx = mod.start_pipeline(tmp_path / "ssd", "cpu", TINY, STAGE_OVER)
    mod.step_load_datasets(ctx)
    st = mod.step_detection(ctx)
    assert st["train_done"] and st["test"]["evaluation_count"] == 1 and (Path(st["model_dir"]) / "torchscript.pt").exists()
