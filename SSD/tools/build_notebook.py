#!/usr/bin/env python3
"""Generate the single Databricks notebook AND the single-file Python script from the package (one source of truth).

  python tools/build_notebook.py            # writes notebooks/ssd_pipeline_databricks.{py,ipynb} and dist/ssd_pipeline.py
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULES = ["config", "env", "data_download", "boxes", "data", "model", "metrics", "ema", "schedule", "curves",
           "diagnose", "serve", "train", "evaluate", "pipeline"]
SEP = "# COMMAND ----------"
HEADER = "# Databricks notebook source"
NB_BASE = ROOT / "notebooks" / "ssd_pipeline_databricks"
PY_OUT = ROOT / "dist" / "ssd_pipeline.py"


def strip_package_imports(src: str) -> str:
    """Remove `from ssd_voc... import ...` (everything lives in one namespace) and refuse nested package imports."""
    tree = ast.parse(src)
    lines = src.splitlines()
    drop = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("ssd_voc"):
            drop.update(range(node.lineno - 1, node.end_lineno))
    out = "\n".join(ln for i, ln in enumerate(lines) if i not in drop).strip("\n") + "\n"
    for node in ast.walk(ast.parse(out)):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and "ssd_voc" in (getattr(node, "module", None) or
                                                                           " ".join(a.name for a in node.names)):
            raise SystemExit(f"nested package import on line {node.lineno}: move it to the top of the module")
    return out


def library_cells() -> list[tuple[str, str]]:
    seen, cells = {}, []
    for m in MODULES:
        src = strip_package_imports((ROOT / "ssd_voc" / f"{m}.py").read_text())
        for node in ast.parse(src).body:
            names = [node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else \
                [t.id for a in [node] if isinstance(a, ast.Assign) for t in a.targets if isinstance(t, ast.Name)]
            for n in names:
                if n in seen and n != "log":
                    raise SystemExit(f"name clash in notebook build: {n} defined in {seen[n]} and {m}")
                seen[n] = m
        cells.append((m, src))
    return cells


def md(text: str) -> str:
    return "\n".join("# MAGIC " + ln if ln else "# MAGIC" for ln in ("%md\n" + text).splitlines())


GUIDE = """# SSD object detector (ResNet-50), end to end: datasets, training, test, save, second round, longer schedule

One notebook, run top to bottom ("Run all"). **Two inputs only**: `path` (a folder for everything) and `device` (where to train).

| `device` | What it does |
|---|---|
| `cpu` | everything on the CPU (SSD is very slow on a CPU: use it for small checks) |
| `gpu` | the model trains on the GPU; CPU workers load and augment images; all datasets are prepared first |
| `combination` | like `gpu`, and the CPU also works in parallel: the COCO download for the later steps runs in the background while the GPU trains the first stage |

**The sequence** (every step trains with early stopping, weight decay, augmentation and EMA to limit overfitting)
1. load datasets: VOC2012 (train/validation), VOC2007 test, COCO
2. detection training on VOC2012, then test on VOC2007 test, then save the model to `path/models/01_detection_voc`
3. second round: train on COCO, fine-tune on VOC2012, test, save the final model to `path/models/02_final_second_round`
4. longer schedules (zoom-out augmentation, twice the iterations): start from the saved final model, COCO then VOC2012, test, save to `path/models/03_final_longer_schedule`
5. summary

Everything is stored under `path`: data, caches, temporary files, checkpoints, MLflow store, models. If the run is interrupted (cluster restart, timeout), run the notebook again with the same `path`: finished steps are skipped and an interrupted training resumes.
The whole pipeline can take many hours or days on a GPU; step 1 prints an estimate. Use the **Machine Learning GPU runtime**."""


def notebook_cells() -> list[tuple[str, str]]:
    cells = [("md", md(GUIDE)), ("code", """# Environment check (torch, torchvision, mlflow, GPU).
import importlib
for mod in ("torch", "torchvision", "mlflow", "PIL", "yaml", "requests", "pandas", "matplotlib"):
    try:
        m = importlib.import_module(mod)
        print(mod, getattr(m, "__version__", "ok"))
    except ImportError:
        raise SystemExit(f"{mod} is missing: use a Databricks 'Machine Learning' runtime (16.4 LTS ML GPU)")
    except Exception as e:
        raise SystemExit(f"importing {mod} failed: {type(e).__name__}: {e}\\nIf this mentions numpy or scipy, a cluster library "
                         "installed a scipy that needs numpy 2 while the runtime has numpy 1.x: uninstall that library from the "
                         "cluster (Libraries tab), or run %pip install -q 'scipy<1.18' 'numpy<2' and restart Python.")
import torch
print("CUDA available:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")"""),
             ("code", """# The only two inputs. (Outside Databricks the defaults below are used.)
_defaults = {"path": "./ssd_run", "device": "gpu"}
try:
    dbutils.widgets.text("path", "/Volumes/main/default/ssd", "Path: folder for data, caches, models and logs")
    dbutils.widgets.dropdown("device", "gpu", ["cpu", "gpu", "combination"], "Train on: cpu, gpu or combination")
    params = {k: dbutils.widgets.get(k) for k in _defaults}
except NameError:
    params = dict(_defaults)
print(params)""")]
    for m, src in library_cells():
        cells.append(("code", f"# [library] {m}.py (generated from the package)\n" + src))
    cells += [
        ("md", md("## Step 0: set up\nCreates the folder layout under `path`, points every cache and temp folder there, checks the device.")),
        ("code", """import logging
import sys as _sys
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=_sys.stdout, force=True)
# `_TEST_OVERRIDES` / `_TEST_STAGE_OVERRIDES` are only defined by the automated tests; they are not parameters.
ctx = start_pipeline(params["path"], params["device"], overrides=globals().get("_TEST_OVERRIDES"),
                     stage_overrides=globals().get("_TEST_STAGE_OVERRIDES"))"""),
        ("md", md("## Step 1: load datasets\nVOC2012 trainval and VOC2007 test first; COCO next (in the background in `combination` mode). Then a short measurement of this machine gives a time estimate.")),
        ("code", """step_load_datasets(ctx)
step_estimate_time(ctx)"""),
        ("md", md("## Step 2: detection training on VOC2012, test, save\nSanity checks first, then training with early stopping; the best-validation checkpoint is tested once on VOC2007 test and saved.")),
        ("code", "step_detection(ctx)"),
        ("md", md("## Step 3: optional second round (COCO, then VOC2012), save the final model")),
        ("code", "step_second_round(ctx)"),
        ("md", md("## Step 4: longer schedules, starting from the saved final model, with the respective datasets (COCO, then VOC2012)")),
        ("code", "step_longer_schedule(ctx)"),
        ("md", md("## Step 5: summary")),
        ("code", """summary = pipeline_summary(ctx)
if on_databricks():
    dbutils.notebook.exit(json.dumps(summary, default=str))"""),
    ]
    return cells


def _lines(lines):
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def write_notebook(out_base: Path = NB_BASE) -> None:
    cells = notebook_cells()
    py = [HEADER]
    for i, (_, text) in enumerate(cells):
        if i:
            py += ["", SEP, ""]
        py.append(text)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    out_base.with_suffix(".py").write_text("\n".join(py) + "\n")
    nb_cells = []
    for i, (kind, text) in enumerate(cells):
        if kind == "md":
            body = [ln[len("# MAGIC "):] if ln.startswith("# MAGIC ") else "" for ln in text.splitlines()][1:]
            nb_cells.append({"cell_type": "markdown", "id": f"cell-{i:02d}", "metadata": {}, "source": _lines(body)})
        else:
            nb_cells.append({"cell_type": "code", "id": f"cell-{i:02d}", "metadata": {}, "execution_count": None,
                             "outputs": [], "source": _lines(text.splitlines())})
    out_base.with_suffix(".ipynb").write_text(json.dumps({
        "cells": nb_cells, "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"language_info": {"name": "python"}, "application/vnd.databricks.v1+notebook": {"language": "python"}}},
        indent=1))


def write_script(out: Path = PY_OUT) -> None:
    parts = ['#!/usr/bin/env python3\n"""SSD (Liu et al., ECCV 2016) with a ResNet-50 base network: the complete pipeline in one file.\n\n'
             "  python ssd_pipeline.py PATH [cpu|gpu|combination]\n\nPATH holds everything (data, caches, runs, models, MLflow). The device is asked for when omitted.\n"
             'Generated from the package by tools/build_notebook.py; do not edit.\n"""\n']
    for m, src in library_cells():
        parts.append(f"\n# {'=' * 100}\n# {m}.py\n# {'=' * 100}\n{src}")
    parts.append('\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts))


if __name__ == "__main__":
    write_notebook()
    write_script()
    print("written:", NB_BASE.with_suffix(".py"), NB_BASE.with_suffix(".ipynb"), PY_OUT)
