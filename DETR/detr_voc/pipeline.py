"""The whole job in order: load datasets -> DETR training (single phase, early stopping) -> test once -> save the model.
Inputs: a path and a device (cpu | gpu). An interrupted run resumes where it stopped."""
import json
import os
import sys
import time
from pathlib import Path

import torch

from detr_voc.config import STAGES, load_config, parse_overrides
from detr_voc.data import build_data
from detr_voc.env import setup_environment
from detr_voc.evaluate import run_test_evaluation
from detr_voc.serve import package_model
from detr_voc.train import benchmark, run_training

DEVICE_MODES = ("cpu", "gpu")
DEVICE_HELP = {"cpu": "everything on the CPU (DETR is very slow on a CPU: use it for small checks)",
               "gpu": "the model trains on the GPU; CPU workers load and augment images and the CPU solves the Hungarian matching"}
STEPS = [{"key": "detr", "title": "DETR training (single phase) -> test -> save", "test": True, "save": "detr"}]


def choose_device_mode(value=None) -> str:
    """Validate the answer, or ask for it when running in a terminal."""
    names = {"1": "cpu", "2": "gpu"}
    if value:
        v = names.get(str(value).strip(), str(value).strip().lower())
        if v not in DEVICE_MODES:
            raise ValueError(f"device must be one of {DEVICE_MODES}, got {value!r}")
        return v
    if not sys.stdin or not sys.stdin.isatty():
        raise SystemExit("Choose where to train: pass cpu or gpu as the second argument.")
    print("Where should training run?")
    for i, m in enumerate(DEVICE_MODES, 1):
        print(f"  {i}) {m:5s} {DEVICE_HELP[m]}")
    while True:
        ans = input("Enter 1 or 2: ").strip()
        if ans in names or ans in DEVICE_MODES:
            return names.get(ans, ans)


def device_overrides(mode: str, cpu_count: int | None = None) -> dict:
    n = cpu_count or os.cpu_count() or 4
    if mode == "cpu":
        return {"device": "cpu", "schedule.amp": "none", "data.num_workers": max(2, n // 4)}
    return {"device": "cuda", "schedule.amp": "bf16", "data.num_workers": min(20, max(2, n - 4))}


def check_device(mode: str) -> str:
    if mode == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("You chose 'gpu' but PyTorch sees no GPU. Choose 'cpu', or use a GPU cluster with the Machine Learning runtime.")
        return torch.cuda.get_device_name(0)
    return "cpu"


class PipelineContext:
    def __init__(self, paths, mode, stage, overrides, state, state_path):
        self.paths, self.mode, self.stage, self.overrides, self.state, self.state_path = paths, mode, stage, overrides, state, state_path

    def log(self, msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2, default=str))

    def cfg(self) -> dict:
        cfg = load_config(self.stage, {**device_overrides(self.mode), **parse_overrides(self.overrides)}, dest=self.paths["dest"])
        cfg["run_label"] = f"1_detr_{cfg['data']['dataset']}"
        return cfg


def start_pipeline(path, device_mode=None, stage="voc", overrides=None) -> PipelineContext:
    mode = choose_device_mode(device_mode)
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; choose from {sorted(STAGES)}")
    paths = setup_environment(path)
    gpu_name = check_device(mode)
    if mode == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - device_overrides("cpu")["data.num_workers"]))
    state_path = Path(paths["dest"]) / "pipeline_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"stages": {}}
    ctx = PipelineContext(paths, mode, stage, overrides, state, state_path)
    state["device_mode"], state["stage"] = mode, stage
    ctx.save()
    ctx.log(f"path: {paths['dest']} | device: {mode} ({DEVICE_HELP[mode]}) | {gpu_name} | stage: {stage}")
    if state["stages"].get("detr", {}).get("train_done"):
        ctx.log("found earlier progress, finished steps will be skipped")
    return ctx


def step_load_datasets(ctx: PipelineContext) -> dict:
    ctx.log("step 1: loading datasets")
    data = build_data(ctx.cfg(), ctx.paths)
    info = {"data_version": data["data_version"], "train": len(data["train"]), "val": len(data["val"] or []), "test": len(data["test"] or [])}
    ctx.state["datasets"] = info
    ctx.save()
    ctx.log(f"datasets ready: {info}")
    return info


def step_estimate_time(ctx: PipelineContext) -> dict:
    cfg = ctx.cfg()
    b = benchmark(cfg, ctx.paths, steps=4, loader_batches=6)
    ctx.log(f"time estimate (upper bound; early stopping usually ends sooner, evaluation is extra): {b}")
    ctx.state["estimate"] = b
    ctx.save()
    return b


def find_resumable(paths: dict, stage: str) -> Path | None:
    best, best_t = None, -1.0
    for f in Path(paths["runs"]).glob("*/stage.json"):
        try:
            meta = json.loads(f.read_text())
        except ValueError:
            continue
        last = f.parent / "checkpoints" / "last.pt"
        if meta.get("stage") == stage and meta.get("status") == "running" and last.exists() and last.stat().st_mtime > best_t:
            best, best_t = last, last.stat().st_mtime
    return best


def _supersede_unfinished(paths: dict, stage: str, keep_run: str) -> None:
    for f in Path(paths["runs"]).glob("*/stage.json"):
        meta = json.loads(f.read_text())
        if meta.get("stage") == stage and meta.get("status") == "running" and f.parent.name != keep_run:
            f.write_text(json.dumps({**meta, "status": "superseded"}))


def step_train(ctx: PipelineContext) -> dict:
    """Train (resuming an interrupted run), test once on the held-out test set, save the model folder."""
    spec, st = STEPS[0], ctx.state["stages"].setdefault("detr", {})
    cfg = ctx.cfg()
    ctx.log("step 2: " + spec["title"])
    if not st.get("train_done"):
        resume = find_resumable(ctx.paths, cfg["stage"])
        if resume is not None:
            cfg["resume_from"] = str(resume)
            ctx.log(f"resuming the interrupted run from {resume}")
        summary = run_training(cfg, ctx.paths)
        _supersede_unfinished(ctx.paths, cfg["stage"], summary["run_id"])
        st.update({"train_done": True, "run_id": summary["run_id"], "summary": summary})
        ctx.save()
    else:
        ctx.log("training already done, skipping")
    if cfg["data"].get("test_set") or cfg["data"]["dataset"] == "synthetic":
        if "test" not in st:
            st["test"] = run_test_evaluation(cfg, ctx.paths, st["run_id"])
            ctx.save()
    if not st.get("model_dir"):
        data = build_data(cfg, ctx.paths)
        example = (data["val"] or data["test"]).source.get(0)[0]
        pkg = package_model(cfg, ctx.paths, st["run_id"], example, Path(ctx.paths["dest"]) / "models" / f"{spec['save']}_{cfg['data']['dataset']}")
        if not pkg["load_verified"]:
            raise RuntimeError(f"the saved model failed its reload check: {pkg}")
        st["model_dir"] = pkg["model_dir"]
        ctx.save()
        ctx.log(f"saved model: {pkg['model_dir']}")
    return st


def pipeline_summary(ctx: PipelineContext) -> dict:
    st = ctx.state["stages"].get("detr", {})
    sm, t = st.get("summary", {}), st.get("test", {})
    row = {"stage": ctx.stage, "epochs": sm.get("epochs"), "stop_reason": sm.get("stop_reason"), "best_epoch": sm.get("best_epoch"),
           "best_val_map50": sm.get("best_val_metric"), "train_minus_val": sm.get("generalisation_gap"),
           "overfit_warning": sm.get("overfit_warning"), "test_map50": t.get("test_map50"), "saved_model": st.get("model_dir")}
    out = {"device_mode": ctx.mode, "dest": str(ctx.paths["dest"]), "result": row}
    (Path(ctx.paths["dest"]) / "pipeline_summary.json").write_text(json.dumps(out, indent=2, default=str))
    ctx.log("summary:")
    print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    return out


def run_pipeline(path, device_mode=None, stage="voc", overrides=None) -> dict:
    """Everything, in order. `stage` and `overrides` exist for tests and the package CLI; the script takes only path and device."""
    ctx = start_pipeline(path, device_mode, stage, overrides)
    step_load_datasets(ctx)
    step_estimate_time(ctx)
    step_train(ctx)
    return pipeline_summary(ctx)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(f"usage: python {Path(sys.argv[0]).name} PATH [cpu|gpu]\n  PATH: folder for everything (data, caches, runs, models)\n"
              "  device: where to train; asked interactively when omitted")
        return 0
    run_pipeline(argv[0], argv[1] if len(argv) > 1 else None)
    return 0
