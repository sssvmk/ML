"""The whole job as one sequence: load datasets -> detection training (+test, save) -> second round (+save) -> longer schedule
(+save). Inputs: a path and a device mode (cpu | gpu | combination). Interrupted runs resume where they stopped."""
import json
import os
import sys
import threading
import time
from pathlib import Path

import torch

from ssd_voc.config import load_config, parse_overrides
from ssd_voc.data import build_data, coco_records
from ssd_voc.data_download import ensure_coco
from ssd_voc.env import setup_environment
from ssd_voc.evaluate import run_test_evaluation
from ssd_voc.serve import package_model
from ssd_voc.train import benchmark, run_training

DEVICE_MODES = ("cpu", "gpu", "combination")
DEVICE_HELP = {
    "cpu": "everything runs on the CPU (very slow for SSD: use it for small checks)",
    "gpu": "the model trains on the GPU; CPU workers only load and augment images; all datasets are prepared first",
    "combination": "like gpu, and the CPU also works in parallel: the COCO download for the later stages runs in the "
                   "background while the GPU trains the first stage",
}

# key, config stage, title, start from, test on VOC2007 test, saved model name
STEPS = [
    {"key": "detection", "stage": "voc", "title": "Detection training on VOC2012", "init": None, "test": True,
     "save": "01_detection_voc"},
    {"key": "coco", "stage": "coco", "title": "Second round, part 1: training on COCO", "init": None, "test": False,
     "save": None},
    {"key": "second_round", "stage": "voc_from_coco", "title": "Second round, part 2: fine-tuning on VOC2012 (final model)",
     "init": "auto", "test": True, "save": "02_final_second_round"},
    {"key": "coco_long", "stage": "coco_long", "title": "Longer schedule, part 1: COCO, starting from the saved final model",
     "init": "saved:second_round", "test": False, "save": None},
    {"key": "long_final", "stage": "voc_from_coco_long", "title": "Longer schedule, part 2: VOC2012 (final model)",
     "init": "auto", "test": True, "save": "03_final_longer_schedule"},
]
STEP = {s["key"]: s for s in STEPS}


def choose_device_mode(value=None) -> str:
    """Validate the answer, or ask for it when running in a terminal."""
    names = {"1": "cpu", "2": "gpu", "3": "combination"}
    if value:
        v = names.get(str(value).strip(), str(value).strip().lower())
        if v not in DEVICE_MODES:
            raise ValueError(f"device must be one of {DEVICE_MODES}, got {value!r}")
        return v
    if not sys.stdin or not sys.stdin.isatty():
        raise SystemExit("Choose where to train: pass cpu, gpu or combination as the second argument.")
    print("Where should training run?")
    for i, m in enumerate(DEVICE_MODES, 1):
        print(f"  {i}) {m:12s} {DEVICE_HELP[m]}")
    while True:
        ans = input("Enter 1, 2 or 3: ").strip()
        if ans in names or ans in DEVICE_MODES:
            return names.get(ans, ans)


def device_overrides(mode: str, cpu_count: int | None = None) -> dict:
    n = cpu_count or os.cpu_count() or 4
    if mode == "cpu":
        workers = max(2, n // 4)
        return {"device": "cpu", "schedule.amp": "none", "schedule.channels_last": False, "data.num_workers": workers}
    return {"device": "cuda", "schedule.amp": "bf16", "schedule.channels_last": True,
            "data.num_workers": min(20, max(2, n - 4))}


def check_device(mode: str) -> str:
    if mode in ("gpu", "combination"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"You chose '{mode}' but PyTorch sees no GPU. Choose 'cpu', or use a GPU cluster with the "
                               "Machine Learning runtime.")
        return torch.cuda.get_device_name(0)
    return "cpu"


class PipelineContext:
    def __init__(self, paths, mode, overrides, stage_overrides, state, state_path):
        self.paths, self.mode, self.overrides = paths, mode, overrides
        self.stage_overrides, self.state, self.state_path = stage_overrides or {}, state, state_path
        self.background = None

    def log(self, msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2, default=str))

    def cfg(self, key: str, need_init: bool = True) -> dict:
        spec = STEP[key]
        over = {**device_overrides(self.mode), **parse_overrides(self.overrides), **parse_overrides(self.stage_overrides.get(key))}
        if spec["init"] == "saved:second_round":
            model_dir = self.state["stages"].get("second_round", {}).get("model_dir")
            if model_dir:
                over["init_from"] = str(Path(model_dir) / "checkpoint.pt")
            elif need_init:
                raise RuntimeError("the longer schedule starts from the saved final model of the second round, which is missing")
        cfg = load_config(spec["stage"], over, dest=self.paths["dest"])
        cfg["run_label"] = f"{[s['key'] for s in STEPS].index(key) + 1}_{key}"
        return cfg


def start_pipeline(path, device_mode=None, overrides=None, stage_overrides=None) -> PipelineContext:
    """Create the folder layout under `path`, route every cache there, check the device, load (or start) the state file."""
    mode = choose_device_mode(device_mode)
    paths = setup_environment(path)
    gpu_name = check_device(mode)
    if mode == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - device_overrides("cpu")["data.num_workers"]))
    state_path = Path(paths["dest"]) / "pipeline_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"stages": {}}
    ctx = PipelineContext(paths, mode, overrides, stage_overrides, state, state_path)
    state["device_mode"] = mode
    ctx.save()
    ctx.log(f"path: {paths['dest']} | device mode: {mode} ({DEVICE_HELP[mode]}) | {gpu_name}")
    done = [k for k, v in state["stages"].items() if v.get("train_done")]
    if done:
        ctx.log(f"found earlier progress, will skip finished steps: {done}")
    return ctx


# ------------------------------------------------------------------ step 1: datasets
def prepare_second_round_data(ctx: PipelineContext, download: bool = True, index: bool = True) -> None:
    cfg = ctx.cfg("coco")
    if cfg["data"]["dataset"] != "coco2017":
        return
    d = cfg["data"]
    if download:
        ensure_coco(ctx.paths["data"], ctx.paths["tmp"], ("val", "train"), d.get("keep_archives", False), d.get("download", True))
    if index:
        for split in ("val", "train"):
            coco_records(ctx.paths["data"], split)


class BackgroundPrep:
    """Runs a function in a thread (the CPU side of 'combination' mode) and re-raises its error at join()."""

    def __init__(self, fn, *args, **kwargs):
        self.fn, self.args, self.kwargs, self.error, self.started, self.finished = fn, args, kwargs, None, time.time(), None
        self.thread = threading.Thread(target=self._run, daemon=True, name="ssd-background-prep")
        self.thread.start()

    def _run(self):
        try:
            self.fn(*self.args, **self.kwargs)
        except BaseException as e:  # noqa: BLE001 - reported at join()
            self.error = e
        finally:
            self.finished = time.time()

    def join(self):
        self.thread.join()
        if self.error is not None:
            raise RuntimeError(f"background data preparation failed: {self.error}") from self.error


def step_load_datasets(ctx: PipelineContext) -> dict:
    """VOC2012 trainval + VOC2007 test first. COCO next (gpu/cpu), or in the background while the GPU trains (combination)."""
    ctx.log("step 1: loading datasets")
    data = build_data(ctx.cfg("detection"), ctx.paths)
    info = {"voc_data_version": data["data_version"], "voc_train": len(data["train"]), "voc_val": len(data["val"] or []),
            "voc_test": len(data["test"] or [])}
    if ctx.mode == "combination":
        ctx.background = BackgroundPrep(prepare_second_round_data, ctx, True, False)   # download only, in the background
        ctx.log("COCO is downloading in the background while the GPU trains the first stage")
    else:
        prepare_second_round_data(ctx)
    ctx.state["datasets"] = info
    ctx.save()
    ctx.log(f"VOC ready: {info}")
    return info


def step_estimate_time(ctx: PipelineContext) -> dict:
    """Measure this machine for a moment and print the longest the whole pipeline could take."""
    cfg = ctx.cfg("detection")
    b = benchmark(cfg, ctx.paths, steps=5, loader_batches=6)
    per_it = 1 / min(1 / b["model_step_s"], b["augmentation_img_per_s"] / cfg["data"]["batch_size"]) if b["augmentation_img_per_s"] == b["augmentation_img_per_s"] else b["model_step_s"]
    total = sum(round(p["iters"] * c["schedule"]["scale"]) for k in STEP for c in [ctx.cfg(k, need_init=False)] for p in c["schedule"]["phases"])
    est = {"seconds_per_iteration": round(per_it, 3), "bottleneck": b["bottleneck"], "max_iterations_all_steps": total,
           "max_hours_all_steps": round(total * per_it / 3600, 1)}
    ctx.log(f"time estimate (upper bound, early stopping usually ends steps sooner, evaluation is extra): {est}")
    ctx.state["estimate"] = est
    ctx.save()
    return est


# ------------------------------------------------------------------ steps 2-4
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


def _example_image(ctx: PipelineContext, cfg: dict):
    data = build_data(cfg, ctx.paths)
    ds = data["val"] or data["test"]
    return ds.source.get(0)[0]


def run_step(ctx: PipelineContext, key: str) -> dict:
    """Train one stage (resuming it if it was interrupted), then test and save it where the plan says so."""
    spec, st = STEP[key], ctx.state["stages"].setdefault(key, {})
    cfg = ctx.cfg(key)
    ctx.log(f"{spec['title']}  [stage {spec['stage']}]")
    if key == "coco" and ctx.background is not None:
        ctx.log("waiting for the background COCO download to finish ...")
        ctx.background.join()
        ctx.background = None
    if key == "coco":
        prepare_second_round_data(ctx)      # builds the compact COCO index if it does not exist yet
    if not st.get("train_done"):
        resume = find_resumable(ctx.paths, spec["stage"])
        if resume is not None:
            cfg["resume_from"] = str(resume)
            ctx.log(f"resuming the interrupted run from {resume}")
        summary = run_training(cfg, ctx.paths)
        _supersede_unfinished(ctx.paths, spec["stage"], summary["run_id"])
        st.update({"train_done": True, "run_id": summary["run_id"], "summary": summary})
        ctx.save()
    else:
        ctx.log("training already done, skipping")
    if spec["test"] and "test" not in st:
        st["test"] = run_test_evaluation(cfg, ctx.paths, st["run_id"])
        ctx.save()
    if spec["save"] and not st.get("model_dir"):
        pkg = package_model(cfg, ctx.paths, st["run_id"], _example_image(ctx, cfg), Path(ctx.paths["dest"]) / "models" / spec["save"])
        if not pkg["load_verified"]:
            raise RuntimeError(f"the saved model {spec['save']} failed its reload check: {pkg}")
        st["model_dir"] = pkg["model_dir"]
        ctx.save()
        ctx.log(f"saved model: {pkg['model_dir']}")
    return st


def step_detection(ctx: PipelineContext) -> dict:
    ctx.log("step 2: detection training -> test -> save")
    return run_step(ctx, "detection")


def step_second_round(ctx: PipelineContext) -> dict:
    ctx.log("step 3: optional second round (COCO, then fine-tune on VOC2012) -> save final model")
    run_step(ctx, "coco")
    return run_step(ctx, "second_round")


def step_longer_schedule(ctx: PipelineContext) -> dict:
    ctx.log("step 4: longer schedules (zoom-out augmentation, 2x iterations) from the saved model, with the respective datasets")
    run_step(ctx, "coco_long")
    return run_step(ctx, "long_final")


def pipeline_summary(ctx: PipelineContext) -> dict:
    rows = []
    for spec in STEPS:
        st = ctx.state["stages"].get(spec["key"], {})
        sm = st.get("summary", {})
        t = st.get("test", {})
        rows.append({"step": spec["key"], "stage": spec["stage"], "iterations": sm.get("iterations"),
                     "stop_reason": sm.get("stop_reason"), "best_val_map50": sm.get("best_val_metric"),
                     "train_minus_val": sm.get("generalisation_gap"), "overfit_warning": sm.get("overfit_warning"),
                     "test_map50": t.get("test_map50"), "saved_model": st.get("model_dir")})
    out = {"device_mode": ctx.mode, "dest": str(ctx.paths["dest"]), "steps": rows}
    (Path(ctx.paths["dest"]) / "pipeline_summary.json").write_text(json.dumps(out, indent=2, default=str))
    ctx.log("summary (choose the final model by VALIDATION mAP; the test set is for reporting):")
    for r in rows:
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    return out


def run_pipeline(path, device_mode=None, overrides=None, stage_overrides=None) -> dict:
    """Everything, in order. `overrides` / `stage_overrides` exist for tests; the notebook and script never pass them."""
    ctx = start_pipeline(path, device_mode, overrides, stage_overrides)
    step_load_datasets(ctx)
    step_estimate_time(ctx)
    step_detection(ctx)
    step_second_round(ctx)
    step_longer_schedule(ctx)
    return pipeline_summary(ctx)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(f"usage: python {Path(sys.argv[0]).name} PATH [cpu|gpu|combination]\n  PATH: folder for everything (data, caches, runs, models)\n"
              "  device: where to train; asked interactively when omitted")
        return 0
    run_pipeline(argv[0], argv[1] if len(argv) > 1 else None)
    return 0
