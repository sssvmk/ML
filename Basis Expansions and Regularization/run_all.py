#!/usr/bin/env python
"""
run_all.py - main orchestrator.

    python run_all.py --data /path/to/zillow.csv --results results [--predictor auto|yearbuilt|taxvaluedollarcnt|...]

[1] EDA  [2] temporal split + target/predictor preparation  [3] 8 methods trained / validated / tested IN PARALLEL with live progress
[4] cross-method comparison (test MSE +- SE, Diebold-Mariano / paired t, Holm) + winner
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "methods"))

METHODS = ["thin_plate_splines", "rkhs", "smoothing_splines", "wavelet_smoothing", "nonparametric_logistic",
           "regression_splines", "natural_cubic_splines", "b_splines"]


def _worker(name, prep_dir, out_dir, cfg_dict, q, blas):
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(blas)
    except Exception:
        pass
    import common
    cfg = common.RunConfig(**cfg_dict)
    prog = common.Progress(name, lambda n, d, t, m: q.put(("tick", n, d, t, m)))
    try:
        s = importlib.import_module(name).run(prep_dir, out_dir, cfg, prog)
        t = s["test"]
        q.put(("done", name, t.get("mse", t.get("log_loss")), 0, "mse" if "mse" in t else "log-loss"))
        return name, True
    except Exception as e:                               # noqa: BLE001
        q.put(("error", name, 0, 0, f"{type(e).__name__}: {e}"))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "ERROR.txt").write_text(traceback.format_exc())
        return name, False


def main():
    ap = argparse.ArgumentParser(description="Zillow logerror: ESLII Ch.5 basis-expansion smoothers compared on a temporal test set")
    ap.add_argument("--data", required=True, help="path to zillow.csv")
    ap.add_argument("--results", default="results")
    ap.add_argument("--methods", nargs="*", default=METHODS, choices=METHODS)
    ap.add_argument("--predictor", default="auto", help="predictor for the 1-D smoothers (default: best on train CV)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--cv-folds", type=int, default=10)
    ap.add_argument("--cv-scheme", choices=["blocked", "random"], default="blocked")
    ap.add_argument("--one-se", choices=["paired", "plain"], default="paired", help="one-SE rule on paired fold differences (default) or the plain ESLII SE")
    ap.add_argument("--refit", choices=["train", "train_val"], default="train_val")
    ap.add_argument("--wavelet-sim-reps", type=int, default=50)
    ap.add_argument("--skip-eda", action="store_true")
    ap.add_argument("--reuse-prepared", action="store_true")
    a = ap.parse_args()

    import common, compare, prepare
    res = Path(a.results); res.mkdir(parents=True, exist_ok=True)
    logf = open(res / "run_all.log", "w")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} | {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    cfg = common.RunConfig(cv_folds=a.cv_folds, cv_scheme=a.cv_scheme, refit=a.refit, one_se_mode=a.one_se, predictor=a.predictor, wavelet_sim_reps=a.wavelet_sim_reps)
    t0 = time.time()
    if not a.skip_eda:
        log("[1/4] exploratory data analysis")
        import eda
        eda.run_eda(a.data, res, logger=log)
    prep_dir = res / "prepared"
    if a.reuse_prepared and (prep_dir / "prepared_meta.json").exists():
        log("[2/4] reusing prepared data"); cfg.to_json(prep_dir / "run_config.json")
    else:
        log("[2/4] temporal split, target and predictor preparation")
        prepare.build_prepared(a.data, prep_dir, cfg, logger=log)
    methods = list(a.methods)
    ncpu = os.cpu_count() or 1
    workers = a.workers or min(len(methods), ncpu)
    blas = max(1, ncpu // workers)
    log(f"[3/4] training {len(methods)} methods in parallel ({workers} processes, {blas} BLAS thread(s) each)")
    ctx = mp.get_context("spawn")
    q = ctx.Manager().Queue()
    finished, failed = {}, {}
    try:
        from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn
        use_rich = True
    except Exception:
        use_rich = False
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = [ex.submit(_worker, m, str(prep_dir), str(res / m), cfg.__dict__, q, blas) for m in methods]
        if use_rich:
            prog = Progress(TextColumn("[bold]{task.fields[name]:<26}"), BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), TextColumn("{task.fields[msg]}"))
            tasks = {m: prog.add_task("", total=1, name=m, msg="queued") for m in methods}
            prog.start()
        while len(finished) + len(failed) < len(methods):
            try:
                kind, name, x, y, msg = q.get(timeout=0.5)
            except Exception:
                if all(f.done() for f in futs):
                    break
                continue
            if kind == "tick":
                if use_rich:
                    prog.update(tasks[name], completed=x, total=y, msg=msg)
                else:
                    print(f"  [{name}] {x}/{y} {msg}", flush=True)
            elif kind == "done":
                finished[name] = x
                if use_rich:
                    prog.update(tasks[name], completed=prog.tasks[tasks[name]].total, msg=f"[green]done: test {msg} {x:.6f}")
                log(f"      {name} finished: test {msg} {x:.6f}")
            else:
                failed[name] = msg
                if use_rich:
                    prog.update(tasks[name], msg=f"[red]FAILED: {msg}")
                log(f"      {name} FAILED: {msg}")
        if use_rich:
            prog.stop()
    log(f"      finished {len(finished)}/{len(methods)}; failed: {list(failed) or 'none'}")
    log("[4/4] cross-method comparison on the common test set")
    compare.compare(res, prep_dir, methods, alpha=cfg.alpha, logger=log)
    log(f"all done in {(time.time() - t0) / 60:.1f} min -> {res.resolve()}")
    logf.close()


if __name__ == "__main__":
    main()
