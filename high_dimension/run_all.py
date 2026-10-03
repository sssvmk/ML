#!/usr/bin/env python
"""
run_all.py - main orchestrator for the Chapter 18 study (p >> N): regression on Riboflavin, classification on SRBCT.

    python run_all.py --task regression     --data riboflavin.csv --results results_ribo
    python run_all.py --task classification --data ISLP           --results results_srbct      # or --data srbct.csv

[1] EDA  [2] split + preparation bundle  [3] the methods tuned (repeated CV) / validated / tested IN PARALLEL with live progress  [4] comparison + winner
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_os.cpu_count() or 1))   # Windows: stops joblib/loky from calling the removed `wmic` tool
try:                                                         # loky asks `wmic` for the physical core count on Windows (removed from Windows 11): answer it ourselves, silently
    from joblib.externals.loky.backend import context as _loky_ctx
    _loky_ctx._count_physical_cores = lambda _n=(_os.cpu_count() or 1): (_n, None)
except Exception:
    pass

import argparse
import importlib
import multiprocessing as mp
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "methods"))

METHODS = {"regression": [("fused_lasso", "reg_fused_lasso"), ("kernel_ridge", "reg_kernel_ridge"), ("lasso", "reg_lasso"), ("supervised_pca", "reg_supervised_pca"), ("ridge", "reg_ridge"), ("mean", "reg_mean")],
           "classification": [("l1_logreg", "clf_l1_logreg"), ("supervised_pca", "clf_supervised_pca"), ("nsc", "clf_nsc"), ("reg_logreg", "clf_reg_logreg"), ("reg_lda", "clf_reg_lda"), ("linear_svc", "clf_linear_svc"),
                              ("diag_lda", "clf_diag_lda"), ("majority", "clf_majority")]}


def _worker(name, module, bundle_dir, out_dir, cfg_dict, q, blas):
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(blas)
    except Exception:
        pass
    import hd_lib
    cfg = hd_lib.Cfg(**{**cfg_dict, "n_jobs": blas})
    prog = hd_lib.Progress(name, lambda n, d, t, m: q.put(("tick", n, d, t, m)))
    try:
        s = importlib.import_module(module).run(bundle_dir, out_dir, cfg, prog)
        t = s["test"]
        q.put(("done", name, t["mse"] if "mse" in t else t["error"], 0, "MSE" if "mse" in t else "error"))
        return name, True
    except Exception as e:                               # noqa: BLE001
        q.put(("error", name, 0, 0, f"{type(e).__name__}: {e}"))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "ERROR.txt").write_text(traceback.format_exc())
        return name, False


def main():
    ap = argparse.ArgumentParser(description="ESLII Ch.18 p >> N: Riboflavin regression / SRBCT classification")
    ap.add_argument("--task", required=True, choices=["regression", "classification"])
    ap.add_argument("--data", default=None, help="riboflavin.csv (regression) or srbct.csv / ISLP (classification, default ISLP)")
    ap.add_argument("--results", default=None)
    ap.add_argument("--methods", nargs="*", default=None, help="subset of the task's methods")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="seed of the random 47/12/12 split and of the CV folds")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--cv-repeats", type=int, default=3)
    ap.add_argument("--winsorize", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--stability-runs", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--compare-existing", action="store_true", help="compare ALL of the task's methods that already have results in --results (use with --methods to retrain only some)")
    ap.add_argument("--skip-eda", action="store_true")
    a = ap.parse_args()

    import hd_lib, hd_data, hd_eda, hd_compare
    if a.data is None and a.task == "regression":
        raise SystemExit("--data riboflavin.csv is required for regression")
    data = a.data or "ISLP"
    res = Path(a.results or f"results_{'ribo' if a.task == 'regression' else 'srbct'}")
    res.mkdir(parents=True, exist_ok=True)
    logf = open(res / "run_all.log", "w")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} | {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    cfg = hd_lib.Cfg(seed=a.seed, cv_folds=a.cv_folds, cv_repeats=a.cv_repeats, winsorize=a.winsorize, stability_runs=a.stability_runs, alpha=a.alpha)
    t0 = time.time()
    bundle_dir = res / "prepared"
    log("[1/4] data, split and exploratory data analysis")
    b, genes, wins = hd_data.build_bundle(a.task, data, bundle_dir, cfg, None, logger=log)
    eda = None if a.skip_eda else hd_eda.run_eda(a.task, b, genes, res, logger=log)
    if eda is not None and cfg.winsorize == "auto":
        hd_data.build_bundle(a.task, data, bundle_dir, cfg, eda, logger=lambda m: None)
        log(f"      winsorising genes at the 1 / 99 % quantiles inside the pipelines: {eda['recommend_winsorize']}")
    del b
    allm = METHODS[a.task]
    names = [n for n, _ in allm]
    methods = a.methods or names
    bad = [m for m in methods if m not in names]
    if bad:
        raise SystemExit(f"methods {bad} do not apply to task '{a.task}': {names}")
    ncpu = _os.cpu_count() or 1
    workers = a.workers or min(len(methods), ncpu)
    blas = max(1, ncpu // workers)
    log(f"[2/4] training {len(methods)} {a.task} methods in parallel ({workers} processes, {blas} BLAS thread(s) each)")
    ctx = mp.get_context("spawn")
    q = ctx.Manager().Queue()
    finished, failed = {}, {}
    try:
        from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn
        use_rich = True
    except Exception:
        use_rich = False
    mod = dict(allm)
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = [ex.submit(_worker, m, mod[m], str(bundle_dir), str(res / m), cfg.__dict__, q, blas) for m in methods]
        if use_rich:
            prog = Progress(TextColumn("[bold]{task.fields[name]:<16}"), BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), TextColumn("{task.fields[msg]}"))
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
                    prog.update(tasks[name], completed=prog.tasks[tasks[name]].total, msg=f"[green]done: test {msg} {x:.5f}")
                log(f"      {name} finished: test {msg} {x:.5f}")
            else:
                failed[name] = msg
                if use_rich:
                    prog.update(tasks[name], msg=f"[red]FAILED: {msg}")
                log(f"      {name} FAILED: {msg}")
        if use_rich:
            prog.stop()
    log(f"      finished {len(finished)}/{len(methods)}; failed: {list(failed) or 'none'}")
    log("[3/4] cross-method comparison on the common test set")
    cmp_methods = [m for m in names if (res / m / "summary.json").exists()] if a.compare_existing else methods
    hd_compare.compare(res, a.task, cmp_methods, alpha=cfg.alpha, logger=log)
    log(f"[4/4] all done in {(time.time() - t0) / 60:.1f} min -> {res.resolve()}")
    logf.close()


if __name__ == "__main__":
    main()
