#!/usr/bin/env python
"""
run_all.py - main orchestrator.

    python run_all.py --data /path/to/zillow.csv --results results

Stages:  [1] EDA  ->  [2] temporal split + feature engineering  ->  [3] cached CV statistics
         [4] 12 linear methods trained / validated / tested IN PARALLEL with live progress  ->  [5] comparison + winner

Everything is logged under <results>/ :
    eda/                    EDA tables + plots
    prepared/               engineered arrays + feature_meta.json (+ run_config.json)
    <method>/               cv_curve.csv, cv_curve.png, validation_metrics.json, test_metrics.json, summary.json,
                            coefficients.csv, test_predictions.npy, <method>.log (+ method specific plots)
    comparison.csv / comparison_test_mse.png / winner.txt / winner.json / run_all.log
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "methods"))

METHODS = ["ols", "best_subset", "backward_stepwise", "forward_stagewise", "ridge", "lasso", "lar", "pcr", "pls",
           "elastic_net", "dantzig", "grouped_lasso"]


def _worker(name, prep_dir, out_dir, cfg_dict, q, blas_threads):
    """Runs in a child process: one method end-to-end."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(blas_threads)
    except Exception:
        pass
    import common
    cfg = common.RunConfig(**{**cfg_dict, "enet_alphas": tuple(cfg_dict["enet_alphas"])})
    prog = common.Progress(name, lambda n, d, t, m: q.put(("tick", n, d, t, m)))
    try:
        mod = importlib.import_module(name)
        s = mod.run(prep_dir, out_dir, cfg, prog)
        q.put(("done", name, s["test"]["mse"], s["test"]["mse_se"], ""))
        return name, True
    except Exception as e:                                   # noqa: BLE001
        q.put(("error", name, 0, 0, f"{type(e).__name__}: {e}"))
        (Path(out_dir) / "ERROR.txt").write_text(traceback.format_exc())
        return name, False


def main():
    ap = argparse.ArgumentParser(description="Zillow logerror: 12 linear regression methods, compared on a temporal test set")
    ap.add_argument("--data", required=True, help="path to zillow.csv")
    ap.add_argument("--results", default="results")
    ap.add_argument("--methods", nargs="*", default=METHODS, choices=METHODS)
    ap.add_argument("--workers", type=int, default=0, help="parallel processes (default: min(#methods, #cpus))")
    ap.add_argument("--cv-folds", type=int, default=10)
    ap.add_argument("--cv-scheme", choices=["blocked", "random"], default="blocked",
                    help="blocked = contiguous time blocks of the (date-sorted) train set; random = shuffled K-fold")
    ap.add_argument("--refit", choices=["train", "train_val"], default="train_val")
    ap.add_argument("--subset-pool", type=int, default=20, help="best-subset candidate pool (exact search up to 22)")
    ap.add_argument("--dantzig-pool", type=int, default=30)
    ap.add_argument("--skip-eda", action="store_true")
    ap.add_argument("--reuse-prepared", action="store_true", help="reuse results/prepared if it exists")
    a = ap.parse_args()

    import common
    import features
    res = Path(a.results)
    res.mkdir(parents=True, exist_ok=True)
    logf = open(res / "run_all.log", "w")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} | {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    cfg = common.RunConfig(cv_folds=a.cv_folds, cv_scheme=a.cv_scheme, refit=a.refit,
                           subset_pool=a.subset_pool, dantzig_pool=a.dantzig_pool)
    t0 = time.time()
    if not a.skip_eda:
        log("[1/5] exploratory data analysis")
        import eda
        eda.run_eda(a.data, res, logger=log)
    else:
        log("[1/5] EDA skipped")

    prep_dir = res / "prepared"
    if a.reuse_prepared and (prep_dir / "feature_meta.json").exists():
        log("[2/5] reusing prepared features")
        cfg.to_json(prep_dir / "run_config.json")
    else:
        log("[2/5] temporal split + feature engineering")
        features.build_prepared(a.data, prep_dir, cfg, logger=log)

    log("[3/5] caching fold statistics for 10-fold CV")
    prep = common.Prepared(prep_dir)
    common.get_bundle(prep, cfg)
    log(f"      p={prep.p} features | train={prep.n_train} val={prep.n_val} test={prep.n_test}")

    methods = list(a.methods)
    import os
    ncpu = os.cpu_count() or 1
    workers = a.workers or min(len(methods), ncpu)
    blas = max(1, ncpu // workers)
    log(f"[4/5] training {len(methods)} methods in parallel ({workers} processes, {blas} BLAS thread(s) each)")

    ctx = mp.get_context("spawn")
    mgr = ctx.Manager()
    q = mgr.Queue()
    state = {m: [0, 1, "queued"] for m in methods}
    finished, failed = {}, {}

    try:
        from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn
        use_rich = True
    except Exception:
        use_rich = False

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = [ex.submit(_worker, m, str(prep_dir), str(res / m), cfg.__dict__, q, blas) for m in methods]
        if use_rich:
            prog = Progress(TextColumn("[bold]{task.fields[name]:<18}"), BarColumn(), TaskProgressColumn(),
                            TimeElapsedColumn(), TextColumn("{task.fields[msg]}"))
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
                state[name] = [x, y, msg]
                if use_rich:
                    prog.update(tasks[name], completed=x, total=y, msg=msg)
                else:
                    print(f"  [{name}] {x}/{y} {msg}", flush=True)
            elif kind == "done":
                finished[name] = (x, y)
                if use_rich:
                    t = prog.tasks[tasks[name]]
                    prog.update(tasks[name], completed=t.total, msg=f"[green]done: test MSE {x:.6f} +- {y:.6f}")
                log(f"      {name} finished: test MSE {x:.6f} +- {y:.6f}")
            else:
                failed[name] = msg
                if use_rich:
                    prog.update(tasks[name], msg=f"[red]FAILED: {msg}")
                log(f"      {name} FAILED: {msg}")
        if use_rich:
            prog.stop()
    log(f"      finished {len(finished)}/{len(methods)} methods; failed: {list(failed) or 'none'}")

    log("[5/5] cross-method comparison on the test set")
    import compare
    compare.compare(res, prep_dir, methods, logger=log)
    log(f"all done in {(time.time() - t0) / 60:.1f} min  -> results in {res.resolve()}")
    logf.close()


if __name__ == "__main__":
    main()
