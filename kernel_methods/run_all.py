#!/usr/bin/env python
"""
run_all.py - main orchestrator for the Chapter 6 kernel-method study.

    python run_all.py --task regression     --data zillow.csv           --results results_reg
    python run_all.py --task classification --data santander_train.csv  --results results_clf

[1] EDA  [2] split + feature preparation  [3] the task's 5 methods trained / validated / tested IN PARALLEL with live progress  [4] comparison + winner
regression methods    : nw_regression, local_polynomial, structured_local_regression, local_likelihood_regression, rbf_network
classification methods: nw_classification, local_logistic, kernel_density_classifier, naive_bayes, gaussian_mixture_classifier
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

METHODS = {"regression": ["structured_local_regression", "rbf_network", "local_polynomial", "local_likelihood_regression", "nw_regression"],
           "classification": ["kernel_density_classifier", "local_logistic", "gaussian_mixture_classifier", "nw_classification", "naive_bayes"]}


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
        q.put(("done", name, t["mse"] if "mse" in t else t["log_loss"], 0, "MSE" if "mse" in t else "log-loss"))
        return name, True
    except Exception as e:                               # noqa: BLE001
        q.put(("error", name, 0, 0, f"{type(e).__name__}: {e}"))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "ERROR.txt").write_text(traceback.format_exc())
        return name, False


def main():
    ap = argparse.ArgumentParser(description="ESLII Ch.6 kernel smoothing methods: regression (Zillow) or classification (Santander)")
    ap.add_argument("--task", required=True, choices=["regression", "classification"])
    ap.add_argument("--data", required=True, help="zillow.csv (regression) or santander_train.csv (classification)")
    ap.add_argument("--results", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--cv-folds", type=int, default=10)
    ap.add_argument("--tune-rows", type=int, default=30000, help="size of the tuning subsample (final model uses ALL training rows)")
    ap.add_argument("--one-se", choices=["paired", "plain"], default="paired")
    ap.add_argument("--predictor", default="auto", help="regression: single predictor for N-W / local polynomial")
    ap.add_argument("--n-top-features", type=int, default=8, help="classification: features used by N-W / local logistic / kernel density")
    ap.add_argument("--nb-compare-rows", type=int, default=60000, help="naive Bayes vs logistic/GAM comparison rows (0 = skip)")
    ap.add_argument("--skip-eda", action="store_true")
    ap.add_argument("--reuse-prepared", action="store_true")
    a = ap.parse_args()

    import common, compare
    res = Path(a.results or f"results_{'reg' if a.task == 'regression' else 'clf'}")
    res.mkdir(parents=True, exist_ok=True)
    logf = open(res / "run_all.log", "w")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} | {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    cfg = common.RunConfig(cv_folds=a.cv_folds, tune_rows=a.tune_rows, one_se_mode=a.one_se, predictor=a.predictor, n_top_features=a.n_top_features, nb_compare_rows=a.nb_compare_rows)
    t0 = time.time()
    prep_dir = res / "prepared"
    methods = a.methods or METHODS[a.task]
    bad = [m for m in methods if m not in METHODS[a.task]]
    if bad:
        raise SystemExit(f"methods {bad} do not belong to task '{a.task}': {METHODS[a.task]}")
    df = None
    if a.task == "classification":
        import pandas as pd
        df = pd.read_csv(a.data)
    if not a.skip_eda and not (a.reuse_prepared and (prep_dir / "prepared_meta.json").exists()):
        log("[1/4] exploratory data analysis")
        if a.task == "regression":
            import eda_reg
            eda_reg.run_eda(a.data, res, logger=log)
        else:
            import eda_clf
            eda_clf.run_eda(df, res, seed=cfg.seed, logger=log)
    if a.reuse_prepared and (prep_dir / "prepared_meta.json").exists():
        log("[2/4] reusing prepared data"); cfg.to_json(prep_dir / "run_config.json")
    else:
        log("[2/4] split + feature preparation")
        if a.task == "regression":
            import prep_reg
            prep_reg.build_prepared(a.data, prep_dir, cfg, logger=log)
        else:
            import prep_clf
            prep_clf.build_prepared(df, prep_dir, cfg, logger=log)
    del df
    ncpu = os.cpu_count() or 1
    workers = a.workers or min(len(methods), ncpu)
    blas = max(1, ncpu // workers)
    log(f"[3/4] training {len(methods)} {a.task} methods in parallel ({workers} processes, {blas} BLAS thread(s) each)")
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
            prog = Progress(TextColumn("[bold]{task.fields[name]:<30}"), BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), TextColumn("{task.fields[msg]}"))
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
    compare.compare(res, prep_dir, a.task, methods, alpha=cfg.alpha, logger=log)
    log(f"all done in {(time.time() - t0) / 60:.1f} min -> {res.resolve()}")
    logf.close()


if __name__ == "__main__":
    main()
