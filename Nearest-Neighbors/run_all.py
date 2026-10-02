#!/usr/bin/env python
"""
run_all.py - main orchestrator for the Chapter 13 study (prototype methods and nearest neighbours).

    python run_all.py --task regression     --data zillow.csv           --results results_reg
    python run_all.py --task classification --data santander_train.csv  --results results_clf

[1] EDA  [2] split + FULL feature engineering (the candidate-variable pool)  [3] variable ranking inside every CV fold
[4] the methods trained / tuned (random-search CV over the hyper-parameters and the number of kept variables) / validated / tested IN PARALLEL with live progress  [5] comparison + winner
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

METHODS = {"regression": ["knn_reg_distance_euclidean", "knn_reg_distance_manhattan", "knn_reg_uniform_euclidean", "knn_reg_uniform_manhattan"],
           "classification": ["dann", "lvq", "gmm", "knn", "tangent", "kmeans"]}
MODULE = {"kmeans": "kmeans_prototypes", "tangent": "tangent_distance"}          # module names must not shadow other modules


def _worker(name, prep_dir, out_dir, cfg_dict, q, blas):
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(blas)
    except Exception:
        pass
    import common
    cfg = common.RunConfig(**{**cfg_dict, "n_jobs": blas})
    prog = common.Progress(name, lambda n, d, t, m: q.put(("tick", n, d, t, m)))
    try:
        s = importlib.import_module(MODULE.get(name, name)).run(prep_dir, out_dir, cfg, prog)
        t = s["test"]
        q.put(("done", name, t["mse"] if "mse" in t else t["auc"], 0, "MSE" if "mse" in t else "AUC"))
        return name, True
    except Exception as e:                               # noqa: BLE001
        q.put(("error", name, 0, 0, f"{type(e).__name__}: {e}"))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "ERROR.txt").write_text(traceback.format_exc())
        return name, False


def main():
    ap = argparse.ArgumentParser(description="ESLII Ch.13 prototype methods and nearest neighbours: regression (Zillow) or classification (Santander)")
    ap.add_argument("--task", required=True, choices=["regression", "classification"])
    ap.add_argument("--data", required=True, help="zillow.csv (regression) or santander_train.csv (classification)")
    ap.add_argument("--results", default=None)
    ap.add_argument("--methods", nargs="*", default=None, help="subset of the task's methods")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--tune-rows", type=int, default=20000, help="size of the tuning subsample (the final model uses ALL training rows)")
    ap.add_argument("--rank-trees", type=int, default=60, help="trees of the ExtraTrees importance used in the variable ranking")
    ap.add_argument("--one-se", choices=["paired", "plain"], default="paired")
    ap.add_argument("--winner-metric", choices=["auc", "log_loss"], default="auc", help="classification winner criterion")
    ap.add_argument("--leak-auc", type=float, default=0.9, help="classification: exclude features whose univariate train AUC >= this (target-leakage guard; 0 = off)")
    ap.add_argument("--n-configs-nn", type=int, default=16, help="random-search budget for LVQ / Gaussian mixtures / DANN (K-means and k-NN grids are searched fully)")
    ap.add_argument("--skip-eda", action="store_true")
    ap.add_argument("--reuse-prepared", action="store_true", help="reuse prepared data AND variable rankings if they exist")
    a = ap.parse_args()

    import common, compare, selection
    res = Path(a.results or f"results_{'reg' if a.task == 'regression' else 'clf'}")
    res.mkdir(parents=True, exist_ok=True)
    logf = open(res / "run_all.log", "w")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} | {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    cfg = common.RunConfig(cv_folds=a.cv_folds, tune_rows=a.tune_rows, one_se_mode=a.one_se, rank_trees=a.rank_trees, n_configs_nn=a.n_configs_nn, leak_auc=a.leak_auc)
    t0 = time.time()
    prep_dir = res / "prepared"
    ncpu = _os.cpu_count() or 1
    reuse = a.reuse_prepared and (prep_dir / "prepared_meta.json").exists() and (prep_dir / "rankings.npz").exists()
    df = None
    if a.task == "classification" and not reuse:
        import pandas as pd
        df = pd.read_csv(a.data)
    if reuse:
        log("[1-3/5] reusing prepared data and variable rankings"); cfg.to_json(prep_dir / "run_config.json")
    else:
        if not a.skip_eda:
            log("[1/5] exploratory data analysis")
            if a.task == "regression":
                import eda_reg
                eda_reg.run_eda(a.data, res, logger=log)
            else:
                import eda_clf
                eda_clf.run_eda(df, res, seed=cfg.seed, logger=log)
        log("[2/5] split + full feature engineering")
        if a.task == "regression":
            import prep_reg
            prep_reg.build_prepared(a.data, prep_dir, cfg, logger=log)
        else:
            import prep_clf
            prep_clf.build_prepared(df, prep_dir, cfg, logger=log)
        del df
        log("[3/5] variable ranking inside each CV fold (leakage-free selection of the top-k variables)")
        selection.build_rankings(prep_dir, cfg, workers=min(a.cv_folds + 1, ncpu), logger=log)
    methods = list(a.methods or METHODS[a.task])
    bad = [m for m in methods if m not in METHODS[a.task]]
    if bad:
        raise SystemExit(f"methods {bad} do not apply to task '{a.task}' (k-NN regression is regression-only; K-means / LVQ / Gaussian mixtures / k-NN / tangent distance / DANN are classification-only): {METHODS[a.task]}")
    workers = a.workers or min(len(methods), ncpu)
    blas = max(1, ncpu // workers)
    log(f"[4/5] training {len(methods)} {a.task} methods in parallel ({workers} processes, {blas} BLAS thread(s) each)")
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
            prog = Progress(TextColumn("[bold]{task.fields[name]:<12}"), BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), TextColumn("{task.fields[msg]}"))
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
    log("[5/5] cross-method comparison on the common test set")
    compare.compare(res, prep_dir, a.task, methods, alpha=cfg.alpha, winner_metric=a.winner_metric, logger=log)
    log(f"all done in {(time.time() - t0) / 60:.1f} min -> {res.resolve()}")
    logf.close()


if __name__ == "__main__":
    main()
