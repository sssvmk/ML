#!/usr/bin/env python
"""
run_all.py - main orchestrator.

    python run_all.py --data /path/to/santander_train.csv --results results

[1] load + stratified 80/10/10 split + EDA   [2] two feature sets (lean, rich)   [3] cached CV statistics
[4] 9 methods x 2 feature sets trained / validated / tested IN PARALLEL with live progress   [5] comparison + winner
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "methods"))

METHODS = ["l1_logistic", "logistic", "rda", "qda", "separating_hyperplane", "perceptron", "lda", "reduced_rank_lda", "indicator_regression"]
FEATURE_SETS = ["lean", "rich"]


def _worker(fs, name, fs_dir, out_dir, cfg_dict, q, blas_threads):
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(blas_threads)
    except Exception:
        pass
    import common
    cfg = common.RunConfig(**{**cfg_dict, "rda_alphas": tuple(cfg_dict["rda_alphas"]), "rda_gammas": tuple(cfg_dict["rda_gammas"])})
    key = f"{fs}/{name}"
    prog = common.Progress(key, lambda n, d, t, m: q.put(("tick", n, d, t, m)))
    try:
        s = importlib.import_module(name).run(fs_dir, out_dir, cfg, prog)
        q.put(("done", key, s["test"]["auc"], s["test"]["error"], ""))
        return key, True
    except Exception as e:                               # noqa: BLE001
        q.put(("error", key, 0, 0, f"{type(e).__name__}: {e}"))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "ERROR.txt").write_text(traceback.format_exc())
        return key, False


def main():
    ap = argparse.ArgumentParser(description="Santander: linear classification methods, two feature sets, compared on a stratified test set")
    ap.add_argument("--data", required=True, help="path to santander_train.csv")
    ap.add_argument("--results", default="results")
    ap.add_argument("--methods", nargs="*", default=METHODS, choices=METHODS)
    ap.add_argument("--feature-sets", nargs="*", default=FEATURE_SETS, choices=FEATURE_SETS)
    ap.add_argument("--workers", type=int, default=0, help="parallel processes (default: min(#jobs, #cpus))")
    ap.add_argument("--cv-folds", type=int, default=10)
    ap.add_argument("--winner-metric", choices=["auc", "error"], default="auc")
    ap.add_argument("--perceptron-epochs", type=int, default=50)
    ap.add_argument("--svm-C", type=float, default=100.0)
    ap.add_argument("--svm-max-iter", type=int, default=300)
    ap.add_argument("--l1-nlambda", type=int, default=30, help="length of the L1-logistic lambda path")
    ap.add_argument("--l1-ratio", type=float, default=1e-3, help="lambda_min / lambda_max")
    ap.add_argument("--skip-eda", action="store_true")
    ap.add_argument("--reuse-prepared", action="store_true")
    a = ap.parse_args()

    import common, features, compare
    res = Path(a.results)
    res.mkdir(parents=True, exist_ok=True)
    logf = open(res / "run_all.log", "w")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} | {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    cfg = common.RunConfig(cv_folds=a.cv_folds, winner_metric=a.winner_metric, perceptron_max_epochs=a.perceptron_epochs,
                           svm_C=a.svm_C, svm_max_iter=a.svm_max_iter, l1_nlambda=a.l1_nlambda, l1_ratio=a.l1_ratio)
    t0 = time.time()
    prep_root = res / "prepared"
    if a.reuse_prepared and (prep_root / "feature_meta.json").exists():
        log("[1-2/5] reusing prepared data (EDA/split/features skipped)")
        cfg.to_json(prep_root / "run_config.json")
    else:
        log("[1/5] loading data, stratified 80/10/10 split, EDA")
        df = features.load_table(a.data)
        if not a.skip_eda:
            import eda
            eda.run_eda(df, res, seed=cfg.seed, logger=log)
        log("[2/5] feature engineering (lean + rich), fitted on train only")
        features.build_prepared(df, prep_root, cfg, logger=log)
        del df

    log("[3/5] caching stratified CV statistics")
    for fs in a.feature_sets:
        prep = common.Prepared(prep_root / fs)
        common.get_bundle(prep, cfg)
        log(f"      {fs}: p={prep.p} | train={prep.n_train} val={prep.n_val} test={prep.n_test}")

    jobs = [(fs, m) for m in a.methods for fs in a.feature_sets]
    ncpu = os.cpu_count() or 1
    workers = a.workers or min(len(jobs), ncpu)
    blas = max(1, ncpu // workers)
    log(f"[4/5] running {len(jobs)} jobs ({len(a.methods)} methods x {len(a.feature_sets)} feature sets) in parallel: {workers} processes, {blas} BLAS thread(s) each")
    ctx = mp.get_context("spawn")
    q = ctx.Manager().Queue()
    finished, failed = {}, {}
    try:
        from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn
        use_rich = True
    except Exception:
        use_rich = False
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = [ex.submit(_worker, fs, m, str(prep_root / fs), str(res / fs / m), cfg.__dict__, q, blas) for fs, m in jobs]
        keys = [f"{fs}/{m}" for fs, m in jobs]
        if use_rich:
            prog = Progress(TextColumn("[bold]{task.fields[name]:<30}"), BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), TextColumn("{task.fields[msg]}"))
            tasks = {k: prog.add_task("", total=1, name=k, msg="queued") for k in keys}
            prog.start()
        while len(finished) + len(failed) < len(jobs):
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
                finished[name] = (x, y)
                if use_rich:
                    prog.update(tasks[name], completed=prog.tasks[tasks[name]].total, msg=f"[green]done: AUC {x:.4f}, error {y:.4f}")
                log(f"      {name} finished: test AUC {x:.5f}, test error {y:.5f}")
            else:
                failed[name] = msg
                if use_rich:
                    prog.update(tasks[name], msg=f"[red]FAILED: {msg}")
                log(f"      {name} FAILED: {msg}")
        if use_rich:
            prog.stop()
    log(f"      finished {len(finished)}/{len(jobs)} jobs; failed: {list(failed) or 'none'}")

    log("[5/5] cross-method comparison on the common test set")
    compare.compare(res, a.feature_sets, a.methods, winner_metric=a.winner_metric, logger=log)
    log(f"all done in {(time.time() - t0) / 60:.1f} min -> {res.resolve()}")
    logf.close()


if __name__ == "__main__":
    main()
