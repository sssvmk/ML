"""Pipeline entry point: input data path in -> experiments, winner, packaged champion, reports out.

    python -m airsat.run_pipeline --data airline.csv [--config config/default.yaml] [--set search.n_trials=10] [--explain]
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import time
from pathlib import Path

import numpy as np
import yaml

from .algorithms import KNOWN, load_algorithm
from .algorithms.base import ExperimentData
from .analysis.final_evaluation import evaluate_on_test
from .analysis.select_winner import declare_winner, load_results
from .config import load_config
from .eda import run_eda
from .features import FeatureEngineer
from .io_utils import load_training_data
from .registry import register_candidate
from .reporting import write_experiment_report, write_model_card
from .schema import infer_schema
from .splitting import adversarial_validation, make_splits, population_report
from .tracking import Tracker
from .utils import dump_json, environment_info, file_sha256, git_commit, seed_everything


def log(msg: str):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def run(cfg: dict, data_path: str, run_id: str | None = None, algorithms: list[str] | None = None,
        skip_eda: bool = False, explain: str | None = None) -> Path:
    t_start = time.perf_counter()
    seed = cfg["project"]["seed"]
    seed_everything(seed)
    run_id = run_id or dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(cfg["project"]["output_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # ---- load -----------------------------------------------------------------------------------------------------
    log(f"loading {data_path}")
    X, y, ids, info = load_training_data(data_path, cfg)
    data_hash = file_sha256(data_path)
    context = {"data_hash": data_hash, "rows": len(X), "git_commit": git_commit(), "seed": seed, "env": environment_info(), **info}

    # ---- steps 1-2: understand data, decide variable types ----------------------------------------------------------
    schema = infer_schema(X, cfg, )
    dump_json(schema.to_dict(), run_dir / "schema.json")
    log(f"schema: {len(schema.continuous)} continuous, {len(schema.ordinal)} ordinal, {len(schema.nominal)} nominal, {len(schema.ignored)} ignored")
    if not skip_eda:
        log("EDA + statistical tests")
        run_eda(X, y, ids, schema, cfg, run_dir / "eda")

    # ---- step 2b: split ------------------------------------------------------------------------------------------------
    splits = make_splits(X, y, cfg)
    pop = population_report(X, y, splits, schema, cfg)
    dump_json(pop, run_dir / "splits" / "population_report.json")
    np.savez_compressed(run_dir / "splits" / "indices.npz", **splits)
    tr, va, te = splits["train"], splits["validation"], splits["test"]
    adv = adversarial_validation(X.iloc[tr], X.iloc[va], seed=seed)
    context.update(adversarial_auc=adv, n_train=len(tr), n_val=len(va), n_test=len(te))
    log(f"split {len(tr):,}/{len(va):,}/{len(te):,}; representative={pop['representative']}; adversarial AUC={adv:.3f}")

    # ---- step 4: features (fit on TRAIN only) -------------------------------------------------------------------------
    fe = FeatureEngineer(schema, cfg["features"]).fit(X.iloc[tr])
    X_tr, X_va = fe.transform(X.iloc[tr]), fe.transform(X.iloc[va])
    data = ExperimentData(X_train=X_tr.reset_index(drop=True), y_train=y.iloc[tr].to_numpy(), X_val=X_va.reset_index(drop=True),
                          y_val=y.iloc[va].to_numpy(), schema=schema, feature_engineer=fe, cfg=cfg,
                          run_meta={"data_hash": data_hash[:16], "git_commit": context["git_commit"], "rows": len(X)})
    dump_json(fe.feature_set.to_dict(), run_dir / "feature_set.json")
    log(f"features: {len(fe.feature_set.model_columns)} model inputs")

    # ---- steps 5-11: every algorithm, same protocol --------------------------------------------------------------------
    names = algorithms or cfg["algorithms"]
    unknown = [n for n in names if n not in KNOWN]
    if unknown:
        raise ValueError(f"unknown algorithms {unknown}; known: {KNOWN}")
    tracker = Tracker(cfg, run_dir)
    with tracker.run(run_id, tags={"phase": "experiment", "data_hash": data_hash[:16]}):
        tracker.log_params({"seed": seed, "primary_metric": cfg["metric"]["primary"], "rows": len(X), "git_commit": context["git_commit"]})
        for name in names:
            log(f"--> {name}")
            res = load_algorithm(name).run(data, tracker, run_dir)
            if res.status == "ok":
                log(f"    {res.status}: val {res.val_primary:.4f} (train {res.train_primary:.4f}), {res.total_seconds:.0f}s")
            else:
                log(f"    {res.status}: {res.message}")

    # ---- step 12: analysis + winner -----------------------------------------------------------------------------------------
    log("declaring winner")
    sel = declare_winner(load_results(run_dir), cfg, run_dir)
    for t in sel["decision_trace"]:
        log("    " + t)
    winner = sel["winner"]
    bundle_dir = run_dir / "algorithms" / winner / "bundle"

    log("one-shot test-set evaluation of the winner")
    X_te = X.iloc[te].reset_index(drop=True)
    test = evaluate_on_test(bundle_dir, X_te, y.iloc[te].to_numpy(), cfg, run_dir)
    log(f"    TEST {cfg['metric']['primary']} = {test['primary_value']:.4f}; accuracy {test['metrics']['accuracy']:.4f}")
    if cfg["selection"].get("report_test_for_all"):
        extra = {}
        for r in load_results(run_dir):
            if r["status"] == "ok":
                extra[r["name"]] = evaluate_on_test(Path(r["artifacts"]["bundle_dir"]), X_te, y.iloc[te].to_numpy(), cfg, run_dir, force=True)["primary_value"]
        dump_json({"warning": "test set touched once per algorithm: selection bias possible", "primary_by_algorithm": extra},
                  run_dir / "test_evaluation_all.json")
        dump_json({**test, "test_set_evaluations": len(extra) + 1}, run_dir / "test_evaluation.json")

    # ---- champion package + registry + reports -----------------------------------------------------------------------------------
    champion = run_dir / "champion"
    if champion.exists():
        shutil.rmtree(champion)
    shutil.copytree(bundle_dir, champion)
    reg = register_candidate(run_dir, champion, cfg, X_te, {"algorithm": winner, "data_hash": data_hash[:16], "seed": seed},
                             tracker.uri if tracker.mlflow else None)
    dump_json(reg, run_dir / "registry.json")
    log(f"registry: {reg}")
    context["total_seconds"] = time.perf_counter() - t_start
    dump_json(context, run_dir / "run_context.json")
    write_experiment_report(run_dir, cfg, context)
    write_model_card(run_dir, cfg, context)

    if explain:
        from .agent.ag2_explainer import explain_run

        log(f"agent explanations ({explain})")
        explain_run(run_dir, cfg, mode=explain)
    log(f"done in {context['total_seconds'] / 60:.1f} min -> {run_dir}")
    return run_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to the training CSV/Parquet (must contain the target column)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="a.b=value", help="override config values")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--algorithms", nargs="*", default=None, help="subset of algorithms to run")
    ap.add_argument("--skip-eda", action="store_true")
    ap.add_argument("--explain", choices=["llm", "offline"], default=None, help="also run the AG2 explanation agent")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    run(cfg, a.data, a.run_id, a.algorithms, a.skip_eda, a.explain)


if __name__ == "__main__":
    main()
