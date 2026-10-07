"""Pipeline entry point: input data path in -> experiments, winner, packaged champion, reports out.

    python -m aeclf.run_pipeline --data airline.csv [--config config/default.yaml] [--set autoencoder.search.n_trials=6] [--explain offline|llm]
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import time
from pathlib import Path

import numpy as np
import yaml

from .config import load_config
from .eda import run_eda
from .encoding import TabularEncoder
from .experiments import ExperimentData, run_all
from .final_evaluation import evaluate_on_test
from .io_utils import load_training_data
from .registry import register_candidate
from .reporting import write_reports
from .schema import infer_schema
from .selection import declare_winner, load_candidates
from .splitting import adversarial_validation, make_splits, population_report
from .tracking import Tracker
from .training import to_tensors
from .utils import dump_json, environment_info, file_sha256, git_commit, seed_everything


def log(msg: str):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def run(cfg: dict, data_path: str, run_id: str | None = None, skip_eda: bool = False, explain: str | None = None, overwrite: bool = False) -> Path:
    t_start = time.perf_counter()
    seed = cfg["project"]["seed"]
    seed_everything(seed)
    run_id = run_id or dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(cfg["project"]["output_dir"]) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{run_dir} already contains a previous run" + (" whose TEST set was already evaluated (test_set_used.lock)" if (run_dir / "test_set_used.lock").exists() else "")
                + ". Use a new --run-id, or pass --overwrite to delete it and start fresh.")
        if (run_dir / "test_set_used.lock").exists():
            log("WARNING: --overwrite deletes a run whose test set was already evaluated. The split is seeded, so the SAME test rows will be scored again. "
                "Fine for trial runs; for the final run do not tune anything based on test results.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    log(f"loading {data_path}")
    X, y, ids, info = load_training_data(data_path, cfg)
    data_hash = file_sha256(data_path)
    ctx = {"data_hash": data_hash, "rows": len(X), "git_commit": git_commit(), "seed": seed, "env": environment_info(), **info}

    schema = infer_schema(X, cfg)
    dump_json(schema.to_dict(), run_dir / "schema.json")
    log(f"schema: {len(schema.continuous)} continuous, {len(schema.ordinal)} ordinal, {len(schema.nominal)} nominal")
    if not skip_eda:
        log("EDA + statistical tests")
        run_eda(X, y, ids, schema, cfg, run_dir / "eda")

    splits = make_splits(X, y, cfg)
    pop = population_report(X, y, splits, schema, cfg)
    dump_json(pop, run_dir / "splits" / "population_report.json")
    np.savez_compressed(run_dir / "splits" / "indices.npz", **splits)
    tr, va, te = splits["train"], splits["validation"], splits["test"]
    ctx.update(adversarial_auc=adversarial_validation(X.iloc[tr], X.iloc[va], seed=seed), n_train=len(tr), n_val=len(va), n_test=len(te))
    log(f"split {len(tr):,}/{len(va):,}/{len(te):,}; representative={pop['representative']}; adversarial AUC={ctx['adversarial_auc']:.3f}")

    enc = TabularEncoder(schema, cfg["encoding"]["ordinal_as"], cfg["encoding"]["skew_threshold"]).fit(X.iloc[tr])
    dump_json({"continuous": enc.cont_cols, "categorical": enc.cat_cols, "cardinalities": enc.cards,
               "log_transformed": [c for c in enc.cont_cols if enc.cont_stats[c]["log"]]}, run_dir / "encoding.json")
    Xtr, Xva = X.iloc[tr].reset_index(drop=True), X.iloc[va].reset_index(drop=True)
    data = ExperimentData(Xtr, y.iloc[tr].to_numpy(), Xva, y.iloc[va].to_numpy(), schema, enc, to_tensors(enc, Xtr, y.iloc[tr]),
                          to_tensors(enc, Xva, y.iloc[va]), cfg, {"data_hash": data_hash[:16], "git_commit": ctx["git_commit"], "rows": len(X)})
    log(f"encoded: {len(enc.cont_cols)} continuous + {len(enc.cat_cols)} categorical columns")

    tracker = Tracker(cfg, run_dir)
    with tracker.run(run_id, tags={"phase": "experiment", "data_hash": data_hash[:16]}):
        tracker.log_params({"seed": seed, "primary_metric": cfg["metric"]["primary"], "rows": len(X)})
        cands, ae_results, le = run_all(data, tracker, run_dir, log)

    log("declaring winner")
    sel = declare_winner(load_candidates(run_dir), cfg, run_dir)
    for t in sel["decision_trace"]:
        log("    " + t)
    bundle_dir = run_dir / "candidates" / sel["winner"] / "bundle"
    log("one-shot test-set evaluation of the winner")
    X_te = X.iloc[te].reset_index(drop=True)
    test = evaluate_on_test(bundle_dir, X_te, y.iloc[te].to_numpy(), cfg, run_dir)
    log(f"    TEST {cfg['metric']['primary']} = {test['primary_value']:.4f}; accuracy {test['metrics']['accuracy']:.4f}")

    champion = run_dir / "champion"
    if champion.exists():
        shutil.rmtree(champion)
    shutil.copytree(bundle_dir, champion)
    reg = register_candidate(run_dir, champion, cfg, X_te, tracker.uri if tracker.mlflow else None)
    dump_json(reg, run_dir / "registry.json")
    ctx["total_seconds"] = time.perf_counter() - t_start
    dump_json(ctx, run_dir / "run_context.json")
    write_reports(run_dir, cfg, ctx)
    if explain:
        from .agent.ag2_explainer import explain_run

        log(f"agent explanations ({explain})")
        explain_run(run_dir, cfg, mode=explain)
    log(f"done in {ctx['total_seconds'] / 60:.1f} min -> {run_dir}")
    return run_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="a.b=value")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--skip-eda", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="delete an existing run with the same --run-id and start fresh")
    ap.add_argument("--explain", choices=["llm", "offline"], default=None)
    a = ap.parse_args(argv)
    try:
        run(load_config(a.config, a.set), a.data, a.run_id, a.skip_eda, a.explain, a.overwrite)
    except FileExistsError as exc:
        raise SystemExit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
