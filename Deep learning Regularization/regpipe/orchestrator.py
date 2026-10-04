"""Single orchestrator: for every method -> tune -> final train -> evaluate -> save, all in MLflow.

Per method (nested MLflow runs: orchestrator > method > trials):
  1. tune hyperparameters (Optuna, validation set)            [tuning.py]
  2. train the best configuration for the full epoch budget    [method.fit]
  3. evaluate on validation, then on the TEST set exactly once [evaluate.py]
  4. save curves, history, trials, bundle; log everything to MLflow
After all methods: comparison table/plots, model card, register the best full-data method.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

from . import reporting, tracking
from .bundle import Predictor, save_bundle
from .config import flatten
from .data import audit_splits, load_raw, make_splits, resolve_device, to_float
from .evaluate import evaluate, fgsm_accuracy, sparsity_stats
from .methods import get_methods
from .methods.base import Ctx, Method
from .model import n_params
from .trainer import set_seed
from .tuning import tune

log = logging.getLogger("regpipe")


def _numeric(d: dict) -> dict:
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)}


def run_method(method: Method, data, cfg: dict, device, out_dir: Path) -> dict:
    mdir = out_dir / "methods" / method.name
    mdir.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    with mlflow.start_run(run_name=method.name, nested=True) as run:
        mlflow.set_tags({"method": method.name, "book_section": method.section, "regime": method.regime,
                         "phase": "method", "test_set_evaluations": "1"})
        t0 = time.time()
        # 1. tuning -----------------------------------------------------------------------------
        base_ctx = Ctx(cfg, device, seed, epochs=cfg["train"]["epochs"])
        best_hp, trials = tune(method, data, cfg, base_ctx)
        trials.to_csv(mdir / "trials.csv", index=False)
        (mdir / "best_hp.json").write_text(json.dumps(best_hp, indent=2))
        mlflow.log_params({f"hp.{k}": v for k, v in best_hp.items()})

        # 2. final training with the tuned configuration ----------------------------------------
        ctx = Ctx(cfg, device, seed, phase="final", epochs=cfg["train"]["epochs"])

        def on_epoch(row):
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=row["epoch"])

        res = method.fit(data, best_hp, ctx, on_epoch=on_epoch)
        model = res.model.cpu()
        pd.DataFrame(res.history).to_csv(mdir / "history.csv", index=False)

        # 3. evaluation: validation for diagnostics, TEST exactly once --------------------------
        cpu = torch.device("cpu")
        val = evaluate(model, data.x_val, data.y_val, cpu)
        test = evaluate(model, data.x_test, data.y_test, cpu, bootstrap=100)        # the one test evaluation
        eps = cfg["method_settings"]["robustness_eps"]
        fgsm = fgsm_accuracy(model, data.x_val, data.y_val, cpu, eps)
        extras = {**res.extras, **sparsity_stats(model, data.x_val, cpu), "n_params": n_params(model),
                  **method.extras(res, data, ctx, best_hp)}
        mlflow.log_metrics({**{f"val_{k}": v for k, v in val.items()},
                            **{f"test_{k}": v for k, v in test.items() if not isinstance(v, list)},
                            "val_fgsm_accuracy": fgsm, "train_seconds": res.seconds,
                            **{f"extra.{k}": v for k, v in _numeric(extras).items()}})

        # 4. artifacts --------------------------------------------------------------------------
        diag = reporting.diagnose(res.history)
        plot = reporting.plot_curves(res.history, mdir / "loss_curve.png", f"{method.title}  ({method.regime})")
        bundle = save_bundle(mdir / "bundle", model, {
            "method": method.name, "title": method.title, "hp": best_hp, "val": val, "test_accuracy": test["accuracy"],
            "data_hash": data.info["data_hash"], "seed": seed, "input": "float32 [n,784] in [0,1], white digit on black"})
        result = {"method": method.name, "title": method.title, "section": method.section, "regime": method.regime,
                  "champion_eligible": method.champion_eligible, "best_hp": best_hp, "val": val, "test": test,
                  "val_fgsm_accuracy": fgsm, "extras": extras, "diagnosis": diag,
                  "history_final": res.history[-1], "seconds": time.time() - t0, "run_id": run.info.run_id,
                  "bundle": str(bundle), "loss_curve": str(plot)}
        (mdir / "result.json").write_text(json.dumps(result, indent=2, default=str))
        mlflow.log_artifacts(str(mdir), artifact_path="method")
        mlflow.set_tag("diagnosis", ",".join(diag["verdict"]))
        log.info("%-24s val_acc=%.4f test_acc=%.4f test_auc=%.5f [%s] %.0fs", method.name, val["accuracy"],
                 test["accuracy"], test["auc"], ",".join(diag["verdict"]), result["seconds"])
        result["history"] = res.history
        return result


def run(cfg: dict, method_names="all", out_dir="results", promote: bool = False, approver: str = "") -> dict:
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])
    uri = tracking.setup(cfg, out)

    # data: pre-split check, splits, audit ---------------------------------------------------------
    raw = load_raw(cfg["data"], cfg["seed"])
    data = make_splits(raw, cfg["data"]["val_size"], cfg["seed"])
    audit = audit_splits(data)
    log.info("data: %s | pre-split train/test: yes, validation carved from train | sizes=%s | duplicates "
             "train/test=%d", data.info["source"], data.info["sizes"], audit["duplicates_train_test"])
    (out / "data_audit.json").write_text(json.dumps({"info": data.info, "audit": audit}, indent=2))

    methods = get_methods(method_names)
    tags = tracking.env_tags()
    with mlflow.start_run(run_name="orchestrator") as parent:
        mlflow.set_tags({**tags, "data_hash": data.info["data_hash"], "phase": "orchestrator"})
        mlflow.log_params({k: str(v) for k, v in flatten(cfg).items()})
        mlflow.log_dict({"info": data.info, "audit": audit}, "data_audit.json")
        results = [run_method(m, data, cfg, device, out) for m in methods]

        # summary -----------------------------------------------------------------------------------
        rows = [{k: v for k, v in r.items() if k != "history"} for r in results]
        base = next((r for r in rows if r["method"] == "baseline"), None)
        md = reporting.results_markdown(rows, base["test"]["accuracy"] if base else None)
        (out / "comparison.md").write_text(md)
        pd.DataFrame([{"method": r["method"], "regime": r["regime"], "val_acc": r["val"]["accuracy"],
                       "test_acc": r["test"]["accuracy"], "test_acc_se": r["test"]["accuracy_se"],
                       "test_auc": r["test"]["auc"], "test_loss": r["test"]["loss"],
                       "fgsm_val_acc": r["val_fgsm_accuracy"], "diagnosis": ",".join(r["diagnosis"]["verdict"])}
                      for r in rows]).to_csv(out / "comparison.csv", index=False)
        reporting.plot_comparison(rows, out / "comparison.png")
        reporting.plot_overview({r["method"]: r["history"] for r in results}, out / "all_loss_curves.png")
        eligible = [r for r in rows if r["champion_eligible"]]
        best = max(eligible, key=lambda r: (r["val"]["accuracy"], -r["val"]["loss"]))
        summary = {"data": data.info, "audit": audit, "tags": tags, "best": best, "robustness_eps": cfg["method_settings"]["robustness_eps"],
                   "note": "Selection by validation accuracy among full-data methods. The test set was evaluated once per method."}
        (out / "MODEL_CARD.md").write_text(reporting.model_card(summary))
        (out / "results.json").write_text(json.dumps({"rows": rows, "best_method": best["method"]}, indent=2, default=str))
        for f in ("comparison.md", "comparison.csv", "comparison.png", "all_loss_curves.png", "MODEL_CARD.md", "results.json"):
            mlflow.log_artifact(str(out / f))

        # registry ----------------------------------------------------------------------------------
        reg = register_best(cfg, best, data, out, uri, promote, approver)
        summary["registry"] = reg
        mlflow.log_dict(reg, "registry.json")
        (out / "registry.json").write_text(json.dumps(reg, indent=2, default=str))
    log.info("best full-data method by validation accuracy: %s | registry: %s", best["method"], reg.get("alias_state"))
    return {"out_dir": str(out), "best": best["method"], "rows": rows, "registry": reg, "tracking_uri": uri}


def register_best(cfg, best, data, out, uri, promote, approver) -> dict:
    x_ex = to_float(data.x_val[:5]).numpy()
    expected = Predictor(best["bundle"]).predict_proba(x_ex)
    with mlflow.start_run(run_name=f"register_{best['method']}", nested=True):
        info = tracking.log_model(Path(best["bundle"]), x_ex)
        tags = {"method": best["method"], "val_accuracy": best["val"]["accuracy"], "test_accuracy": best["test"]["accuracy"],
                "data_hash": data.info["data_hash"], "source_run": best["run_id"], "git_commit": tracking.git_commit()}
        name, version = tracking.register_candidate(cfg, info, tags)
    reload_check = tracking.verify_reload(f"models:/{name}/{version}", x_ex, expected, uri)
    gates = tracking.evaluate_gates(cfg, name, version, {"test_accuracy": best["test"]["accuracy"],
                                                         "val_accuracy": best["val"]["accuracy"],
                                                         "data_hash": data.info["data_hash"]}, reload_check)
    reg = {"registered_model": name, "version": version, "method": best["method"], "reload_check": reload_check,
           "gates": gates, "alias_state": "candidate"}
    if promote:
        if tracking.promote(cfg, name, version, approver, gates, out):
            reg["alias_state"] = "champion"
        else:
            reg["promotion_refused"] = "gates failed or no approver given"
    return reg
