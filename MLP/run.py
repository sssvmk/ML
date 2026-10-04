#!/usr/bin/env python3
r"""
MLP pipeline: hyperparameter search -> final training -> test evaluation -> model registry,
as ONE command, with everything tracked in MLflow and written to one result folder.

    python run.py --data WHERE --task TASK --target COLUMN --out FOLDER

  --data    where the data comes from:  california | mnist | file.csv | file.parquet | module:Class
  --task    classification | regression   (implied for california and mnist)
  --target  the column to predict          (implied for california and mnist)
  --out     result folder: model.pt, results.json, MODEL_CARD.md, the MLflow store, the search study

    python run.py --data california --out results/california
    python run.py --data mnist --out results/mnist
    python run.py --data train.csv --task classification --target target --out results/santander \
                  --source-opt id_cols=ID_code
    python run.py --data prices.parquet --task regression --target price --out results/prices
    python run.py --data my_source.py:MySource --task regression --target y --out results/mine \
                  --source-opt table=sales          # your own data source (see datasources.py)
    mlflow ui --backend-store-uri sqlite:///results/santander/mlflow.db     # browse the runs

Data sources are plug-ins (datasources.py): the pipeline asks a source for ready-to-train data and
knows nothing about files or databases. Built in: MNIST, California Housing, CSV, Parquet. To plug in
another one, subclass TableSource and implement read_tables(); --source-opt KEY=VALUE passes it settings.

What the pipeline does with any source: audits the splits (sizes, class balance, NaN/Inf, duplicate
rows across splits), searches hyperparameters with Optuna (every trial a nested MLflow run), trains the
best configuration, evaluates on the test set ONCE, reports classification -> ROC-AUC (macro
one-vs-rest, bootstrap 95% CI), accuracy +- SE, log-loss +- SE, macro-F1; regression -> MSE +- SE,
RMSE/MAE +- SE, R2, and registers the model as MLP-<name> (alias `challenger`; `champion` only if
the reload-and-predict check passes, the optional --min-auc / --max-mse target is met, and validation
loss beats the current champion on the same data).

Splits: a source that ships pre-split data keeps its test set (MNIST: official 60k train / 10k test,
validation carved from train); a single table is split 60/20/20 (stratified for classification).

MLflow layout (experiment "MLP-<name>"):
  pipeline-<time>            parent run: split + audit, search settings, best config, tuning plot
    |- trial-000 ... trial-N   nested run per Optuna trial (params, per-epoch metrics, state)
    |- final-train             nested run: per-epoch metrics, test evaluation, plots, worst errors,
                               model card, and the packaged model (preprocessing included)
By default the MLflow store (mlflow.db + mlartifacts/) is created INSIDE the result folder; use
--tracking-uri (or $MLFLOW_TRACKING_URI) for a shared server.
"""
import argparse
import json
import math
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow import MlflowClient
from mlflow.models import infer_signature

import datasources
import report
import tracking
import tuning
from mlp_core import (DEFAULTS, PRIMARY, build_mlp, fingerprint, fit, get_device,
                      make_loaders, make_loss, run_epoch, set_seed, subset)
from serving_model import MLPModel
from splits import audit_splits

HERE = Path(__file__).resolve().parent
CODE_PATHS = [str(HERE / f) for f in ("mlp_core.py", "infer.py", "serving_model.py", "tabular.py")]
OVERRIDE_KEYS = ["hidden", "optimizer", "lr", "momentum", "batch_size", "epochs", "gamma",
                 "weight_decay", "dropout", "patience"]


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, usage=(
            "run.py --data WHERE [--task classification|regression] [--target COLUMN] [--out FOLDER] [options]"))
    core = ap.add_argument_group("what to run")
    core.add_argument("--data", required=True, metavar="WHERE",
                      help="built-in name (%s), a path to a .csv / .csv.gz / .parquet file or directory, "
                           "or a plug-in 'module:Class' / 'file.py:Class'" % ", ".join(datasources.registered_names()))
    core.add_argument("--task", choices=["classification", "regression", "auto"],
                      help="implied for california and mnist; required for your own data ('auto' guesses it "
                           "from the target column: text or few integer values -> classification)")
    core.add_argument("--target", help="name of the column to predict (required for your own data)")
    core.add_argument("--out", help="result folder: model.pt, results.json, MODEL_CARD.md, the MLflow store "
                                    "and the search study (default runs/<name>)")
    d = ap.add_argument_group("data source")
    d.add_argument("--source-opt", action="append", default=[], metavar="KEY=VALUE",
                   help="setting for the data source, repeatable; lists are comma-separated. CSV / Parquet: "
                        "id_cols, drop_cols, date_cols, categorical_cols, val_fraction, test_fraction, "
                        "max_categories; CSV also test_csv, val_csv, join_csv, join_on; Parquet also test_file, "
                        "val_file. A plug-in declares its own in OPTIONS.")
    d.add_argument("--name", help="short name for the experiment / registered model / default result folder "
                                  "(default: the source's name, e.g. the file name without extension)")
    d.add_argument("--data-dir", default="./data", help="download cache for built-in datasets")
    g = ap.add_argument_group("advanced: hyperparameter search (part of the pipeline)")
    g.add_argument("--trials", type=int, default=30, help="Optuna trials; 0 = skip tuning")
    g.add_argument("--timeout", type=float, help="stop the search after this many seconds")
    g.add_argument("--trial-epochs", type=int, help="max epochs per trial (default 10 mnist / 40 housing)")
    g.add_argument("--trial-patience", type=int, default=3)
    g.add_argument("--train-subset", type=int,
                   help="rows of training data used during search (default 20000 mnist, all housing); "
                        "the final training always uses the full training split")
    g.add_argument("--startup-trials", type=int, default=8, help="random trials before TPE kicks in")
    g.add_argument("--resume", action="store_true", help="continue the previous Optuna study in --study-dir")
    g.add_argument("--study-dir", help="Optuna study storage (default: the result folder)")
    g.add_argument("--params", help="JSON config to use instead of searching (a best_params.json, or a plain dict)")
    t = ap.add_argument_group("advanced: training (overrides apply only with --trials 0 or --params)")
    t.add_argument("--hidden", type=int, nargs="+")
    t.add_argument("--optimizer", choices=["sgd", "adamw"])
    t.add_argument("--lr", type=float)
    t.add_argument("--momentum", type=float)
    t.add_argument("--batch-size", dest="batch_size", type=int)
    t.add_argument("--epochs", type=int)
    t.add_argument("--gamma", type=float)
    t.add_argument("--weight-decay", dest="weight_decay", type=float)
    t.add_argument("--dropout", type=float)
    t.add_argument("--patience", type=int)
    t.add_argument("--final-epochs", type=int, help="max epochs for the final training run")
    a = ap.add_argument_group("advanced: evaluation and acceptance")
    a.add_argument("--bootstrap", type=int, default=200,
                   help="bootstrap resamples for the AUC confidence interval (classification)")
    a.add_argument("--min-auc", type=float, help="acceptance target: minimum test AUC (classification)")
    a.add_argument("--max-mse", type=float, help="acceptance target: maximum test MSE (regression)")
    a.add_argument("--promotion-margin", type=float, default=0.001,
                   help="a challenger must beat the champion's validation loss by this relative margin "
                        "(default 0.1%%) to be promoted")
    m = ap.add_argument_group("advanced: MLflow")
    m.add_argument("--tracking-uri", help="default: $MLFLOW_TRACKING_URI or sqlite:///mlflow.db")
    m.add_argument("--experiment", help="default MLP-<name>")
    m.add_argument("--registered-model-name", help="default MLP-<name>")
    m.add_argument("--no-register", action="store_true", help="log the model but do not register it")
    m.add_argument("--run-name")
    m.add_argument("--seed", type=int, default=42)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)

    # ---- 1. the data source (the plug-in point): resolve it, load it, check what it returned ----
    try:
        source = datasources.resolve_source(
            args.data, task=args.task, target=args.target, data_dir=args.data_dir,
            options=datasources.parse_option_pairs(args.source_opt))
        tag = args.name or source.default_name()
        from tabular import validate_name
        validate_name(tag)
    except ValueError as e:  # DataSourceError is a ValueError
        ap.error(str(e))
    ds = source.profile  # which default hyperparameters / search ranges apply: mnist | housing | tabular

    out_dir = Path(args.out or f"runs/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    study_dir = args.study_dir or str(out_dir)
    model_name = args.registered_model_name or f"MLP-{tag}"

    overrides = {k: getattr(args, k) for k in OVERRIDE_KEYS if getattr(args, k) is not None}
    searching = args.trials > 0 and not args.params
    if overrides and searching:
        ap.error("training overrides (--lr, --hidden, ...) only apply without a search: use --trials 0 or --params")

    set_seed(args.seed)
    device = get_device()
    print(f"data source: {source.describe()} | name: {tag} | result folder: {out_dir}")
    try:
        splits, info = source.load(args.seed)
        datasources.validate_loaded(splits, info, who=source.describe())
    except (ValueError, OSError, ImportError, RuntimeError) as e:
        raise SystemExit(f"error: could not load the data from {source.describe()}: {e}")
    uri = tracking.setup(args.tracking_uri, args.experiment or f"MLP-{tag}", default_dir=out_dir)
    print(f"MLflow tracking: {uri} | experiment: {args.experiment or f'MLP-{tag}'} | device: {device}")

    task, key = info["task"], PRIMARY[info["task"]]
    fp = fingerprint(splits)
    sizes = {k: len(v) for k, v in splits.items()}
    print(f"task: {task} | splits: {sizes} | data fingerprint: {fp}")
    if args.min_auc is not None and task != "classification":
        ap.error("--min-auc only applies to classification datasets")
    if args.max_mse is not None and task != "regression":
        ap.error("--max-mse only applies to regression datasets")
    split_info = info.get("split_info", {"presplit": None, "strategy": "not described by the loader"})
    audit = audit_splits(splits, info)
    print(f"split strategy: {split_info['strategy']}")
    print(f"data audit: {'OK' if audit['ok'] else 'FAILED'} | warnings: {audit['warnings'] or 'none'}")

    run_name = args.run_name or f"pipeline-{time.strftime('%Y%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as parent:
        mlflow.set_tags({**tracking.env_tags(device), "phase": "pipeline", "dataset": tag, "task": task,
                         "data_source": info.get("source", ds), "data_fingerprint": fp,
                         "tuned": str(searching).lower()})
        mlflow.log_params({
            "dataset": tag, "seed": args.seed, "metric_primary": key, "selection_metric": "best validation loss",
            "split_train": sizes["train"], "split_val": sizes["val"], "split_test": sizes["test"],
            "search_trials": args.trials if searching else 0,
        })
        mlflow.set_tags({"split_presplit": str(split_info["presplit"]).lower(),
                         "split_strategy": split_info["strategy"], "data_audit_ok": str(audit["ok"]).lower()})
        mlflow.log_dict(audit, "data/audit.json")
        if info.get("lineage"):
            mlflow.log_dict(info["lineage"], "data/lineage.json")
        if info.get("notes"):
            mlflow.log_text("\n".join(info["notes"]), "data/notes.txt")
            print("data notes:\n  - " + "\n  - ".join(info["notes"]))
        if not audit["ok"]:
            raise RuntimeError(f"data audit failed, nothing was trained: {audit['errors']}")
        if info["baseline"]:
            mlflow.log_metrics({f"ref_{k}": v for k, v in info["baseline"].items()})

        # ------------------------------- 1) hyperparameter search ------------------------------- #
        search_summary = None
        if searching:
            sub = args.train_subset if args.train_subset is not None else (20000 if ds == "mnist" else None)  # csv: pass --train-subset for big files
            search_splits = dict(splits)
            search_splits["train"] = subset(splits["train"], sub)
            trial_epochs = args.trial_epochs or (10 if ds == "mnist" else 40)
            mlflow.log_params({"search_trial_epochs": trial_epochs, "search_trial_patience": args.trial_patience,
                               "search_train_rows": len(search_splits["train"]), "search_sampler": "TPE",
                               "search_pruner": "MedianPruner", "search_startup_trials": args.startup_trials})
            search_summary = tuning.run_search(
                ds, search_splits, info, device, trials=args.trials, timeout=args.timeout,
                trial_epochs=trial_epochs, trial_patience=args.trial_patience,
                startup_trials=args.startup_trials, seed=args.seed, study_dir=study_dir, resume=args.resume)
            cfg = dict(search_summary["best_cfg"])
            study = search_summary["study"]
            mlflow.log_metrics({"search_best_val_loss": search_summary["best_value"],
                                "search_trials_run": search_summary["n_trials"],
                                "search_trials_pruned": search_summary["n_pruned"]})
            if search_summary["baseline_value"] is not None:
                mlflow.log_metrics({"search_default_val_loss": search_summary["baseline_value"],
                                    "search_improvement_vs_default":
                                        search_summary["baseline_value"] - search_summary["best_value"]})
            mlflow.log_text(tuning.trials_csv_text(study), "search/trials.csv")
            fig = tuning.tuning_figure(study)
            mlflow.log_figure(fig, "search/tuning.png")
            import matplotlib.pyplot as plt
            plt.close(fig)
            mlflow.log_dict({"best_trial": search_summary["best_trial"], "params": search_summary["best_params"],
                             "config": cfg}, "search/best_params.json")
            mlflow.set_tag("best_trial", str(search_summary["best_trial"]))
            print(f"\nbest trial {search_summary['best_trial']}: val_loss {search_summary['best_value']:.4f}"
                  + (f" (default config: {search_summary['baseline_value']:.4f})"
                     if search_summary["baseline_value"] is not None else ""))
        else:
            cfg = dict(DEFAULTS[ds])
            if args.params:
                data = json.loads(Path(args.params).read_text())
                cfg.update(data.get("config", data))
                mlflow.set_tag("config_source", args.params)
            cfg.update(overrides)

        cfg["epochs"] = args.final_epochs or (cfg["epochs"] if not searching else DEFAULTS[ds]["epochs"])
        if searching:
            cfg["patience"] = DEFAULTS[ds]["patience"]

        # ------------------------------- 2) final training + evaluation ------------------------------- #
        with mlflow.start_run(run_name="final-train", nested=True,
                              tags={"phase": "final_candidate", "dataset": tag, "task": task,
                                    "data_fingerprint": fp}) as final:
            set_seed(args.seed)
            mlflow.log_params(tracking.flat_params(cfg))
            loaders = make_loaders(splits, cfg["batch_size"])
            model = build_mlp(info["in_dim"], cfg["hidden"], info["out_dim"], cfg["dropout"]).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            mlflow.log_param("n_params", n_params)

            v0, _, _ = run_epoch(model, loaders["val"], make_loss(task), device, task)
            mlflow.log_metric("initial_val_loss", v0)
            print(f"\nfinal training | params {n_params:,} | initial val loss {v0:.4f} "
                  f"(sanity check, roughly {math.log(info['out_dim']) if task == 'classification' else 1.0:.3f}"
                  f"{' = ln(classes)' if task == 'classification' else ''}; random initialisation and class "
                  f"imbalance can move it, a huge value would signal a bug)")

            t0 = time.time()
            res = fit(model, loaders, cfg, info, device, on_epoch=tracking.log_epoch)
            if res["best_state"] is None:
                raise RuntimeError("Final training diverged; lower the learning rate.")
            model.load_state_dict(res["best_state"])
            best_row = res["rows"][res["best_epoch"] - 1]
            mlflow.log_metrics({"final_val_loss": res["best_val_loss"], "best_epoch": res["best_epoch"],
                                f"final_val_{key}": best_row[f"val_{key}"],
                                "train_seconds": time.time() - t0})

            fig = tracking.curves_figure(res["rows"], task)
            mlflow.log_figure(fig, "curves.png")
            import matplotlib.pyplot as plt
            plt.close(fig)
            diag = tracking.diagnose(res["rows"], res["best_epoch"])
            mlflow.log_dict(diag, "diagnosis.json")
            mlflow.log_metric("val_over_train_loss", diag["val_over_train_loss"])

            # Test set: evaluated exactly once, here.
            test_m = tracking.evaluate_and_log(model, loaders["test"], info, device, prefix="test",
                                               n_boot=args.bootstrap, seed=args.seed)
            target_ok, target_msg = tracking.check_targets(task, test_m, args.min_auc, args.max_mse)
            mlflow.set_tag("acceptance_target", target_msg)

            # Self-contained bundle: weights + architecture + preprocessing.
            bundle_path = out_dir / "model.pt"
            torch.save({
                "format_version": 1, "dataset": tag, "task": task,
                "in_dim": info["in_dim"], "out_dim": info["out_dim"],
                "hidden": list(cfg["hidden"]), "dropout": float(cfg["dropout"]),
                "state_dict": {k: v.cpu() for k, v in res["best_state"].items()},
                "preproc": info["preproc"], "class_names": info["class_names"],
                "test_metrics": test_m, "config": cfg, "seed": args.seed,
            }, bundle_path)

            # Input example + signature; the packaged model must reproduce the original.
            from infer import Predictor
            pred = Predictor(str(bundle_path))
            z = splits["test"].tensors[0][:5].numpy()
            if info["preproc"]["type"] == "tabular":
                example = info["example_input"]  # raw rows; the model applies the stored preprocessing
                expected = pred.tabular_frame(example)
            elif task == "regression":
                pre = info["preproc"]
                raw = z * np.array(pre["x_scale"]) + np.array(pre["x_mean"])
                example = pd.DataFrame(raw.astype("float64"), columns=pre["feature_names"])
                r = pred.predict_housing(example.to_numpy(np.float32))
                expected = pd.DataFrame({"prediction_100k": [x["prediction_100k"] for x in r],
                                         "prediction_usd": [x["prediction_usd"] for x in r]})
            else:
                px = np.clip(z * info["preproc"]["std"] + info["preproc"]["mean"], 0, 1).astype("float32")
                example = px
                probs = pred.mnist_probs(px)
                expected = pd.DataFrame({"label": probs.argmax(1).astype("int64"),
                                         "confidence": probs.max(1).astype("float64")})
            signature = infer_signature(example, MLPModel_output_example(pred, example, task))

            model_info = mlflow.pyfunc.log_model(
                name="model", python_model=MLPModel(), artifacts={"bundle": str(bundle_path)},
                signature=signature, input_example=example, code_paths=CODE_PATHS,
                pip_requirements=[f"mlflow=={mlflow.__version__}", f"torch=={torch.__version__.split('+')[0]}",
                                  f"numpy=={np.__version__}", f"pandas=={pd.__version__}"],
                registered_model_name=None if args.no_register else model_name)

            promoted, decision = None, "registration skipped (--no-register)"
            if not args.no_register:
                client = MlflowClient()
                version = model_info.registered_model_version
                ok, msg = tracking.verify_registered_model(f"models:/{model_name}/{version}", example, expected)
                mlflow.set_tag("registry_verification", msg)
                promoted, decision = tracking.register_and_promote(
                    client, model_name, version, final.info.run_id, res["best_val_loss"], fp, ok, msg,
                    target_ok=target_ok, target_msg=target_msg, margin=args.promotion_margin)
                mlflow.set_tags({"registered_model": model_name, "registered_version": str(version),
                                 "promotion_decision": decision})

            env = tracking.env_tags(device)
            card = report.model_card(
                dataset=tag, task=task, cfg=cfg, n_params=n_params, info=info, audit=audit,
                val_metrics={k: v for k, v in best_row.items() if k.startswith("val_")},
                test_metrics=test_m, diagnosis=diag, search=search_summary, promotion=decision,
                ids={"fingerprint": fp, "experiment": mlflow.get_experiment(parent.info.experiment_id).name,
                     "parent_run_id": parent.info.run_id, "final_run_id": final.info.run_id},
                versions=env, git_commit=env["git_commit"], target_msg=target_msg, kind=ds)
            (out_dir / "MODEL_CARD.md").write_text(card)
            mlflow.log_text(card, "MODEL_CARD.md")

            results = {"dataset": tag, "task": task, "config": cfg, "seed": args.seed, "n_params": n_params,
                       "split": split_info, "audit": audit, "acceptance_target": target_msg,
                       "best_epoch": res["best_epoch"], "final_val_loss": res["best_val_loss"],
                       "test_metrics": test_m, "reference": info["baseline"], "diagnosis": diag,
                       "mlflow": {"tracking_uri": uri, "experiment": mlflow.get_experiment(
                           parent.info.experiment_id).name, "parent_run_id": parent.info.run_id,
                           "final_run_id": final.info.run_id, "promotion": decision}}
            (out_dir / "results.json").write_text(json.dumps(results, indent=2))
            mlflow.log_dict(results, "results.json")

        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
        mlflow.log_metric("final_val_loss", res["best_val_loss"])

    print("\n=== TEST RESULTS (best-validation checkpoint, test set used once) ===")
    for line in report.headline_lines(task, test_m, info["baseline"], info):
        print("  " + line)
    print("  (standard errors/intervals reflect test-set size only, not training-seed variation)")
    print(f"registry: {decision}")
    print(f"local files: {out_dir.resolve()} | browse runs: mlflow ui --backend-store-uri {uri}")
    return results


def MLPModel_output_example(pred, example, task):
    """Run the packaged wrapper's logic once to get an output example for the signature."""
    from serving_model import MLPModel as _M

    class _Ctx:  # minimal stand-in for the MLflow python-model context
        artifacts = {}

    m = _M()
    m.predictor = pred
    return m.predict(_Ctx(), example)


if __name__ == "__main__":
    main()
