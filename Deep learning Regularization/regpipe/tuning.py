"""Hyperparameter tuning INSIDE the pipeline: one Optuna study per method.

* Every trial trains the method with a short budget (tune.epochs) and is scored on the
  VALIDATION set (tune.metric).  The test set is never touched.
* The method's default configuration is always the first trial, so tuning cannot do worse
  than the defaults on validation.
* Every trial is a nested MLflow run (params, per-epoch metrics, final score).
* Unpromising trials are pruned (median pruner); diverging trials are pruned, not crashed.
"""
from __future__ import annotations

import logging
from dataclasses import replace

import mlflow
import optuna

from .methods.base import Ctx, Method
from .trainer import DivergenceError

log = logging.getLogger("regpipe.tuning")
optuna.logging.set_verbosity(optuna.logging.WARNING)

METRICS = {"accuracy": ("val_acc", "maximize"), "loss": ("val_loss", "minimize"), "auc": ("val_auc", "maximize")}


def score_of(result, metric: str) -> float:
    key, _ = METRICS[metric]
    row = result.history[result.extras["best_epoch"] - 1] if "best_epoch" in result.extras else result.history[-1]
    return float(row[key])


def tune(method: Method, data, cfg: dict, ctx: Ctx):
    """Returns (best_hp, trials_dataframe)."""
    tcfg = cfg["tune"]
    key, direction = METRICS[tcfg["metric"]]
    tctx = replace(ctx, phase="tune", epochs=int(tcfg["epochs"]))

    def objective(trial: optuna.Trial) -> float:
        hp = {**method.defaults(), "lr": trial.suggest_float("lr", 5e-3, 5e-1, log=True)}
        hp.update(method.space(trial))
        with mlflow.start_run(run_name=f"trial_{trial.number:02d}", nested=True):
            mlflow.log_params({k: v for k, v in hp.items()})
            mlflow.set_tag("phase", "tune")

            def on_epoch(row):
                mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=row["epoch"])
                trial.report(row[key], row["epoch"])
                if trial.should_prune():
                    raise optuna.TrialPruned()

            try:
                res = method.fit(data, hp, tctx, on_epoch=on_epoch)
            except DivergenceError:
                mlflow.set_tag("diverged", "true")
                raise optuna.TrialPruned()
            value = score_of(res, tcfg["metric"])
            mlflow.log_metric(f"tuned_{key}", value)
            return value

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=ctx.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=2),
    )
    study.enqueue_trial({k: v for k, v in method.defaults().items()})   # defaults are always evaluated
    study.optimize(objective, n_trials=int(tcfg["n_trials"]), gc_after_trial=True)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        log.warning("%s: no completed trials; falling back to defaults", method.name)
        return method.defaults(), study.trials_dataframe()
    best_hp = {**method.defaults(), **study.best_params}
    log.info("%s best %s=%.4f hp=%s", method.name, key, study.best_value, best_hp)
    return best_hp, study.trials_dataframe()
