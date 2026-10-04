"""Hyperparameter tuning INSIDE the pipeline: one Optuna study per method.

* hyperparameters = learning rate (unless the method has none) + L1 and L2 penalty strengths (every method trains with
  L1 + L2) + the method's own parameters;
* every trial trains with a short budget (tune.epochs, or tune.fullbatch_iters for full-batch methods) and is scored on the
  VALIDATION set (tune.metric). The test set is never touched;
* the method's default configuration is always the first trial, so tuning cannot do worse than the defaults on validation;
* every trial is a nested MLflow run; unpromising trials are pruned, diverging trials are pruned instead of crashing.
"""
from __future__ import annotations

import logging
from dataclasses import replace

import mlflow
import optuna

from .core import Ctx, DivergenceError
from .methods.base import Method

log = logging.getLogger("optpipe.tuning")
optuna.logging.set_verbosity(optuna.logging.WARNING)

METRICS = {"accuracy": ("val_acc", "maximize"), "loss": ("val_loss", "minimize"), "auc": ("val_auc", "maximize")}


def suggest_hp(method: Method, trial, cfg: dict) -> dict:
    reg = cfg["regularization"]
    hp = dict(method.defaults())
    if method.tune_lr:
        hp["lr"] = trial.suggest_float("lr", *method.lr_range, log=True)
    hp["l2"] = trial.suggest_float("l2", *reg["l2_range"], log=True)
    hp["l1"] = trial.suggest_float("l1", *reg["l1_range"], log=True)
    hp.update(method.space(trial))
    return hp


def default_trial_params(method: Method, cfg: dict) -> dict:
    """Parameters of the enqueued first trial = the method's defaults (restricted to names the search space uses)."""
    d = method.defaults()
    ft = optuna.trial.FixedTrial({k: v for k, v in d.items()})
    suggest_hp(method, ft, cfg)
    return {k: d[k] for k in ft.params}


def tune(method: Method, data, cfg: dict, ctx: Ctx):
    """Returns (best_hp, trials_dataframe)."""
    tcfg = cfg["tune"]
    key, direction = METRICS[tcfg["metric"]]
    tctx = replace(ctx, phase="tune", epochs=method.budget(cfg, "tune"))

    def objective(trial: optuna.Trial) -> float:
        hp = suggest_hp(method, trial, cfg)
        with mlflow.start_run(run_name=f"trial_{trial.number:02d}", nested=True):
            mlflow.log_params(hp)
            mlflow.set_tag("phase", "tune")

            def on_epoch(row):
                mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=row["epoch"])
                trial.report(row[key], row["epoch"])
                if trial.should_prune():
                    raise optuna.TrialPruned()

            try:
                res = method.fit(data, hp, tctx, on_epoch=on_epoch)
            except (DivergenceError, FloatingPointError, RuntimeError) as e:
                if isinstance(e, RuntimeError) and not isinstance(e, DivergenceError) and "singular" not in str(e).lower() \
                        and "nan" not in str(e).lower():
                    raise
                mlflow.set_tag("diverged", "true")
                raise optuna.TrialPruned()
            value = float(res.history[-1][key])
            if value != value:                      # NaN metric
                raise optuna.TrialPruned()
            mlflow.log_metric(f"tuned_{key}", value)
            return value

    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=ctx.seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=2))
    study.enqueue_trial(default_trial_params(method, cfg))                     # defaults are always evaluated
    study.optimize(objective, n_trials=int(tcfg["n_trials"]), gc_after_trial=True)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        log.warning("%s: no completed trials; falling back to defaults", method.name)
        return method.defaults(), study.trials_dataframe()
    best_hp = {**method.defaults(), **study.best_params}
    log.info("%s best %s=%.4f hp=%s", method.name, key, study.best_value, best_hp)
    return best_hp, study.trials_dataframe()
