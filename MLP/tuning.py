"""
Hyperparameter search (Optuna) that is part of the training pipeline.

  * TPE sampler: random start, then concentrates on regions with low validation loss.
  * MedianPruner: stops trials that fall behind earlier ones at the same epoch.
  * Trial 0 is the untuned default config, so the benefit of tuning is measurable.
  * Every trial is a nested MLflow run (params, per-epoch metrics, final state, tags).
  * Objective = best validation loss. The test set is never touched here.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import optuna

import tracking
from mlp_core import DEFAULTS, build_mlp, fit, make_loaders, set_seed


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #
# (layer widths, batch sizes) per dataset. Each dataset's default config must lie inside these.
SEARCH_RANGES = {
    "mnist": ([64, 128, 256, 512, 1024], [64, 128, 256, 512]),
    "housing": ([32, 64, 128, 256, 512], [32, 64, 128, 256]),
    "tabular": ([32, 64, 128, 256, 512], [64, 128, 256, 512]),
}

def suggest_config(trial, dataset):
    """Map an Optuna trial (or FixedTrial) to a full training config."""
    cfg = dict(DEFAULTS[dataset])
    widths, batches = SEARCH_RANGES[dataset]

    n_layers = trial.suggest_int("n_layers", 1, 4)
    width = trial.suggest_categorical("width", widths)
    taper = trial.suggest_categorical("taper", [False, True])  # halve width at each layer
    hidden, w = [], width
    for _ in range(n_layers):
        hidden.append(max(w, 8))
        if taper:
            w //= 2
    cfg["hidden"] = hidden

    cfg["optimizer"] = trial.suggest_categorical("optimizer", ["sgd", "adamw"])
    if cfg["optimizer"] == "sgd":
        cfg["lr"] = trial.suggest_float("sgd_lr", 1e-3, 0.3, log=True)
        cfg["momentum"] = trial.suggest_float("momentum", 0.5, 0.99)
    else:
        cfg["lr"] = trial.suggest_float("adamw_lr", 1e-4, 1e-2, log=True)
    cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    cfg["batch_size"] = trial.suggest_categorical("batch_size", batches)
    cfg["gamma"] = trial.suggest_float("gamma", 0.85, 0.995)
    cfg["dropout"] = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
    return cfg


def default_trial_params(dataset):
    """The built-in default config in search-space terms (enqueued as trial 0)."""
    d = DEFAULTS[dataset]
    return {
        "n_layers": len(d["hidden"]), "width": d["hidden"][0], "taper": True,
        "optimizer": d["optimizer"], "sgd_lr": d["lr"], "momentum": d["momentum"],
        "weight_decay": d["weight_decay"], "batch_size": d["batch_size"],
        "gamma": d["gamma"], "dropout": d["dropout"],
    }


def config_from_params(params, dataset):
    return suggest_config(optuna.trial.FixedTrial(params), dataset)


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def trials_csv_text(study):
    import csv
    import io
    keys = sorted({k for t in study.trials for k in t.params})
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["trial", "state", "val_loss", "best_epoch", "seconds"] + keys)
    for t in study.trials:
        secs = t.duration.total_seconds() if t.duration else ""
        w.writerow([t.number, t.state.name, t.value, t.user_attrs.get("best_epoch", ""), secs]
                   + [t.params.get(k, "") for k in keys])
    return buf.getvalue()


def tuning_figure(study):
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].scatter([t.number for t in done], [t.value for t in done], label="completed")
    if pruned:
        ax[0].scatter([t.number for t in pruned], [min(t.value for t in done)] * len(pruned),
                      marker="x", color="gray", alpha=0.5, label="pruned (stopped early)")
    best, run = [], float("inf")
    for t in done:
        run = min(run, t.value)
        best.append(run)
    ax[0].plot([t.number for t in done], best, color="tab:red", label="best so far")
    ax[0].set(title="Validation loss per trial", xlabel="trial", ylabel="validation loss", yscale="log")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    try:
        if len(done) >= 10:
            imp = optuna.importance.get_param_importances(study)
            ax[1].barh(list(imp.keys())[::-1], list(imp.values())[::-1])
            ax[1].set(title="Hyperparameter importance", xlabel="relative importance")
            ax[1].grid(alpha=0.3)
        else:
            ax[1].axis("off")
            ax[1].text(0.1, 0.5, "importance needs >= 10 completed trials")
    except Exception as e:  # importance is best-effort
        ax[1].axis("off")
        ax[1].text(0.05, 0.5, f"importance unavailable:\n{e}", fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# The search (call inside the active MLflow parent run)
# --------------------------------------------------------------------------- #
def run_search(dataset, search_splits, info, device, *, trials, timeout, trial_epochs,
               trial_patience, startup_trials, seed, study_dir, resume, log=print):
    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    db = study_dir / "study.db"
    if db.exists() and not resume:
        db.unlink()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=f"{dataset}_mlp", storage=f"sqlite:///{db}", load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=startup_trials),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    if len(study.trials) == 0:
        study.enqueue_trial(default_trial_params(dataset))  # trial 0 = untuned defaults

    def objective(trial):
        cfg = suggest_config(trial, dataset)
        cfg["epochs"], cfg["patience"] = trial_epochs, trial_patience
        set_seed(seed + trial.number)
        mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True, tags={
            "phase": "search_trial", "optuna_trial": str(trial.number),
            "is_default_baseline": str(trial.number == 0 and not resume).lower()})
        status, state = "FINISHED", "COMPLETE"
        try:
            mlflow.log_params({**tracking.flat_params(cfg), "search_train_rows": len(search_splits["train"])})
            model = build_mlp(info["in_dim"], cfg["hidden"], info["out_dim"], cfg["dropout"]).to(device)
            res = fit(model, make_loaders(search_splits, cfg["batch_size"]), cfg, info, device,
                      trial=trial, on_epoch=tracking.log_epoch, verbose=False)
            if res["diverged"]:
                state, status = "DIVERGED", "KILLED"
                raise optuna.TrialPruned()
            trial.set_user_attr("best_epoch", res["best_epoch"])
            mlflow.log_metrics({"best_val_loss": res["best_val_loss"], "best_epoch": res["best_epoch"]})
            return res["best_val_loss"]
        except optuna.TrialPruned:
            if state == "COMPLETE":
                state, status = "PRUNED", "KILLED"
            raise
        except BaseException:
            state, status = "FAILED", "FAILED"
            raise
        finally:
            mlflow.set_tag("optuna_state", state)
            mlflow.end_run(status)

    def log_trial(study_, t):
        shown = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in t.params.items()}
        val = f"{t.value:.4f}" if t.value is not None else "  -   "
        log(f"trial {t.number:3d} {t.state.name:8s} val_loss {val} | {shown}")

    # An enqueued (WAITING) default trial is run as part of the n_trials below, so only
    # finished trials count toward the budget.
    finished = sum(t.state != optuna.trial.TrialState.WAITING for t in study.trials)
    remaining = max(0, trials - finished)
    log(f"search: {finished} finished trials, running {remaining} more "
        f"(<= {trial_epochs} epochs each, pruning on)")
    if remaining:
        study.optimize(objective, n_trials=remaining, timeout=timeout, callbacks=[log_trial])

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError("No search trial completed; check the data and trial settings.")
    best = study.best_trial
    base = study.trials[0]
    return {
        "study": study, "best_trial": best.number, "best_value": best.value,
        "best_params": best.params, "best_cfg": config_from_params(best.params, dataset),
        "baseline_value": base.value if (base.number == 0 and base.value is not None) else None,
        "n_trials": len(study.trials),
        "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
    }
