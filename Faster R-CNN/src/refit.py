"""Early-stopping meta-algorithm (Deep Learning book Alg. 7.2): after early stopping chose the epoch count on the
validation split, retrain from scratch on ALL of trainval for that many epochs, with no validation split."""
from __future__ import annotations

import copy

import mlflow
from mlflow.tracking import MlflowClient

from src.train import run_training


def run_refit(cfg: dict, env: dict, source_run_id: str, config_path: str | None = None) -> dict:
    mlflow.set_tracking_uri(env["tracking_uri"])
    src = MlflowClient().get_run(source_run_id)
    if "best_epoch" not in src.data.tags:
        raise RuntimeError(f"run {source_run_id} has no best_epoch tag; run `train` first")
    epochs = int(src.data.tags["best_epoch"]) + 1
    c = copy.deepcopy(cfg)
    c["phase"] = "refit"
    c["data"]["val_fraction"] = 0.0
    c["training"]["max_epochs"] = epochs
    c["training"]["early_stopping"]["enabled"] = False
    c["sanity"]["enabled"] = False  # already passed in the source run
    print(f"refit: training on all of trainval for {epochs} epochs (best epoch of {source_run_id} + 1)")
    return run_training(c, env, config_path, extra_tags={"refit_of": source_run_id, "refit_epochs": str(epochs)})
