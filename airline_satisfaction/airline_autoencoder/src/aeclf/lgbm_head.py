"""LightGBM on top of the pretrained encoder (tree-based classification head).

Three feature sets, all fed to the same LightGBM learner:
  lgbm_latent  latent code only            -> still a pure encoder-based classifier (the task's definition)
  lgbm_hybrid  latent code + raw encoded columns (standardised continuous + categorical codes)  -> upper bound of what the encoder can add
  lgbm_raw     raw encoded columns only (NO encoder)  -> control: does the latent code add anything to a strong tabular model?
The encoder is frozen here; trees early-stop on a 10% hold-out of TRAIN so validation stays clean for model selection.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from .training import Tensors

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

def require_lightgbm():
    if lgb is None:
        raise ImportError("lightgbm is not installed in this environment. Install it with `uv pip install lightgbm` "
                          "(or `pip install lightgbm`), or remove lgbm_latent / lgbm_hybrid from classifier.modes and set "
                          "classifier.include_lgbm_raw_control=false.")


LGBM_MODES = ("lgbm_latent", "lgbm_hybrid", "lgbm_raw")
LGBM_KEYS = ["learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree", "reg_lambda", "reg_alpha"]


def lgbm_space(trial) -> dict:
    return {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 127, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)}


@torch.no_grad()
def build_features(ae, t: Tensors, mode: str, batch: int = 16384) -> np.ndarray:
    parts = []
    if mode in ("lgbm_latent", "lgbm_hybrid"):
        ae.eval()
        parts.append(np.concatenate([ae.encode(t.xc[i:i + batch], t.xk[i:i + batch])[0].numpy() for i in range(0, len(t), batch)]))
    if mode in ("lgbm_hybrid", "lgbm_raw"):
        parts += [t.xc.numpy(), t.xk.numpy().astype(np.float32)]
    return np.concatenate(parts, axis=1).astype(np.float32)


def train_lgbm(params: dict, X: np.ndarray, y: np.ndarray, seed: int):
    """Returns (fitted LGBMClassifier, history DataFrame[iteration, train_loss, val_loss])."""
    require_lightgbm()
    Xa, Xe, ya, ye = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)
    m = lgb.LGBMClassifier(objective="binary", n_estimators=1500, subsample_freq=1, n_jobs=-1, random_state=seed, verbosity=-1, **params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Xa, ya, eval_set=[(Xa, ya), (Xe, ye)], eval_names=["fit", "es"], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    log = m.evals_result_
    hist = pd.DataFrame({"iteration": range(len(log["fit"]["binary_logloss"])), "train_loss": log["fit"]["binary_logloss"], "val_loss": log["es"]["binary_logloss"]})
    return m, hist


def leaf_count_proxy(model, params: dict) -> int:
    """Complexity proxy for tie-breaking: trees x max leaves (upper bound; not comparable one-to-one with NN weights)."""
    return int(model.booster_.num_trees() * params["num_leaves"])
