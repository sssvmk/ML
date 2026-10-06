"""Algorithm registry. Each algorithm lives in its own module, named exactly like the algorithm, and exposes `Algorithm`."""
from __future__ import annotations

import importlib

from .base import AlgorithmModule, AlgorithmResult, ExperimentData  # noqa: F401

KNOWN = [
    "majority_class_baseline",
    "logistic_regression_elasticnet",
    "random_forest_bagged_trees",
    "xgboost_gradient_boosted_trees",
    "lightgbm_gradient_boosted_trees",
    "pytorch_residual_mlp",
]


def load_algorithm(name: str) -> AlgorithmModule:
    module = importlib.import_module(f"airsat.algorithms.{name}")
    return module.Algorithm()
