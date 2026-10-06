"""Floor reference: always predicts the training prior. Any real model must beat this by a clear margin;
if it does not, the features carry no signal or the pipeline is broken (Goodfellow et al., Ch. 11)."""
from __future__ import annotations

from sklearn.dummy import DummyClassifier

from ..assumptions import check_class_balance, check_sample_size
from .base import AlgorithmModule


class Algorithm(AlgorithmModule):
    name = "majority_class_baseline"
    display_name = "Majority-class baseline"
    family = "baseline"
    description = "Predicts the class prior for every passenger. Defines the floor: ROC-AUC 0.5, accuracy = majority share."
    complexity_rank = 0
    loss_function = "None (no learning). Log loss of a constant prior prediction is reported for reference."
    optimisation_notes = "Nothing to optimise."
    searchable = False
    scale_inputs = False
    nominal_encoding = "ordinal"

    def default_params(self) -> dict:
        return {}

    def build_estimator(self, params, seed):
        return DummyClassifier(strategy="prior", random_state=seed)

    def check_assumptions(self, ctx):
        return [check_class_balance(ctx), check_sample_size(ctx)]
