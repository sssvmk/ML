"""Reference: the plain ReLU MLP with no regularizer.  Every method is compared to this."""
from .base import Method


class Baseline(Method):
    name = "baseline"
    title = "Baseline (no regularization)"
    section = "reference"
