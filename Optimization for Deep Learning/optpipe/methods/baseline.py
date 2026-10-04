"""Reference: ReLU MLP, He-normal weights, zero biases, SGD + momentum 0.9, exponential lr decay,
L1 + L2 penalties.  Every method below changes ONE thing relative to this."""
from .base import Method


class Baseline(Method):
    name = "baseline"
    title = "Baseline: SGD + momentum + exponential decay, He init"
    section = "reference"
    group = "reference"
