"""8.4 (12.8)  Zero biases: compatible with most weight schemes.  Weights are He-normal, every bias exactly 0.
(Identical to the baseline by construction; it is the reference for the bias/parameter group.)"""
import torch

from ._init_utils import zero_biases
from .base import Method


class BiasZero(Method):
    name = "bias_zero"
    title = "Zero biases"
    section = "8.4 (12.8)"
    group = "initialization"

    def initialize(self, model, hp, ctx, data):
        zero_biases(model)
