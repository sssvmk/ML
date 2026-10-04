"""7.5  Noise robustness, (a) noise on the inputs.

Adding zero-mean Gaussian noise of variance sigma^2 to the inputs is equivalent, for small
sigma, to a penalty on the sensitivity of the output to the input.  Training only."""
import torch

from .base import Method


def add_input_noise(x, sigma: float):
    return x + sigma * torch.randn_like(x)


class InputNoise(Method):
    name = "input_noise"
    title = "Noise on inputs"
    section = "7.5 (inputs)"

    def defaults(self):
        return {"lr": 0.05, "sigma": 0.1}

    def space(self, trial):
        return {"sigma": trial.suggest_float("sigma", 0.01, 0.6, log=True)}

    def transform(self, batch, model, hp):
        return {**batch, "x": add_input_noise(batch["x"], hp["sigma"])}
