"""7.5  Noise robustness, (b) noise on the weights.

Each step the weights are perturbed with Gaussian noise, the gradient is computed at the
perturbed point, the weights are restored, and the update is applied to the clean weights.
Pushes the model toward flat minima (regions insensitive to small weight changes)."""
import contextlib

import torch

from .base import Method, weights_only


@contextlib.contextmanager
def perturbed_weights(model, sigma: float):
    ws = weights_only(model)
    originals = [w.detach().clone() for w in ws]
    with torch.no_grad():
        for w in ws:
            w.add_(torch.randn_like(w) * sigma)
    try:
        yield
    finally:
        with torch.no_grad():
            for w, o in zip(ws, originals):
                w.copy_(o)


class WeightNoise(Method):
    name = "weight_noise"
    title = "Noise on weights"
    section = "7.5 (weights)"

    def defaults(self):
        return {"lr": 0.05, "sigma": 0.01}

    def space(self, trial):
        return {"sigma": trial.suggest_float("sigma", 1e-3, 5e-2, log=True)}

    def forward_context(self, model, hp):
        return perturbed_weights(model, hp["sigma"])
