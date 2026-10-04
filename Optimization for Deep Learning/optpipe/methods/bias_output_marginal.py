"""8.4 (12.9)  Output bias set to the training-data marginal statistics: solve softmax(b) = c, where c is the vector of
class frequencies, so the network starts out predicting the class prior.  The book assumes the initial weights are small
enough that the output is determined by the bias, so the output weights are shrunk by `out_weight_scale`.
`strength` interpolates between b = 0 (0) and b = log c (1)."""
import torch

from .base import Method


class BiasOutputMarginal(Method):
    name = "bias_output_marginal"
    title = "Output bias = training-data marginal (softmax(b) = class frequencies)"
    section = "8.4 (12.9)"
    group = "initialization"

    def defaults(self):
        return {**super().defaults(), "strength": 1.0, "out_weight_scale": 0.1}

    def space(self, trial):
        return {"strength": trial.suggest_float("strength", 0.0, 1.0),
                "out_weight_scale": trial.suggest_float("out_weight_scale", 0.01, 1.0, log=True)}

    @torch.no_grad()
    def initialize(self, model, hp, ctx, data):
        freq = torch.bincount(data.y_train, minlength=10).float()
        freq = freq / freq.sum()
        logc = torch.log(freq.clamp_min(1e-12))
        model.out.bias.copy_(hp["strength"] * (logc - logc.mean()))     # softmax is shift-invariant
        model.out.weight.mul_(hp["out_weight_scale"])
        entropy = float(-(freq * torch.log(freq.clamp_min(1e-12))).sum())
        return {"class_prior_entropy": entropy}
