"""8.4 (12.12)  Variance / precision parameters set to 1 or to the marginal variance.
Model: p(y | x) = N(y | f(x), 1/beta) on the one-hot target vector, with a LEARNED precision beta (log-parameterised).
Loss = beta/2 * ||y - f(x)||^2 - (K/2) log beta.   Predicted class probabilities = softmax(beta * f(x)).
beta is initialised either to 1 ("one") or to 1 / marginal variance of the targets ("marginal", mean_k c_k(1-c_k)).
NOTE: this method trains a Gaussian-output model, not cross-entropy; accuracy/AUC are still evaluated on its softmax outputs."""
import math

import torch
import torch.nn.functional as F

from ..model import MLP
from .base import Method


class VariancePrecision(Method):
    name = "variance_precision"
    title = "Variance/precision parameter init (Gaussian output, learned precision)"
    section = "8.4 (12.12)"
    group = "initialization"
    regime = "Gaussian-output model (squared error + learned precision)"
    lr_range = (1e-4, 3e-2)         # the loss gradient is scaled by beta (~11 at the marginal init): small steps needed

    def defaults(self):
        return {**super().defaults(), "lr": 1e-3, "precision_init": "marginal"}

    def space(self, trial):
        return {"precision_init": trial.suggest_categorical("precision_init", ["one", "marginal"])}

    def build(self, hp, ctx):
        return MLP(ctx.hidden, gaussian_output=True, init=ctx.init)

    @torch.no_grad()
    def initialize(self, model, hp, ctx, data):
        freq = torch.bincount(data.y_train, minlength=10).float()
        freq = freq / freq.sum()
        marginal_var = float((freq * (1 - freq)).mean())
        model.log_beta.fill_(0.0 if hp["precision_init"] == "one" else -math.log(marginal_var))
        return {"marginal_variance": marginal_var, "beta_init": float(torch.exp(model.log_beta))}

    def data_loss(self, model, batch, hp):
        f = model.mean_output(batch["x"])
        y1h = F.one_hot(batch["y"], f.shape[1]).float()
        beta = torch.exp(model.log_beta)
        return (0.5 * beta * ((y1h - f) ** 2).sum(1)).mean() - 0.5 * f.shape[1] * model.log_beta

    def final_stats(self, model, state):
        return {"final_precision": float(torch.exp(model.log_beta))}
