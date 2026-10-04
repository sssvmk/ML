"""7.9  Parameter tying and parameter sharing.

input layer -> `depth` applications of one width x width block -> output layer.
  hard tying : a single block is reused `depth` times (the parameters are the same tensor)
  soft tying : `depth` separate blocks with a penalty  lambda * sum_i ||W_i - mean(W)||^2
The tuner chooses between the two.  Reported diagnostic: parameter count vs the untied model."""
from ..model import TiedDepthMLP, n_params
from .base import Method


class ParameterSharing(Method):
    name = "parameter_sharing"
    title = "Parameter tying and parameter sharing"
    section = "7.9"

    def defaults(self):
        return {"lr": 0.05, "mode": "hard", "depth": 3, "lam_tie": 0.1}

    def space(self, trial):
        mode = trial.suggest_categorical("mode", ["hard", "soft"])
        hp = {"mode": mode, "depth": trial.suggest_int("depth", 2, 6)}
        if mode == "soft":
            hp["lam_tie"] = trial.suggest_float("lam_tie", 1e-3, 10.0, log=True)
        return hp

    def build(self, hp, ctx):
        return TiedDepthMLP(width=ctx.hidden[0], depth=hp["depth"], mode=hp["mode"], init=ctx.init)

    def penalty(self, model, batch, hp):
        return hp.get("lam_tie", 0.0) * model.tying_penalty() if hp["mode"] == "soft" else 0.0

    def extras(self, result, data, ctx, hp):
        width = ctx.hidden[0]
        return {"n_params": n_params(result.model),
                "n_params_untied_same_depth": n_params(TiedDepthMLP(width, hp["depth"], "soft"))}
