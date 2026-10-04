"""7.12  Dropout.

Each training step multiplies every input/hidden unit by an independent Bernoulli mask (inverted
dropout: survivors are scaled by 1/(1-p), so no rescaling is needed at test time; model.eval()
switches it off).  Trains an exponentially large ensemble of weight-sharing sub-networks."""
from ..model import MLP
from .base import Method


class Dropout(Method):
    name = "dropout"
    title = "Dropout"
    section = "7.12"

    def defaults(self):
        return {"lr": 0.05, "p_in": 0.1, "p_hidden": 0.3}

    def space(self, trial):
        return {"p_in": trial.suggest_float("p_in", 0.0, 0.3),
                "p_hidden": trial.suggest_float("p_hidden", 0.0, 0.7)}

    def build(self, hp, ctx):
        return MLP(ctx.hidden, p_in=hp["p_in"], p_hidden=hp["p_hidden"], init=ctx.init)
