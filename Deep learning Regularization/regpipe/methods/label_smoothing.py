"""7.5.1  Noise on the output targets: label smoothing.

Hard 0/1 targets make a softmax classifier chase ever larger weights; replacing them with
1-eps on the true class and eps/(K-1) elsewhere stops that without discouraging correct
classification.  Validation/test losses stay plain hard-label cross-entropy."""
from .base import Method, soft_cross_entropy


class LabelSmoothing(Method):
    name = "label_smoothing"
    title = "Noise on output targets (label smoothing)"
    section = "7.5.1"

    def defaults(self):
        return {"lr": 0.05, "eps": 0.1}

    def space(self, trial):
        return {"eps": trial.suggest_float("eps", 0.01, 0.3)}

    def data_loss(self, model, batch, hp):
        return soft_cross_entropy(model(batch["x"]), batch["y"], hp["eps"])
