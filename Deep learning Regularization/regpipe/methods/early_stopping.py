"""7.8  Early stopping.

Evaluate on the validation set every epoch, remember the best weights, stop after `patience`
epochs without improvement of the validation loss and return the BEST weights, not the last.
(Every other method in this pipeline returns last-epoch weights, so this is the only one that
selects an epoch.)  For a quadratic loss this is equivalent to L2 regularization, with
epochs * learning-rate playing the role of 1/alpha."""
from .base import Method


class EarlyStopping(Method):
    name = "early_stopping"
    title = "Early stopping"
    section = "7.8"
    selection = "best"

    def defaults(self):
        return {"lr": 0.05, "patience": 5}

    def space(self, trial):
        return {"patience": trial.suggest_int("patience", 2, 10)}

    def should_stop(self, history, hp):
        losses = [h["val_loss"] for h in history]
        best_epoch = min(range(len(losses)), key=losses.__getitem__)
        return len(losses) - 1 - best_epoch >= hp["patience"]
