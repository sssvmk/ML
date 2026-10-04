"""7.4  Dataset augmentation: train on randomly shifted / rotated / zoomed copies of each image.

Class-preserving transforms only: rotations are limited (a 180-degree rotation would turn 6
into 9) and there are no flips.  Applied to training batches only, never to validation/test."""
from ..transforms import random_affine
from .base import Method


class Augmentation(Method):
    name = "augmentation"
    title = "Dataset augmentation"
    section = "7.4"

    def defaults(self):
        return {"lr": 0.05, "max_shift": 2.0, "max_rot": 10.0, "max_scale": 0.1}

    def space(self, trial):
        return {"max_shift": trial.suggest_float("max_shift", 0.0, 4.0),
                "max_rot": trial.suggest_float("max_rot", 0.0, 25.0),
                "max_scale": trial.suggest_float("max_scale", 0.0, 0.25)}

    def transform(self, batch, model, hp):
        return {**batch, "x": random_affine(batch["x"], hp["max_shift"], hp["max_rot"], hp["max_scale"])}
