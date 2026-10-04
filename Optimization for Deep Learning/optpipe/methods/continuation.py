"""8.7.6  Continuation methods: optimise a sequence of cost functions J^(0), J^(1), ..., J^(n)=J, the first very smooth/easy, each
used as the starting point for the next.  Here the cost is smoothed by BLURRING THE INPUTS with a Gaussian of width sigma_e,
annealed linearly from sigma0 to 0 over the first `anneal_frac` of training; the last epochs optimise the true cost on sharp images.
Validation/test always use sharp images, so early validation numbers show the shift."""
from ..transforms import gaussian_blur
from .base import Method


class Continuation(Method):
    name = "continuation"
    title = "Continuation method (annealed input blur)"
    section = "8.7.6 (a)"
    group = "meta-algorithms"

    def defaults(self):
        return {**super().defaults(), "sigma0": 1.5, "anneal_frac": 0.6}

    def space(self, trial):
        return {"sigma0": trial.suggest_float("sigma0", 0.3, 3.0), "anneal_frac": trial.suggest_float("anneal_frac", 0.2, 0.9)}

    @staticmethod
    def sigma(hp, epoch: int, total: int) -> float:
        T = max(1.0, hp["anneal_frac"] * total)
        return hp["sigma0"] * max(0.0, 1.0 - (epoch - 1) / T)

    def transform(self, batch, hp, epoch, ctx):
        return {**batch, "x": gaussian_blur(batch["x"], self.sigma(hp, epoch, ctx.epochs))}
