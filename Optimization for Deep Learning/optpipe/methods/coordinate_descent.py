"""8.7.2  Coordinate descent, in its block form: only ONE block of parameters (one layer: weights + bias) is updated at a time,
the others stay exactly where they are, and the active block cycles.  Each block has its own SGD+momentum inner solver.
`cycle` = epochs spent on a block before moving to the next.  (Exact per-coordinate minimisation is not available for a
neural network; block coordinate descent with an inner minibatch solver is the practical form.)"""
from ..optim import BlockCoordinateSGD
from .base import Method


class CoordinateDescent(Method):
    name = "coordinate_descent"
    title = "Block coordinate descent (one layer at a time)"
    section = "8.7.2"
    group = "meta-algorithms"

    def defaults(self):
        return {**super().defaults(), "cycle": 1}

    def space(self, trial):
        return {"cycle": trial.suggest_int("cycle", 1, 3)}

    def make_optimizer(self, model, hp, ctx):
        return BlockCoordinateSGD(model.blocks(), lr=hp["lr"], momentum=ctx.momentum)

    def epoch_start(self, model, opt, hp, epoch, state):
        opt.set_active((epoch - 1) // hp["cycle"])
        state["order"] = state.get("order", []) + [opt.active]

    def final_stats(self, model, state):
        return {"block_schedule": state.get("order", [])}
