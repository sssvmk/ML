"""8.6.2  Conjugate gradients (nonlinear CG, Algorithm 8.9): each new direction is conjugate to the previous ones,
d_t = -g_t + beta_t d_{t-1}, with a strong-Wolfe line search along it.

  beta variants   PR+ (Polak-Ribiere, clipped at 0 -- the book's choice plus the usual safeguard)   |   FR (Fletcher-Reeves)
  restart         beta = 0 every `restart` iterations or when the direction stops being a descent direction"""
import torch

from ..fullbatch import Stepper
from ..linesearch import strong_wolfe
from .base import Method


class NonlinearCGStepper(Stepper):
    def __init__(self, variant="PR+", restart=50, c2=0.1):
        self.variant, self.restart, self.c2 = variant, int(restart), c2
        self.f = self.g = self.d = None
        self.k = 0
        self.alpha_prev = self.dphi_prev = None
        self.resets = 0

    def _beta(self, g_new, g):
        if self.variant == "FR":
            return float(g_new @ g_new) / float(g @ g)
        return max(0.0, float((g_new - g) @ g_new) / float(g @ g))

    def step(self, prob):
        if self.d is None:
            self.f, self.g = prob.value_and_grad()
            self.d = -self.g
        theta = prob.get()
        for attempt in range(2):
            d, g, f = self.d, self.g, self.f
            dphi0 = float(g @ d)
            if dphi0 >= 0:
                d, dphi0 = -g, -float(g @ g)
                self.resets += 1

            def phi(a, d=d):
                f_, g_ = prob.value_and_grad(theta + a * d)
                return f_, float(g_ @ d), g_

            if self.alpha_prev is None:
                a0 = min(1.0, 1.0 / max(float(g.abs().sum()), 1e-12))
            else:
                a0 = min(10.0, max(1e-3, self.alpha_prev * self.dphi_prev / dphi0))
            res = strong_wolfe(phi, f, dphi0, a0, c2=self.c2, alpha_max=20.0, max_iter=12)
            if res is not None:
                a, f_new, g_new, _ = res
                prob.set(theta + a * d)
                self.note_progress(f, f_new)
                self.note_gradient(g_new)
                self.k += 1
                beta = 0.0 if self.k % self.restart == 0 else self._beta(g_new, g)
                self.d = -g_new + beta * d
                self.f, self.g = f_new, g_new
                self.alpha_prev, self.dphi_prev = a, dphi0
                return {"loss": f_new, "alpha": a}
            self.d = -g                                   # line search failed: restart from steepest descent
            self.resets += 1
        prob.set(theta)
        self.done = True
        return {"loss": self.f, "alpha": 0.0}

    def summary(self):
        return {"cg_restarts": self.resets}


class ConjugateGradients(Method):
    name = "conjugate_gradients"
    title = "Conjugate gradients (nonlinear, PR+/FR)"
    section = "8.6.2"
    group = "second-order"
    mode = "fullbatch"
    tune_lr = False

    def defaults(self):
        d = super().defaults()
        d.pop("lr")
        return {**d, "variant": "PR+", "restart": 50, "c2": 0.1}

    def space(self, trial):
        return {"variant": trial.suggest_categorical("variant", ["PR+", "FR"]),
                "restart": trial.suggest_int("restart", 10, 200),
                "c2": trial.suggest_float("c2", 0.05, 0.5)}

    def make_stepper(self, model, hp, ctx):
        return NonlinearCGStepper(hp["variant"], hp["restart"], hp["c2"])
