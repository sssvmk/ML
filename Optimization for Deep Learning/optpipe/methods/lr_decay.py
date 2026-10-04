"""8.3.1  Learning-rate decay: plain SGD with a schedule.

  linear      e_k = (1-a) e_0 + a e_tau,  a = k/tau, then constant e_tau          (book eq. 8.14, the recommended form)
  exponential e_k = e_0 * gamma^epoch
  inverse     e_k = e_0 / (1 + d * epoch)
  step        halve every `step_epochs` epochs
The schedule shape, tau and the final fraction e_tau/e_0 are tuned.  Compare with `sgd` (constant lr)."""
from ..optim import SGD as SGDOpt
from .base import Method


class LRDecay(Method):
    name = "lr_decay"
    title = "Learning-rate decay (schedule within SGD)"
    section = "8.3.1"
    group = "update rules"

    def defaults(self):
        return {**super().defaults(), "schedule": "linear", "tau_frac": 0.8, "final_frac": 0.01, "gamma": 0.9,
                "d": 0.5, "step_epochs": 5}

    def space(self, trial):
        sched = trial.suggest_categorical("schedule", ["linear", "exponential", "inverse", "step"])
        hp = {"schedule": sched}
        if sched == "linear":
            hp["tau_frac"] = trial.suggest_float("tau_frac", 0.3, 1.0)
            hp["final_frac"] = trial.suggest_float("final_frac", 1e-3, 0.3, log=True)
        elif sched == "exponential":
            hp["gamma"] = trial.suggest_float("gamma", 0.7, 0.99)
        elif sched == "inverse":
            hp["d"] = trial.suggest_float("d", 0.05, 2.0, log=True)
        else:
            hp["step_epochs"] = trial.suggest_int("step_epochs", 1, 10)
        return hp

    def make_optimizer(self, model, hp, ctx):
        return SGDOpt(model.parameters(), lr=hp["lr"], momentum=0.0)

    def lr_factor(self, t, hp, ctx):
        s = hp["schedule"]
        if s == "linear":
            tau = max(hp["tau_frac"] * ctx.epochs, 1e-9)
            a = min(t / tau, 1.0)
            return (1 - a) + a * hp["final_frac"]
        if s == "exponential":
            return hp["gamma"] ** t
        if s == "inverse":
            return 1.0 / (1.0 + hp["d"] * t)
        return 0.5 ** int(t // hp["step_epochs"])
