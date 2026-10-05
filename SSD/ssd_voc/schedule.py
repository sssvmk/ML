"""Iteration-based LR phases (the paper's step schedule), linear warmup, and phase-aware early stopping."""
import math


class PhaseSchedule:
    """Paper recipe: constant LR per phase (1e-3, then 1e-4, ...), measured in iterations. A phase can also be
    ended early (plateau) via advance(). Per-group `lr_mult` is respected."""

    def __init__(self, optimizer, phases, scale=1.0, warmup_iters=0, warmup_factor=0.1):
        self.opt = optimizer
        self.phases = [(float(lr), max(1, int(round(n * scale)))) for lr, n in phases]
        self.warmup_iters, self.warmup_factor = int(warmup_iters), float(warmup_factor)
        self.phase, self.it_in_phase, self.global_it = 0, 0, 0
        for g in optimizer.param_groups:
            g.setdefault("lr_mult", 1.0)

    @property
    def total_iters(self) -> int:
        return sum(n for _, n in self.phases)

    @property
    def is_last_phase(self) -> bool:
        return self.phase == len(self.phases) - 1

    def phase_done(self) -> bool:
        return self.it_in_phase >= self.phases[self.phase][1]

    def lr_now(self) -> float:
        base = self.phases[self.phase][0]
        if self.phase == 0 and self.global_it < self.warmup_iters:
            base *= self.warmup_factor + (1.0 - self.warmup_factor) * self.global_it / self.warmup_iters
        return base

    def step(self) -> float:
        lr = self.lr_now()
        for g in self.opt.param_groups:
            g["lr"] = lr * g["lr_mult"]
        self.global_it += 1
        self.it_in_phase += 1
        return lr

    def advance(self) -> bool:
        """Move to the next phase; False if this was the last one."""
        if self.is_last_phase:
            return False
        self.phase += 1
        self.it_in_phase = 0
        return True

    def state_dict(self):
        return {"phase": self.phase, "it_in_phase": self.it_in_phase, "global_it": self.global_it}

    def load_state_dict(self, s):
        self.phase, self.it_in_phase, self.global_it = s["phase"], s["it_in_phase"], s["global_it"]


class PhaseEarlyStopper:
    """Tracks the best validation metric (higher is better) and the number of evaluations without improvement in the
    CURRENT phase. `exhausted` tells the trainer to take the next LR step early or, in the last phase, to stop."""

    def __init__(self, patience: int, min_delta: float = 0.0, enabled: bool = True):
        self.patience, self.min_delta, self.enabled = int(patience), float(min_delta), bool(enabled)
        self.best, self.best_it, self.bad = -math.inf, -1, 0

    def update(self, value: float, iteration: int) -> bool:
        """Returns True when `value` is a new best."""
        if value > self.best + self.min_delta:
            self.best, self.best_it, self.bad = value, iteration, 0
            return True
        self.bad += 1
        return False

    @property
    def exhausted(self) -> bool:
        return self.enabled and self.bad >= self.patience

    def reset_phase(self) -> None:
        self.bad = 0

    def state_dict(self):
        return {"best": self.best, "best_it": self.best_it, "bad": self.bad}

    def load_state_dict(self, s):
        self.best, self.best_it, self.bad = s["best"], s["best_it"], s["bad"]
