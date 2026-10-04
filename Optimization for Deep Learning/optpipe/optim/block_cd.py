"""Block coordinate descent optimizer (section 8.7.2): only ONE block of parameters is updated at a time
(each block has its own SGD+momentum state); the active block cycles."""
import torch

from .sgd import SGD


class BlockCoordinateSGD:
    def __init__(self, blocks, lr=0.05, momentum=0.9):
        self.blocks = [SGD(b, lr=lr, momentum=momentum) for b in blocks]
        self.active = 0
        self.params = [p for b in blocks for p in b]

    @property
    def param_groups(self):
        return [g for o in self.blocks for g in o.param_groups]

    def set_active(self, i: int):
        self.active = i % len(self.blocks)

    def prepare(self):
        return None

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None if set_to_none else torch.zeros_like(p)

    def step(self):
        self.blocks[self.active].step()                # the other blocks stay exactly where they are
