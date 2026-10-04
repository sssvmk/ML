"""Algorithm 8.4 AdaGrad:  r <- r + g*g;  theta <- theta - e / (delta + sqrt(r)) * g   (delta = 1e-7)."""
import torch

from .base import BookOptimizer, params_with_grad


class AdaGrad(BookOptimizer):
    def __init__(self, params, lr=0.05, delta=1e-7):
        super().__init__(params, dict(lr=lr, delta=delta))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in params_with_grad(group):
                st = self.state[p]
                r = st.get("r")
                if r is None:
                    r = st["r"] = torch.zeros_like(p)
                r.addcmul_(p.grad, p.grad)
                p.addcdiv_(p.grad, r.sqrt().add_(group["delta"]), value=-group["lr"])
