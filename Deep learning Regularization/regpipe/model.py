"""Models.  Every model maps RAW pixels in [0,1] (shape [B,784]) to class logits.

Standardisation is a layer inside the model, so serving never has to repeat preprocessing.
A model is fully described by a JSON-serialisable `spec`, which is what bundle.py stores
next to the weights so the inference system can rebuild it without any training code.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import DIM, N_CLASSES

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(MNIST_MEAN))
        self.register_buffer("std", torch.tensor(MNIST_STD))

    def forward(self, x):
        return (x - self.mean) / self.std


def init_linear(m: nn.Module, scheme: str = "he_normal") -> None:
    """Initialise an nn.Linear.  he_normal (weights) + zero bias suits ReLU units."""
    if not isinstance(m, nn.Linear):
        return
    if scheme == "he_normal":
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)
    elif scheme == "default":
        m.reset_parameters()  # PyTorch's own nn.Linear default
    else:
        raise ValueError(f"unknown init {scheme!r}")


class MLP(nn.Module):
    """Fully connected network with ReLU hidden units.

    Optional pieces used by individual methods:
      p_in / p_hidden : dropout rates                                   (dropout)
      aux_heads       : extra classification heads on the last hidden   (multitask)
      with_decoder    : reconstruction head sharing the encoder         (semi-supervised)
    """

    def __init__(self, hidden=(256, 128), p_in=0.0, p_hidden=0.0, init="he_normal",
                 aux_heads: dict | None = None, with_decoder: bool = False):
        super().__init__()
        self.spec = dict(arch="mlp", hidden=list(hidden), p_in=p_in, p_hidden=p_hidden, init=init,
                         aux_heads=dict(aux_heads or {}), with_decoder=with_decoder)
        self.norm = Normalize()
        self.drop_in = nn.Dropout(p_in)
        dims = [DIM, *hidden]
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.drop = nn.ModuleList(nn.Dropout(p_hidden) for _ in hidden)
        self.out = nn.Linear(dims[-1], N_CLASSES)
        self.aux = nn.ModuleDict({k: nn.Linear(dims[-1], n) for k, n in (aux_heads or {}).items()})
        self.decoder = nn.Linear(dims[-1], DIM) if with_decoder else None
        self.apply(lambda m: init_linear(m, init))

    # -- internals
    def encode(self, x, return_hidden=False):
        h = self.drop_in(self.norm(x))
        acts = []
        for lin, drop in zip(self.hidden, self.drop):
            h = drop(F.relu(lin(h)))
            acts.append(h)
        return (h, acts) if return_hidden else h

    def forward(self, x, return_hidden=False):
        h, acts = self.encode(x, return_hidden=True)
        logits = self.out(h)
        return (logits, acts) if return_hidden else logits

    def forward_all(self, x) -> dict:
        h = self.encode(x)
        out = {"logits": self.out(h)}
        out.update({f"aux_{k}": head(h) for k, head in self.aux.items()})
        if self.decoder is not None:
            out["recon"] = torch.sigmoid(self.decoder(h))
        return out

    def weight_matrices(self):
        return [m.weight for m in [*self.hidden, self.out]]


class TiedDepthMLP(nn.Module):
    """Parameter tying/sharing across depth.

    input layer -> `depth` applications of a width x width block -> output layer.
      mode="hard": ONE block reused `depth` times (parameters are literally shared)
      mode="soft": `depth` separate blocks pulled together by a penalty on their distance
    """

    def __init__(self, width=256, depth=3, mode="hard", init="he_normal"):
        super().__init__()
        assert mode in ("hard", "soft")
        self.spec = dict(arch="tied", width=width, depth=depth, mode=mode, init=init)
        self.mode, self.depth = mode, depth
        self.norm = Normalize()
        self.inp = nn.Linear(DIM, width)
        self.blocks = nn.ModuleList(nn.Linear(width, width) for _ in range(1 if mode == "hard" else depth))
        self.out = nn.Linear(width, N_CLASSES)
        self.apply(lambda m: init_linear(m, init))
        if mode == "soft":  # start the copies identical so the penalty has a meaningful meaning
            with torch.no_grad():
                for b in self.blocks[1:]:
                    b.weight.copy_(self.blocks[0].weight)
                    b.bias.copy_(self.blocks[0].bias)

    def forward(self, x, return_hidden=False):
        h = F.relu(self.inp(self.norm(x)))
        acts = [h]
        for i in range(self.depth):
            h = F.relu(self.blocks[0 if self.mode == "hard" else i](h))
            acts.append(h)
        logits = self.out(h)
        return (logits, acts) if return_hidden else logits

    def tying_penalty(self) -> torch.Tensor:
        """sum_i ||W_i - mean(W)||^2 over the soft-tied blocks (0 in hard mode)."""
        if self.mode == "hard":
            return torch.zeros((), device=self.inp.weight.device)
        w = torch.stack([b.weight for b in self.blocks])
        return ((w - w.mean(0, keepdim=True)) ** 2).sum()

    def weight_matrices(self):
        return [self.inp.weight, *[b.weight for b in self.blocks], self.out.weight]


class Ensemble(nn.Module):
    """Average of member softmax probabilities.  forward() returns log-probabilities, so
    softmax(forward(x)) is exactly the averaged distribution and downstream code is unchanged."""

    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.spec = dict(arch="ensemble", members=[m.spec for m in members])

    def forward(self, x, return_hidden=False):
        p = torch.stack([F.softmax(m(x), dim=-1) for m in self.members]).mean(0)
        out = torch.log(p.clamp_min(1e-12))
        return (out, []) if return_hidden else out


def build_model(spec: dict) -> nn.Module:
    a = spec["arch"]
    if a == "mlp":
        return MLP(spec["hidden"], spec["p_in"], spec["p_hidden"], spec["init"],
                   spec.get("aux_heads"), spec.get("with_decoder", False))
    if a == "tied":
        return TiedDepthMLP(spec["width"], spec["depth"], spec["mode"], spec["init"])
    if a == "ensemble":
        return Ensemble([build_model(s) for s in spec["members"]])
    raise ValueError(f"unknown arch {a!r}")


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
