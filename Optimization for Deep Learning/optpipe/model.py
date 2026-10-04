"""Fully connected networks.  Every model maps RAW pixels in [0,1] (shape [B,784]) to class logits;
standardisation is a layer inside the model so serving never repeats preprocessing.

One flexible class covers the architecture variants studied in chapter 8:
    activation      relu | leaky_relu | elu | tanh | sigmoid | maxout     (designing models to aid optimisation)
    batch_norm      BatchNorm before each activation                       (8.7.1)
    residual        skip connections between equal-width hidden layers      (8.7.5)
    gated           multiplicative sigmoid gate on each hidden layer        (12.11 gate biases)
    gaussian_output linear output + learned precision beta                  (12.12 variance parameters)
    aux_heads       extra classifiers on hidden layers (deep supervision)   (8.7.5; stripped for inference)
A model is described by a JSON-serialisable `spec`, which is stored next to the weights so the
inference system can rebuild it without any training code.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import DIM, N_CLASSES

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
ACTIVATIONS = {"relu": F.relu, "leaky_relu": lambda z: F.leaky_relu(z, 0.1), "elu": F.elu,
               "tanh": torch.tanh, "sigmoid": torch.sigmoid}


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(MNIST_MEAN))
        self.register_buffer("std", torch.tensor(MNIST_STD))

    def forward(self, x):
        return (x - self.mean) / self.std


def init_linear(m: nn.Module, scheme: str = "he_normal") -> None:
    if not isinstance(m, nn.Linear):
        return
    if scheme == "he_normal":
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)
    elif scheme == "default":
        m.reset_parameters()
    else:
        raise ValueError(f"unknown init {scheme!r}")


class MLP(nn.Module):
    def __init__(self, hidden=(256, 128), activation="relu", batch_norm=False, residual=False, gated=False,
                 gaussian_output=False, aux_heads=(), init="he_normal"):
        super().__init__()
        if activation != "maxout" and activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}")
        self.spec = dict(arch="mlp", hidden=list(hidden), activation=activation, batch_norm=batch_norm,
                         residual=residual, gated=gated, gaussian_output=gaussian_output,
                         aux_heads=list(aux_heads), init=init)
        self.activation, self.residual, self.gated, self.gaussian_output = activation, residual, gated, gaussian_output
        k = 2 if activation == "maxout" else 1
        dims = [DIM, *hidden]
        self.norm = Normalize()
        self.hidden = nn.ModuleList(nn.Linear(a, b * k) for a, b in zip(dims[:-1], dims[1:]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(b * k) if batch_norm else nn.Identity() for b in hidden)
        self.gates = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])) if gated else None
        self.out = nn.Linear(dims[-1], N_CLASSES)
        self.aux = nn.ModuleDict({str(i): nn.Linear(hidden[i], N_CLASSES) for i in aux_heads})
        if gaussian_output:
            self.log_beta = nn.Parameter(torch.zeros(()))
        self.act_grad_clip = None      # training-time hook threshold (clipping the back-propagated gradient)
        self.clip_hits = [0, 0]        # [clipped elements, total elements]
        self.apply(lambda m: init_linear(m, init))

    # ------------------------------------------------------------------ forward
    def _clip_hook(self, g):
        v = self.act_grad_clip
        self.clip_hits[0] += int((g.abs() > v).sum())
        self.clip_hits[1] += g.numel()
        return g.clamp(-v, v)

    def _act(self, z):
        if self.activation == "maxout":
            return z.view(z.shape[0], -1, 2).max(-1).values
        return ACTIVATIONS[self.activation](z)

    def trunk(self, x):
        h = self.norm(x)
        acts = []
        for i, (lin, bn) in enumerate(zip(self.hidden, self.bns)):
            z = bn(lin(h))
            if self.act_grad_clip and self.training and z.requires_grad:
                z.register_hook(self._clip_hook)
            u = self._act(z)
            if self.gated:
                u = torch.sigmoid(self.gates[i](h)) * u
            if self.residual and u.shape == h.shape:
                u = u + h
            h = u
            acts.append(h)
        return h, acts

    def mean_output(self, x):
        return self.out(self.trunk(x)[0])

    def forward(self, x, return_hidden=False):
        h, acts = self.trunk(x)
        f = self.out(h)
        logits = f * torch.exp(self.log_beta) if self.gaussian_output else f
        return (logits, acts) if return_hidden else logits

    def forward_all(self, x) -> dict:
        h, acts = self.trunk(x)
        f = self.out(h)
        out = {"logits": f * torch.exp(self.log_beta) if self.gaussian_output else f, "mean": f}
        out.update({f"aux_{i}": self.aux[i](acts[int(i)]) for i in self.aux})
        return out

    # ------------------------------------------------------------------ structure helpers
    def weight_layers(self) -> list:
        """The nn.Linear layers that carry the main signal: hidden layers then the output layer."""
        return [*self.hidden, self.out]

    def blocks(self) -> list:
        """Parameter groups for block coordinate descent: one per hidden layer, plus the output block."""
        bl = []
        for i in range(len(self.hidden)):
            ps = [*self.hidden[i].parameters(), *self.bns[i].parameters()]
            if self.gated:
                ps += list(self.gates[i].parameters())
            bl.append(ps)
        out = list(self.out.parameters()) + [p for a in self.aux.values() for p in a.parameters()]
        if self.gaussian_output:
            out.append(self.log_beta)
        bl.append(out)
        return bl


def build_model(spec: dict) -> nn.Module:
    if spec["arch"] != "mlp":
        raise ValueError(f"unknown arch {spec['arch']!r}")
    return MLP(spec["hidden"], spec["activation"], spec["batch_norm"], spec["residual"], spec["gated"],
               spec["gaussian_output"], spec["aux_heads"], spec["init"])


def strip_aux(model: MLP) -> MLP:
    """Discard auxiliary heads (they are training aids); returns an equivalent inference model."""
    if not model.aux:
        return model
    spec = {**model.spec, "aux_heads": []}
    new = build_model(spec)
    own = {k: v for k, v in model.state_dict().items() if not k.startswith("aux.")}
    new.load_state_dict(own)
    return new


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
