"""Evaluation: accuracy, macro one-vs-rest AUC, log-loss (always HARD-label cross-entropy so
that numbers are comparable across methods), uncertainty, FGSM robustness, sparsity."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .data import N_CLASSES, to_float


@torch.no_grad()
def predict_logits(model, x_u8: torch.Tensor, device, bs: int = 2048) -> torch.Tensor:
    was_training = model.training
    model.eval()
    outs = []
    for i in range(0, len(x_u8), bs):
        outs.append(model(to_float(x_u8[i : i + bs]).to(device)).float().cpu())
    model.train(was_training)
    return torch.cat(outs)


def auc_macro_ovr(y: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=list(range(N_CLASSES))))
    except ValueError:  # a class is missing from y (tiny smoke-test splits)
        present = np.unique(y)
        return float(np.mean([roc_auc_score((y == c).astype(int), probs[:, c]) for c in present])) \
            if len(present) > 1 else float("nan")


def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor, bootstrap: int = 0, seed: int = 0) -> dict:
    probs = F.softmax(logits, dim=-1)
    y_np, p_np = y.numpy(), probs.numpy()
    pred = p_np.argmax(1)
    n = len(y_np)
    acc = float((pred == y_np).mean())
    out = {
        "loss": float(F.cross_entropy(logits, y)),
        "accuracy": acc,
        "accuracy_se": math.sqrt(acc * (1 - acc) / n),
        "auc": auc_macro_ovr(y_np, p_np),
    }
    if bootstrap:
        g = np.random.default_rng(seed)
        aucs = []
        for _ in range(bootstrap):
            idx = g.integers(0, n, n)
            aucs.append(auc_macro_ovr(y_np[idx], p_np[idx]))
        out["auc_ci95"] = [float(np.nanpercentile(aucs, 2.5)), float(np.nanpercentile(aucs, 97.5))]
    return out


def evaluate(model, x_u8, y, device, bootstrap: int = 0) -> dict:
    return metrics_from_logits(predict_logits(model, x_u8, device), y, bootstrap=bootstrap)


def fgsm_accuracy(model, x_u8, y, device, eps: float, bs: int = 1000) -> float:
    """Accuracy on FGSM examples x + eps * sign(grad_x CE), clipped to [0,1]."""
    was_training = model.training
    model.eval()
    correct = 0
    for i in range(0, len(x_u8), bs):
        x = to_float(x_u8[i : i + bs]).to(device).requires_grad_(True)
        yb = y[i : i + bs].to(device)
        loss = F.cross_entropy(model(x), yb)
        (g,) = torch.autograd.grad(loss, x)
        with torch.no_grad():
            adv = (x + eps * g.sign()).clamp(0, 1)
            correct += (model(adv).argmax(1) == yb).sum().item()
    model.train(was_training)
    return correct / len(x_u8)


@torch.no_grad()
def sparsity_stats(model, x_u8, device, n: int = 5000) -> dict:
    """Fraction of exactly-(near-)zero hidden activations and of near-zero weights."""
    was_training = model.training
    model.eval()
    out = {}
    try:
        _, acts = model(to_float(x_u8[:n]).to(device), return_hidden=True)
        if acts:
            out["activation_sparsity"] = float(torch.cat([(a.abs() < 1e-6).float().flatten() for a in acts]).mean())
    except TypeError:
        pass
    if hasattr(model, "weight_matrices"):
        w = torch.cat([m.detach().abs().flatten() for m in model.weight_matrices()])
        out["weight_sparsity"] = float((w < 1e-3).float().mean())
    model.train(was_training)
    return out
