"""Evaluation: accuracy, macro one-vs-rest AUC, hard-label log-loss, uncertainty, init diagnostics."""
from __future__ import annotations

import copy
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
    except ValueError:
        present = np.unique(y)
        return float(np.mean([roc_auc_score((y == c).astype(int), probs[:, c]) for c in present])) \
            if len(present) > 1 else float("nan")


def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor, bootstrap: int = 0, seed: int = 0) -> dict:
    probs = F.softmax(logits, dim=-1)
    y_np, p_np = y.numpy(), probs.numpy()
    n = len(y_np)
    acc = float((p_np.argmax(1) == y_np).mean())
    out = {"loss": float(F.cross_entropy(logits, y)), "accuracy": acc,
           "accuracy_se": math.sqrt(acc * (1 - acc) / n), "auc": auc_macro_ovr(y_np, p_np)}
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


def init_diagnostics(model, x_u8, y, device, n: int = 256) -> dict:
    """What the network looks like BEFORE the first update: initial loss (a K-class model that
    knows nothing should start near ln K), per-layer activation scale and weight-gradient norm."""
    m = copy.deepcopy(model).to(device).train()
    x = to_float(x_u8[:n]).to(device)
    yb = y[:n].to(device)
    logits, acts = m(x, return_hidden=True)
    loss = F.cross_entropy(logits, yb)
    m.zero_grad()
    loss.backward()
    layers = m.weight_layers() if hasattr(m, "weight_layers") else []
    gn = [float(l.weight.grad.norm()) for l in layers if l.weight.grad is not None]
    out = {"init_loss": loss.item(), "init_loss_over_lnK": loss.item() / math.log(N_CLASSES),
           "init_act_std": [float(a.std()) for a in acts]}
    if len(gn) >= 2 and gn[-1] > 0:
        out["init_grad_norm_first_over_last"] = gn[0] / gn[-1]
    out["init_grad_norms"] = gn
    return out
