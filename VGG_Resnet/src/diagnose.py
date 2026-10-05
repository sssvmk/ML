"""Sanity checks (Deep Learning book Ch. 11 debugging), worst-error export, reference statistics."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn

from src.data import denormalize
from src.model import build_model


def check_initial_loss(model, loader, device, num_classes, tol):
    """Untrained model on one batch should give loss close to ln(K)."""
    m = copy.deepcopy(model).to(device).train()
    x, y = next(iter(loader))
    with torch.no_grad():
        loss = nn.functional.cross_entropy(m(x.to(device)), y.to(device)).item()
    expected = math.log(num_classes)
    return {"initial_loss": loss, "expected_ln_k": expected, "passed": abs(loss - expected) < tol}


def overfit_tiny_batch(cfg, loader, num_classes, device, n=32, steps=80):
    """The model must be able to memorise a handful of examples; failure means a bug."""
    mcfg = {**cfg["model"], "dropout": 0.0}
    model = build_model(mcfg, num_classes).to(device).train()
    xs, ys = [], []
    for x, y in loader:
        xs.append(x), ys.append(y)
        if sum(len(v) for v in ys) >= n:
            break
    x, y = torch.cat(xs)[:n].to(device), torch.cat(ys)[:n].to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_val, acc, used = float("inf"), 0.0, 0
    for step in range(steps):
        used = step + 1
        opt.zero_grad()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        loss_val, acc = loss.item(), (logits.argmax(1) == y).float().mean().item()
        if acc == 1.0 and loss_val < 0.05:
            break
    return {"overfit_final_loss": loss_val, "overfit_acc": acc, "overfit_steps_used": used,
            "passed": acc >= 0.95 or loss_val < 0.1}


def run_sanity_checks(cfg, train_loader, num_classes, device):
    s = cfg["sanity"]
    init = check_initial_loss(build_model(cfg["model"], num_classes), train_loader, device, num_classes,
                              s["initial_loss_tol"])
    over = overfit_tiny_batch(cfg, train_loader, num_classes, device, s["overfit_batch"], s["overfit_steps"])
    return {**init, **{k: v for k, v in over.items() if k != "passed"},
            "initial_loss_ok": init["passed"], "overfit_ok": over["passed"],
            "passed": bool(init["passed"] and over["passed"])}


def export_worst_errors(val: dict, val_ds, classes, out_dir, norm, k=32):
    """Save the k most confidently wrong validation images (grid + CSV) for human review."""
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    wrong = torch.nonzero(val["pred"] != val["target"]).flatten()
    if len(wrong) == 0:
        return
    order = wrong[val["sample_loss"][wrong].argsort(descending=True)][:k].tolist()
    cols = 8
    rows = math.ceil(len(order) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.1), squeeze=False)
    with open(out_dir / "worst_errors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["val_index", "true", "pred", "confidence", "loss"])
        for ax in axes.flat:
            ax.axis("off")
        for ax, i in zip(axes.flat, order, strict=False):
            img, _ = val_ds[i]
            ax.imshow(denormalize(img[None], *norm)[0].permute(1, 2, 0).numpy())
            t, p, c = classes[int(val["target"][i])], classes[int(val["pred"][i])], float(val["conf"][i])
            ax.set_title(f"{t}\n→ {p} ({c:.2f})", fontsize=6)
            w.writerow([i, t, p, f"{c:.4f}", f"{float(val['sample_loss'][i]):.4f}"])
    fig.tight_layout()
    fig.savefig(out_dir / "worst_errors.png", dpi=130)
    plt.close(fig)


def reference_stats(val_ds, val: dict, norm, max_images=2000):
    """Training-time reference for drift monitoring: pixel statistics + confidence histogram."""
    n = min(len(val_ds), max_images)
    imgs = torch.stack([denormalize(val_ds[i][0][None], *norm)[0] for i in range(n)])
    mean, std = imgs.mean((0, 2, 3)), imgs.std((0, 2, 3))
    per_image_mean = imgs.mean((1, 2, 3))
    bins = np.linspace(0, 1, 11)
    hist = np.histogram(val["conf"].numpy(), bins=bins)[0] / len(val["conf"])
    return {
        "channel_mean": mean.tolist(), "channel_std": std.tolist(),
        "per_image_mean_std": float(per_image_mean.std()),
        "confidence_hist": hist.tolist(), "confidence_bins": bins.tolist(), "n_images": n,
    }
