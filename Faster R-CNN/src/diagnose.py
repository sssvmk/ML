"""Sanity checks (Deep Learning book Ch. 11 debugging), qualitative review, reference statistics, next-step advice."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn

from src.model import set_loss_mode


def _to_dev(images, targets, device):
    imgs = [i.to(device) for i in images]
    tg = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items() if k != "difficult"}
          for t in targets]
    return imgs, tg


def check_initial_loss(model, batch, device, num_foreground, tol):
    """With a freshly initialised predictor the classification loss must be close to ln(K+1)."""
    m = set_loss_mode(copy.deepcopy(model).to(device))
    imgs, tg = _to_dev(*batch, device)
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(0)
        losses = m(imgs, tg)
    lc = float(losses["loss_classifier"])
    expected = math.log(num_foreground + 1)
    return {"initial_loss_classifier": lc, "expected_ln_k": expected,
            "initial_total": float(sum(losses.values())), "passed": bool(abs(lc - expected) < tol)}


def overfit_tiny_batch(model, batch, device, steps=60, drop=0.6, lr=0.01):
    """The model must be able to memorise a handful of images; failure means a bug in data, loss or optimisation."""
    m = copy.deepcopy(model).to(device).train()
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = 0.0
    imgs, tg = _to_dev(*batch, device)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    history = []
    for _ in range(steps):
        opt.zero_grad()
        loss = sum(m(imgs, tg).values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        history.append(float(loss.detach()))
    final = float(np.mean(history[-5:]))
    return {"overfit_initial_loss": history[0], "overfit_final_loss": final, "overfit_steps": steps,
            "overfit_ratio": final / max(history[0], 1e-9), "passed": bool(final < drop * history[0])}


def run_sanity_checks(cfg, model, loader, num_foreground, device):
    s = cfg["sanity"]
    batch = next(iter(loader))
    batch = (batch[0][: s["overfit_batch"]], batch[1][: s["overfit_batch"]])
    init = check_initial_loss(model, batch, device, num_foreground, s["initial_loss_tol"])
    over = overfit_tiny_batch(model, batch, device, s["overfit_steps"], s["overfit_drop"])
    return {**init, **{k: v for k, v in over.items() if k != "passed"}, "initial_loss_ok": init["passed"],
            "overfit_ok": over["passed"], "passed": bool(init["passed"] and over["passed"])}


@torch.inference_mode()
def qualitative_grid(model, dataset, classes, device, out_path, n=8, thr=0.5):
    """Predicted (red) vs ground-truth (green) boxes on the first n images of a dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision.utils import draw_bounding_boxes

    model.eval()
    cols = 4
    rows = math.ceil(min(n, len(dataset)) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.0), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, range(min(n, len(dataset))), strict=False):
        img, tgt = dataset[i]
        out = model([img.to(device)])[0]
        keep = out["scores"] >= thr
        u8 = (img * 255).round().byte()
        gt_boxes, gt_labels = tgt["boxes"].as_subclass(torch.Tensor), tgt["labels"]
        u8 = draw_bounding_boxes(u8, gt_boxes, [classes[int(c) - 1] for c in gt_labels], colors="lime", width=2)
        pb, pl, ps = out["boxes"][keep].cpu(), out["labels"][keep].cpu(), out["scores"][keep].cpu()
        u8 = draw_bounding_boxes(u8, pb, [f"{classes[int(c) - 1]} {s:.2f}" for c, s in zip(pl, ps, strict=True)], colors="red",
                                 width=2)
        ax.imshow(u8.permute(1, 2, 0).numpy())
    fig.suptitle(f"green = ground truth, red = predictions (score >= {thr})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


@torch.inference_mode()
def reference_stats(model, dataset, device, score_thr=0.5, max_images=200):
    """Training-time reference for drift monitoring: pixel statistics, detections per image, score and class mix."""
    model.eval()
    n = min(len(dataset), max_images)
    means, sizes, scores, labels, counts = [], [], [], [], []
    for i in range(n):
        img, _ = dataset[i]
        out = model([img.to(device)])[0]
        means.append(img.mean((1, 2)).numpy())
        sizes.append(img.shape[-2:])
        sc, lb = out["scores"].cpu().numpy(), out["labels"].cpu().numpy()
        scores.extend(sc.tolist())
        keep = sc >= score_thr
        labels.extend(lb[keep].tolist())
        counts.append(int(keep.sum()))
    means = np.stack(means)
    bins = np.linspace(0, 1, 11)
    hist = np.histogram(scores, bins=bins)[0] / max(len(scores), 1)
    return {"channel_mean": means.mean(0).tolist(), "channel_mean_std": means.std(0).tolist(),
            "mean_height": float(np.mean([s[0] for s in sizes])), "mean_width": float(np.mean([s[1] for s in sizes])),
            "detections_per_image": float(np.mean(counts)), "score_hist": hist.tolist(), "score_bins": bins.tolist(),
            "predicted_class_counts": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True),
                                                                      strict=False)},
            "score_threshold": score_thr, "n_images": n}


LEVERS = {
    "overfitting": [
        "Stronger augmentation: data.aug=strong (photometric + zoom-out + IoU-crop + flip), multi-scale model.min_size=[256,288,320]",
        "Dropout on the box head: model.box_head_dropout=0.2 (raise to 0.3-0.5 only if the gap persists)",
        "More weight decay: training.weight_decay=0.0005; lower the backbone LR: training.backbone_lr_mult=0.1",
        "Freeze more of the backbone: model.trainable_backbone_layers=1 or 2",
        "Keep EMA on (training.ema.enabled=true) and keep early stopping; stop at the best validation epoch",
        "Afterwards retrain on train+val for the best epoch count (python run.py refit --run-id <RUN_ID>)"],
    "underfitting": [
        "Free more capacity: model.trainable_backbone_layers=5 (or 6 for MobileNet), or a larger arch",
        "Higher input resolution: model.min_size=[480] / max_size=800 (slower)",
        "Train longer / higher LR: training.max_epochs up, training.lr x2 (watch for instability)",
        "Remove regularisation first: data.aug=flip, model.box_head_dropout=0, lower weight decay",
        "Check labels and boxes (see qualitative_val.png): label noise looks like underfitting"],
    "unstable": [
        "Lower training.lr (x0.5), keep training.grad_clip=10, lengthen training.warmup_iters",
        "Check for degenerate boxes (zero width/height) and the sanity.json initial-loss check"],
    "not_learning": [
        "Run the sanity checks (sanity.json); inspect qualitative_val.png; verify image/box alignment and class ids"],
    "still_improving": ["Train longer: raise training.max_epochs; the validation metric has not plateaued"],
    "healthy": ["No red flags in the curves. Compare with the target, then evaluate once on the test set"],
}


def next_steps(diagnosis: dict) -> dict:
    verdicts = diagnosis.get("verdict", [])
    return {"verdict": verdicts, "levers": {v: LEVERS[v] for v in verdicts if v in LEVERS},
            "note": "A diagnosis is a hypothesis. Change ONE lever at a time and compare runs in MLflow."}
