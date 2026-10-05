"""Sanity checks (Deep Learning book Ch. 11 debugging), qualitative review, and next-step advice for DETR."""
import copy
import math

import numpy as np
import torch
import torch.nn as nn

from detr_voc.infer import denormalize, postprocess
from detr_voc.ops import box_cxcywh_to_xyxy


def _to_dev(images, mask, targets, device):
    return images.to(device), mask.to(device), [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]


def check_initial_loss(model, criterion, batch, device, num_classes, tol):
    """Untrained heads: every query is nearly uniform, so the class loss is close to ln(K + 1)."""
    m = copy.deepcopy(model).to(device).eval()
    images, mask, targets = _to_dev(*batch, device)
    with torch.no_grad():
        ls = criterion(m(images, mask), targets)
    expected, got = math.log(num_classes + 1), float(ls["loss_ce"])
    return {"initial_loss_ce": got, "expected_ln_k": expected, "initial_total": float(ls["total"]),
            "passed": bool(abs(got - expected) <= tol * expected)}


def overfit_tiny_batch(model, criterion, batch, device, steps=80, ratio=0.85, lr=3e-4, grad_clip=0.1):
    """The network must be able to reduce its loss on a handful of images (dropout off); failure points to a bug."""
    m = copy.deepcopy(model).to(device).train()
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = 0.0
        if isinstance(mod, nn.MultiheadAttention):
            mod.dropout = 0.0
    images, mask, targets = _to_dev(*batch, device)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    hist = []
    for _ in range(steps):
        opt.zero_grad()
        loss = criterion(m(images, mask), targets)["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        hist.append(float(loss.detach()))
    final = float(np.mean(hist[-5:]))
    return {"overfit_initial_loss": hist[0], "overfit_final_loss": final, "overfit_ratio": final / max(hist[0], 1e-9),
            "passed": bool(final < ratio * hist[0])}


def run_sanity_checks(cfg, model, criterion, batch, device):
    s = cfg["sanity"]
    n = s["overfit_images"]
    small = (batch[0][:n], batch[1][:n], batch[2][:n])
    init = check_initial_loss(model, criterion, small, device, cfg["data"]["num_foreground"], s["initial_loss_tol"])
    over = overfit_tiny_batch(model, criterion, small, device, s["overfit_steps"], s["overfit_ratio"], s["overfit_lr"])
    return {**init, **{k: v for k, v in over.items() if k != "passed"}, "initial_loss_ok": init["passed"],
            "overfit_ok": over["passed"], "passed": bool(init["passed"] and over["passed"])}


@torch.inference_mode()
def qualitative_grid(model, dataset, classes, device, out_path, thr=0.5, n=8):
    """Predicted (red) vs ground-truth (green) boxes on the first n images of a dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision.utils import draw_bounding_boxes
    model.eval()
    cols = 4
    n = min(n, len(dataset))
    rows = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, range(n), strict=False):
        x, tgt, meta = dataset[i]
        h, w = x.shape[1:]
        out = model(x[None].to(device), torch.zeros(1, h, w, dtype=torch.bool, device=device))
        det = postprocess(out, torch.tensor([[w, h]]))[0]
        keep = det["scores"] >= thr
        u8 = (denormalize(x) * 255).round().byte()
        if len(tgt["labels"]):
            gt = box_cxcywh_to_xyxy(tgt["boxes"]) * torch.tensor([w, h, w, h])
            u8 = draw_bounding_boxes(u8, gt, [classes[int(c)] for c in tgt["labels"]], colors="lime", width=2)
        if keep.any():
            u8 = draw_bounding_boxes(u8, det["boxes"][keep].cpu(),
                                     [f"{classes[int(c) - 1]} {s:.2f}" for c, s in zip(det["labels"][keep].cpu(), det["scores"][keep].cpu(), strict=True)],
                                     colors="red", width=2)
        ax.imshow(u8.permute(1, 2, 0).numpy())
    fig.suptitle(f"green = ground truth, red = predictions (score >= {thr})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


LEVERS = {
    "overfitting": [
        "Keep early stopping on: the saved model is the best-validation epoch, not the last one",
        "More data is the real fix for a transformer: add data.voc_train_sets=[\"2012_trainval\",\"2007_trainval\"] or train on COCO (stage coco)",
        "Stronger regularisation: model.dropout=0.2, schedule.weight_decay=0.0002, fewer queries model.num_queries=50, smaller model.enc_layers/dec_layers=4",
        "Enable weight averaging: schedule.ema.enabled=true",
        "Lower the backbone LR further: schedule.lr_backbone=0.000005"],
    "underfitting": [
        "DETR converges slowly: train longer (schedule.phases epochs up, schedule.scale=1.5); the paper used 300-500 epochs",
        "Check the learning rates: schedule.phases[0].lr=0.0002 for batch sizes above 32; keep schedule.grad_clip=0.1",
        "More capacity: model.backbone=resnet101 or model.dilation=true (DC5, 2x compute)",
        "Remove regularisation first: model.dropout=0, aug.crop_prob=0",
        "Check labels and boxes in qualitative_val.png"],
    "unstable": ["Lower the first-phase LR (x0.5), keep schedule.grad_clip=0.1, check for degenerate boxes"],
    "not_learning": ["DETR often shows near-zero mAP for many epochs on small data when trained from an ImageNet backbone: see "
                     "sanity.json, qualitative_val.png, and consider more data or a longer schedule"],
    "still_improving": ["Train longer: raise the phase lengths in schedule.phases; the validation metric had not plateaued"],
    "healthy": ["No red flags in the curves. Compare with the target, then evaluate once on the test set"],
}


def next_steps(diagnosis: dict) -> dict:
    verdicts = diagnosis.get("verdict", [])
    return {"verdict": verdicts, "levers": {v: LEVERS[v] for v in verdicts if v in LEVERS},
            "note": "A diagnosis is a hypothesis. Change ONE lever at a time and compare runs in MLflow."}
