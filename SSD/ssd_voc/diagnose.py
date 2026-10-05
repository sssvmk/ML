"""Sanity checks (Deep Learning book Ch. 11 debugging), qualitative review, and next-step advice."""
import copy
import math

import numpy as np
import torch

from ssd_voc.boxes import postprocess
from ssd_voc.data import denormalize


def check_initial_loss(model, loss_fn, batch, device, num_classes, tol):
    """Untrained model: every sampled box (positives + 3x hard negatives) costs ~ln(C), so the confidence part of the
    loss is close to (1 + neg_pos_ratio) * ln(C)."""
    images, targets = batch[0].to(device), batch[1]
    m = copy.deepcopy(model).to(device).eval()
    with torch.no_grad():
        loc, conf = m(images)
        losses = loss_fn(loc.float(), conf.float(), targets)
    expected = (1 + loss_fn.neg_pos_ratio) * math.log(num_classes)
    got = float(losses["conf"])
    return {"initial_conf_loss": got, "expected_conf_loss": expected, "initial_total": float(losses["total"]),
            "num_positives": int(losses["num_pos"]), "passed": bool(abs(got - expected) <= tol * expected)}


def overfit_tiny_batch(model, loss_fn, batch, device, steps=60, ratio=0.7, lr=0.01):
    """The network must be able to memorise a handful of images; failure points to a bug in data, loss or optimiser."""
    m = copy.deepcopy(model).to(device).train()
    images, targets = batch[0].to(device), batch[1]
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    hist = []
    for _ in range(steps):
        opt.zero_grad()
        loc, conf = m(images)
        loss = loss_fn(loc.float(), conf.float(), targets)["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        hist.append(float(loss.detach()))
    final = float(np.mean(hist[-5:]))
    return {"overfit_initial_loss": hist[0], "overfit_final_loss": final, "overfit_ratio": final / max(hist[0], 1e-9),
            "passed": bool(final < ratio * hist[0])}


def run_sanity_checks(cfg, model, loss_fn, batch, device):
    s = cfg["sanity"]
    n = s["overfit_images"]
    small = (batch[0][:n], batch[1][:n])
    init = check_initial_loss(model, loss_fn, small, device, cfg["data"]["num_foreground"] + 1, s["initial_loss_tol"])
    over = overfit_tiny_batch(model, loss_fn, small, device, s["overfit_steps"], s["overfit_ratio"])
    return {**init, **{k: v for k, v in over.items() if k != "passed"}, "initial_loss_ok": init["passed"],
            "overfit_ok": over["passed"], "passed": bool(init["passed"] and over["passed"])}


@torch.inference_mode()
def qualitative_grid(model, dataset, classes, device, out_path, thr=0.5, n=8, post_cfg=None, variances=(0.1, 0.2)):
    """Predicted (red) vs ground-truth (green) boxes on the first n images of a dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision.utils import draw_bounding_boxes

    model.eval()
    post_cfg = post_cfg or {}
    cols = 4
    n = min(n, len(dataset))
    rows = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, range(n), strict=False):
        x, tgt, _ = dataset[i]
        loc, conf = model(x[None].to(device))
        det = postprocess(loc, conf, model.priors, variances, post_cfg.get("score_thresh", 0.01),
                          post_cfg.get("nms_iou", 0.45), post_cfg.get("max_detections", 200))[0]
        keep = det["scores"] >= thr
        s = x.shape[-1]
        u8 = (denormalize(x) * 255).round().byte()
        if len(tgt["labels"]):
            u8 = draw_bounding_boxes(u8, tgt["boxes"] * s, [classes[int(c) - 1] for c in tgt["labels"]], colors="lime",
                                     width=2)
        if keep.any():
            u8 = draw_bounding_boxes(u8, (det["boxes"][keep] * s).cpu(),
                                     [f"{classes[int(c) - 1]} {sc:.2f}" for c, sc in
                                      zip(det["labels"][keep].cpu(), det["scores"][keep].cpu(), strict=True)],
                                     colors="red", width=2)
        ax.imshow(u8.permute(1, 2, 0).numpy())
    fig.suptitle(f"green = ground truth, red = predictions (score >= {thr})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


LEVERS = {
    "overfitting": [
        "Stronger augmentation: aug.expansion=true (zoom-out), schedule.scale=2 (the paper's longer schedule), or lower aug.crop_min_scale",
        "More weight decay: schedule.weight_decay=0.001; freeze more of the backbone: model.freeze_up_to=layer2",
        "Add training data: data.voc_train_sets=[\"2012_trainval\",\"2007_trainval\"] or start from a COCO model (stage voc_from_coco)",
        "Smaller input or a smaller backbone: model.backbone=resnet34",
        "Keep early stopping and EMA on; the best-validation checkpoint is the one that is saved and tested"],
    "underfitting": [
        "Unfreeze more: model.freeze_up_to=\"\" (nothing frozen) or model.freeze_bn=false",
        "More capacity: model.input_size=512 (SSD512) or model.backbone=resnet101",
        "Train longer / lift the first-phase LR: schedule.scale=2, schedule.phases[0].lr=0.002",
        "Remove regularisation first: aug.expansion=false, aug.crop_min_scale=0.5, schedule.weight_decay=0.0001",
        "Check labels and boxes in qualitative_val.png: broken boxes look like underfitting"],
    "unstable": ["Lower the first-phase LR (x0.5), keep grad_clip=10, raise schedule.warmup_iters"],
    "not_learning": ["Check sanity.json, qualitative_val.png, class ids and box coordinates (VOC is 1-based)"],
    "still_improving": ["Train longer: schedule.scale=2; the validation metric had not plateaued"],
    "healthy": ["No red flags in the curves. Compare with the target, then evaluate once on the test set"],
}


def next_steps(diagnosis: dict) -> dict:
    verdicts = diagnosis.get("verdict", [])
    return {"verdict": verdicts, "levers": {v: LEVERS[v] for v in verdicts if v in LEVERS},
            "note": "A diagnosis is a hypothesis. Change ONE lever at a time and compare runs in MLflow."}
