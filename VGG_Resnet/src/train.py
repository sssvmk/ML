"""Training loop with per-epoch MLflow + CSV logging, checkpointing, early stopping, resume."""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import mlflow
import torch
import torch.nn as nn

from src import diagnose
from src.data import build_datasets, build_loaders, get_norm
from src.model import build_model, count_params
from src.utils import ROOT, env_info, flatten, git_commit, select_device, set_seed

CSV_FIELDS = ["epoch", "train_loss", "val_loss", "train_metric", "val_metric", "lr",
              "val_top5", "grad_norm", "epoch_time_s"]


def make_optimizer(model, t):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]
    groups = [{"params": decay, "weight_decay": t["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    if t["optimizer"] == "sgd":
        return torch.optim.SGD(groups, lr=t["lr"], momentum=t["momentum"], nesterov=t.get("nesterov", True))
    if t["optimizer"] == "adamw":
        return torch.optim.AdamW(groups, lr=t["lr"])
    raise ValueError(f"unknown optimizer {t['optimizer']}")


def make_scheduler(optimizer, steps_per_epoch, t):
    total = max(1, steps_per_epoch * t["max_epochs"])
    warm = steps_per_epoch * t["warmup_epochs"]

    def lr_lambda(step):
        if warm and step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, total - warm)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_model(model, loader, device, criterion=None, collect=False):
    """Returns loss, top1, top5 (and per-sample probs/preds/targets/losses when collect=True)."""
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="none")
    n = c1 = c5 = 0
    loss_sum = 0.0
    store = {"conf": [], "pred": [], "target": [], "sample_loss": []}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses = ce(logits, y)
        probs = logits.softmax(1)
        k = min(5, logits.shape[1])
        topk = logits.topk(k, 1).indices
        c1 += (topk[:, 0] == y).sum().item()
        c5 += (topk == y[:, None]).any(1).sum().item()
        loss_sum += losses.sum().item()
        n += y.numel()
        if collect:
            conf, pred = probs.max(1)
            for key, val in (("conf", conf), ("pred", pred), ("target", y), ("sample_loss", losses)):
                store[key].append(val.cpu())
    out = {"loss": loss_sum / n, "top1": c1 / n, "top5": c5 / n, "n": n}
    if collect:
        out.update({k: torch.cat(v) for k, v in store.items()})
    return out


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, amp, grad_clip, channels_last):
    model.train()
    loss_sum = correct = n = steps = 0
    gn_sum = 0.0
    params = [p for p in model.parameters() if p.requires_grad]
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss: lower the learning rate or check the data")
        max_norm = grad_clip if grad_clip else float("inf")
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = nn.utils.clip_grad_norm_(params, max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            gn = nn.utils.clip_grad_norm_(params, max_norm)
            optimizer.step()
        scheduler.step()
        loss_sum += loss.item() * y.numel()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
        gn_sum += float(gn)
        steps += 1
    return {"loss": loss_sum / n, "top1": correct / n, "grad_norm": gn_sum / max(1, steps)}


def fit(cfg, loaders, num_classes, classes, device, out_dir: Path, log_mlflow=True):
    t = cfg["training"]
    amp = bool(t["amp"] and device.type == "cuda")
    channels_last = bool(t["channels_last"] and device.type == "cuda")
    model = build_model(cfg["model"], num_classes).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=t["label_smoothing"])
    optimizer = make_optimizer(model, t)
    scheduler = make_scheduler(optimizer, len(loaders["train"]), t)
    scaler = torch.amp.GradScaler("cuda") if amp else None

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = ckpt_dir / "best.pt", ckpt_dir / "last.pt"
    start_epoch, best, bad = 0, -1.0, 0
    if t.get("resume_from"):
        state = torch.load(t["resume_from"], map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best, bad = state["epoch"] + 1, state["best"], state["bad"]

    csv_path = out_dir / "metrics.csv"
    new_file = not (t.get("resume_from") and csv_path.exists())
    with open(csv_path, "a" if not new_file else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        for epoch in range(start_epoch, t["max_epochs"]):
            t0 = time.time()
            tr = train_one_epoch(model, loaders["train"], criterion, optimizer, scheduler, scaler,
                                 device, amp, t["grad_clip"], channels_last)
            va = evaluate_model(model, loaders["val"], device)
            lr = optimizer.param_groups[0]["lr"]
            row = {"epoch": epoch, "train_loss": tr["loss"], "val_loss": va["loss"],
                   "train_metric": tr["top1"], "val_metric": va["top1"], "lr": lr,
                   "val_top5": va["top5"], "grad_norm": tr["grad_norm"], "epoch_time_s": time.time() - t0}
            writer.writerow(row)
            f.flush()
            if log_mlflow:
                mlflow.log_metrics({"train_loss": tr["loss"], "val_loss": va["loss"], "train_top1": tr["top1"],
                                    "val_top1": va["top1"], "val_top5": va["top5"], "lr": lr,
                                    "grad_norm": tr["grad_norm"], "epoch_time_s": row["epoch_time_s"]}, step=epoch)
            print(f"epoch {epoch:03d} | train {tr['loss']:.3f}/{tr['top1']:.3f} | "
                  f"val {va['loss']:.3f}/{va['top1']:.3f} | lr {lr:.4f} | {row['epoch_time_s']:.0f}s", flush=True)

            if va["top1"] > best:
                best, bad = va["top1"], 0
                mean, std = get_norm(cfg)
                torch.save({"state_dict": model.state_dict(), "model_cfg": cfg["model"], "classes": classes,
                            "num_classes": num_classes, "epoch": epoch, "val_top1": best,
                            "norm": {"mean": list(mean), "std": list(std)}, "dataset": cfg["data"]["dataset"]},
                           best_path)
            else:
                bad += 1
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch, "best": best, "bad": bad}, last_path)
            if bad >= t["early_stopping_patience"]:
                print(f"early stopping at epoch {epoch} (no val improvement for {bad} epochs)")
                break
    return best_path, csv_path, best


def run_training(cfg: dict, env: dict, config_path: str | None = None) -> dict:
    set_seed(cfg["seed"])
    device = select_device(cfg["training"]["device"])
    ds = build_datasets(cfg)
    loaders = build_loaders(cfg, ds, device)
    k = len(ds["classes"])

    mlflow.set_tracking_uri(env["tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])
    with mlflow.start_run(run_name=cfg.get("run_name")) as run:
        run_id = run.info.run_id
        out_dir = ROOT / "reports" / "runs" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        mlflow.log_params(flatten(cfg))
        mlflow.set_tags({
            "phase": cfg["phase"], "task": "image-classification", "git_commit": git_commit(),
            "data_version": ds["data_version"], "metric": cfg["metric"]["name"],
            "metric_target": str(cfg["metric"]["target_value"]), "test_evaluated": "false",
            **{f"env.{a}": str(b) for a, b in env_info(device).items()},
        })
        proto = build_model(cfg["model"], k)
        mlflow.log_param("num_parameters", count_params(proto))
        print(f"run {run_id} | device {device} | params {count_params(proto):,} | data {ds['data_version']}")

        if cfg["sanity"]["enabled"]:
            report = diagnose.run_sanity_checks(cfg, loaders["train"], k, device)
            (out_dir / "sanity.json").write_text(json.dumps(report, indent=2))
            mlflow.log_dict(report, "diagnostics/sanity.json")
            mlflow.log_metrics({"sanity_initial_loss": report["initial_loss"],
                                "sanity_overfit_final_loss": report["overfit_final_loss"]})
            if cfg["sanity"]["fail_hard"] and not report["passed"]:
                mlflow.set_tag("sanity_passed", "false")
                raise RuntimeError(f"sanity checks failed: {json.dumps(report)}")
            mlflow.set_tag("sanity_passed", str(report["passed"]).lower())

        best_path, csv_path, best = fit(cfg, loaders, k, ds["classes"], device, out_dir)
        mlflow.log_metric("best_val_top1", best)

        # diagnosis from the learning curves (hypothesis, not a conclusion)
        diag_json, curves_png = out_dir / "diagnosis.json", out_dir / "curves.png"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "diagnose_curves.py"), str(csv_path),
                        "--out", str(diag_json), "--plot", str(curves_png),
                        "--target", str(cfg["metric"]["target_value"])], check=False)

        # worst errors + reference statistics for drift monitoring (validation set only)
        state = torch.load(best_path, map_location=device, weights_only=True)
        model = build_model(cfg["model"], k).to(device)
        model.load_state_dict(state["state_dict"])
        val = evaluate_model(model, loaders["val"], device, collect=True)
        diagnose.export_worst_errors(val, ds["val"], ds["classes"], out_dir / "worst_errors", ds["norm"])
        ref = diagnose.reference_stats(ds["val"], val, ds["norm"])
        (out_dir / "reference_stats.json").write_text(json.dumps(ref, indent=2))
        (out_dir / "class_names.json").write_text(json.dumps(ds["classes"]))
        mlflow.log_metrics({"final_val_top1": val["top1"], "final_val_top5": val["top5"], "final_val_loss": val["loss"]})

        # artifacts
        for p in (csv_path, curves_png, diag_json, out_dir / "reference_stats.json", out_dir / "class_names.json"):
            if p.exists():
                mlflow.log_artifact(str(p))
        mlflow.log_artifacts(str(out_dir / "worst_errors"), "worst_errors")
        mlflow.log_artifact(str(best_path), "checkpoints")
        if config_path and Path(config_path).exists():
            mlflow.log_artifact(config_path, "config")
        req = ROOT / "requirements.txt"
        if req.exists():
            mlflow.log_artifact(str(req), "config")
        mlflow.log_dict({p: metadata.version(p) for p in ("torch", "torchvision") },
                        "config/versions.json")
        target = cfg["metric"]["target_value"]
        print(f"best val top-1: {best:.4f} (target {target}) | artifacts: {out_dir}")
        return {"run_id": run_id, "best_val_top1": best, "out_dir": str(out_dir), "target_met": best >= target}


def benchmark(cfg: dict, steps: int = 6) -> dict:
    """Time a few training steps on the configured device and project the cost of a full run."""
    from src.hwcheck import benchmark_one
    device = select_device(cfg["training"]["device"])
    return benchmark_one(cfg, device, cfg["training"]["amp"], cfg["training"]["channels_last"], steps)
