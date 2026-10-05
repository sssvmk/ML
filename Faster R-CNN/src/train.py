"""Faster R-CNN training: SGD/AdamW + warmup + cosine/multistep/plateau LR, EMA, early stopping, MLflow logging."""
from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import mlflow
import torch
import torch.nn as nn

from src import diagnose
from src.data import build_datasets, build_loaders
from src.ema import ModelEMA
from src.metrics import DetectionEvaluator, error_breakdown
from src.model import build_model, count_params, set_loss_mode
from src.utils import ROOT, env_info, flatten, git_commit, select_device, set_seed

LOSS_KEYS = ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")
CSV_FIELDS = ["epoch", "train_loss", "val_loss", "train_metric", "val_metric", "lr", "val_map50", "val_map",
              "val_map75", "val_mar100", *[f"train_{k}" for k in LOSS_KEYS], "grad_norm", "epoch_time_s",
              "images_per_s"]


# ------------------------------------------------------------------ optimisation helpers
def make_optimizer(model: nn.Module, t: dict):
    """Param groups: (backbone | head) x (weights with decay | biases and norm params without decay)."""
    groups: dict[tuple, list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = (name.startswith("backbone."), p.ndim > 1)
        groups.setdefault(key, []).append(p)
    param_groups = []
    for (is_backbone, decay), params in groups.items():
        param_groups.append({"params": params, "weight_decay": t["weight_decay"] if decay else 0.0,
                             "lr": t["lr"] * (t.get("backbone_lr_mult", 1.0) if is_backbone else 1.0)})
    if t["optimizer"] == "sgd":
        return torch.optim.SGD(param_groups, lr=t["lr"], momentum=t["momentum"], nesterov=t.get("nesterov", False))
    if t["optimizer"] == "adamw":
        return torch.optim.AdamW(param_groups, lr=t["lr"])
    raise ValueError(f"unknown optimizer {t['optimizer']}")


class LRController:
    """Per-iteration LR: linear warmup, then cosine | multistep | plateau (plateau reacts to the validation metric)."""

    def __init__(self, optimizer, steps_per_epoch: int, t: dict):
        self.opt, self.t, self.it = optimizer, t, 0
        self.total = max(1, steps_per_epoch * t["max_epochs"])
        self.warm = int(min(t.get("warmup_iters", 500), self.total // 3))
        self.schedule = t.get("schedule", "cosine")
        self.plateau_scale, self.best, self.bad = 1.0, -math.inf, 0
        for g in optimizer.param_groups:
            g.setdefault("initial_lr", g["lr"])

    def factor(self, it: int) -> float:
        wf = self.t.get("warmup_factor", 0.001)
        if self.warm and it < self.warm:
            return wf + (1.0 - wf) * it / self.warm
        if self.schedule == "cosine":
            p = (it - self.warm) / max(1, self.total - self.warm)
            lo = self.t.get("lr_min_ratio", 0.01)
            return lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
        if self.schedule == "multistep":
            k = sum(it >= int(m * self.total) for m in self.t.get("milestones", [0.67, 0.89]))
            return self.t.get("gamma", 0.1) ** k
        return self.plateau_scale  # plateau

    def step(self) -> float:
        f = self.factor(self.it)
        for g in self.opt.param_groups:
            g["lr"] = g["initial_lr"] * f
        self.it += 1
        return f

    def on_epoch_end(self, metric: float | None) -> None:
        if self.schedule != "plateau" or metric is None or self.it < self.warm:
            return
        if metric > self.best + 1e-4:
            self.best, self.bad = metric, 0
        else:
            self.bad += 1
            if self.bad >= self.t.get("plateau_patience", 2):
                self.plateau_scale *= self.t.get("plateau_gamma", 0.1)
                self.bad = 0

    def state_dict(self):
        return {"it": self.it, "plateau_scale": self.plateau_scale, "best": self.best, "bad": self.bad}

    def load_state_dict(self, s):
        self.it, self.plateau_scale, self.best, self.bad = s["it"], s["plateau_scale"], s["best"], s["bad"]


class EarlyStopper:
    """Stop when the monitored metric (higher is better) has not improved by min_delta for `patience` epochs."""

    def __init__(self, patience: int, min_delta: float = 0.0, min_epochs: int = 0, enabled: bool = True):
        self.patience, self.min_delta, self.min_epochs, self.enabled = patience, min_delta, min_epochs, enabled
        self.best, self.best_epoch, self.bad = -math.inf, -1, 0

    def step(self, epoch: int, value: float) -> tuple[bool, bool]:
        """Returns (improved, should_stop)."""
        if value > self.best + self.min_delta:
            self.best, self.best_epoch, self.bad = value, epoch, 0
            return True, False
        self.bad += 1
        return False, bool(self.enabled and self.bad >= self.patience and epoch + 1 >= self.min_epochs)


# ------------------------------------------------------------------ train / eval loops
def _to_dev(images, targets, device):
    return diagnose._to_dev(images, targets, device)


def train_one_epoch(model, loader, optimizer, lrc, ema, scaler, device, amp, grad_clip):
    model.train()
    sums = {k: 0.0 for k in LOSS_KEYS}
    n_img = n_it = 0
    gn_sum = 0.0
    params = [p for p in model.parameters() if p.requires_grad]
    for images, targets in loader:
        imgs, tg = _to_dev(images, targets, device)
        lrc.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            losses = model(imgs, tg)
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss {float(loss)}: lower training.lr or check boxes "
                                     f"({ {k: float(v) for k, v in losses.items()} })")
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
        if ema is not None:
            ema.update(model)
        for k in LOSS_KEYS:
            sums[k] += float(losses[k]) * len(imgs)
        n_img += len(imgs)
        n_it += 1
        gn_sum += float(gn)
    out = {k: v / max(n_img, 1) for k, v in sums.items()}
    out["total"] = sum(out.values())
    out["grad_norm"] = gn_sum / max(n_it, 1)
    out["n_images"] = n_img
    return out


@torch.inference_mode()
def predict_loader(model, loader, device):
    model.eval()
    preds, tgts = [], []
    for images, targets in loader:
        outs = model([i.to(device) for i in images])
        for o in outs:
            preds.append({k: o[k].detach().cpu().numpy() for k in ("boxes", "scores", "labels")})
        for t in targets:
            tgts.append({k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in t.items()})
    return preds, tgts


def evaluate_map(model, loader, device, classes):
    preds, tgts = predict_loader(model, loader, device)
    ev = DetectionEvaluator(classes)
    ev.update(preds, tgts)
    return ev.compute(), preds, tgts


@torch.no_grad()
def compute_val_loss(model, loader, device):
    """Validation loss: torchvision only returns losses in train mode, so use train mode with BN/Dropout frozen."""
    set_loss_mode(model)
    tot = {k: 0.0 for k in LOSS_KEYS}
    n = 0
    with torch.random.fork_rng():
        torch.manual_seed(1234)  # same RPN/RoI sampling every epoch -> comparable numbers
        for images, targets in loader:
            imgs, tg = _to_dev(images, targets, device)
            losses = model(imgs, tg)
            for k in LOSS_KEYS:
                tot[k] += float(losses[k]) * len(imgs)
            n += len(imgs)
    model.eval()
    return sum(tot.values()) / max(n, 1)


# ------------------------------------------------------------------ fit
def fit(cfg, loaders, classes, device, out_dir: Path, model, log_mlflow=True):
    t = cfg["training"]
    amp = bool(t["amp"] and device.type == "cuda")
    model.to(device)
    optimizer = make_optimizer(model, t)
    lrc = LRController(optimizer, len(loaders["train"]), t)
    scaler = torch.amp.GradScaler("cuda") if amp else None
    e = t.get("ema", {})
    ema = ModelEMA(model, e.get("decay", 0.999), e.get("tau", 500.0)) if e.get("enabled", False) else None
    eval_model = (lambda: ema.module) if (ema is not None and t.get("eval_ema", True)) else (lambda: model)  # noqa: E731

    has_val = loaders["val"] is not None
    es_cfg = t.get("early_stopping", {})
    stopper = EarlyStopper(es_cfg.get("patience", 3), es_cfg.get("min_delta", 0.0), es_cfg.get("min_epochs", 0),
                           bool(es_cfg.get("enabled", True) and has_val))
    metric_name = cfg["metric"]["name"]
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = ckpt_dir / "best.pt", ckpt_dir / "last.pt"
    start_epoch = 0
    if t.get("resume_from"):
        s = torch.load(t["resume_from"], map_location=device, weights_only=False)
        model.load_state_dict(s["model"]); optimizer.load_state_dict(s["optimizer"]); lrc.load_state_dict(s["lrc"])  # noqa: E702
        if ema is not None and s.get("ema"):
            ema.load_state_dict(s["ema"])
        stopper.best, stopper.best_epoch, stopper.bad = s["best"], s["best_epoch"], s["bad"]
        start_epoch = s["epoch"] + 1

    csv_path = out_dir / "metrics.csv"
    if t.get("resume_from"):  # a resumed run is a new MLflow run: carry the best checkpoint and the history forward
        prev = Path(t["resume_from"]).parent.parent
        if (prev / "checkpoints" / "best.pt").exists() and not best_path.exists():
            shutil.copy2(prev / "checkpoints" / "best.pt", best_path)
        if (prev / "metrics.csv").exists() and not csv_path.exists():
            shutil.copy2(prev / "metrics.csv", csv_path)
    resumed = bool(t.get("resume_from")) and csv_path.exists()
    best_summary = {}
    stop = False
    with open(csv_path, "a" if resumed else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not resumed:
            writer.writeheader()
        for epoch in range(start_epoch, t["max_epochs"]):
            t0 = time.time()
            tr = train_one_epoch(model, loaders["train"], optimizer, lrc, ema, scaler, device, amp, t["grad_clip"])
            row = {"epoch": epoch, "train_loss": tr["total"], "lr": optimizer.param_groups[0]["lr"],
                   "grad_norm": tr["grad_norm"], **{f"train_{k}": tr[k] for k in LOSS_KEYS}}
            vm = None
            if has_val:
                m_eval = eval_model()
                row["val_loss"] = compute_val_loss(m_eval, loaders["val"], device)
                vm, _, _ = evaluate_map(m_eval, loaders["val"], device, classes)
                row.update({"val_metric": vm[metric_name], "val_map50": vm["map50"], "val_map": vm["map"],
                            "val_map75": vm["map75"], "val_mar100": vm["mar100"]})
            if loaders.get("train_eval") is not None:
                tm, _, _ = evaluate_map(eval_model(), loaders["train_eval"], device, classes)
                row["train_metric"] = tm[metric_name]
            row["epoch_time_s"] = time.time() - t0
            row["images_per_s"] = tr["n_images"] / max(row["epoch_time_s"], 1e-9)
            writer.writerow(row)
            f.flush()
            if log_mlflow:
                mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch" and v is not None
                                    and not (isinstance(v, float) and math.isnan(v))}, step=epoch)
            print(f"epoch {epoch:03d} | train loss {row['train_loss']:.3f}"
                  + (f" | val loss {row['val_loss']:.3f} | val {metric_name} {row['val_metric']:.4f}" if has_val else "")
                  + (f" | train {metric_name} {row['train_metric']:.4f}" if "train_metric" in row else "")
                  + f" | lr {row['lr']:.5f} | {row['epoch_time_s']:.0f}s", flush=True)

            improved, stop = (True, False) if not has_val else stopper.step(epoch, row["val_metric"])
            if improved or not has_val:
                if not has_val:
                    stopper.best_epoch = epoch
                else:
                    best_summary = {k: row[k] for k in ("val_metric", "val_loss", "val_map50", "val_map")}
                torch.save({"state_dict": eval_model().state_dict(), "model_cfg": cfg["model"],
                            "num_foreground": len(classes), "classes": classes, "epoch": epoch,
                            "metric_name": metric_name,
                            "val_metric": row.get("val_metric"), "ema": bool(ema is not None and t.get("eval_ema", True))},
                           best_path)
            lrc.on_epoch_end(row.get("val_metric"))
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "lrc": lrc.state_dict(),
                        "ema": ema.state_dict() if ema is not None else None, "epoch": epoch,
                        "best": stopper.best, "best_epoch": stopper.best_epoch, "bad": stopper.bad}, last_path)
            if stop:
                print(f"early stopping at epoch {epoch}: no improvement in val {metric_name} for {stopper.bad} epochs "
                      f"(best epoch {stopper.best_epoch})")
                break
    return {"best_path": best_path, "csv_path": csv_path, "best_epoch": stopper.best_epoch,
            "best_val_metric": stopper.best if has_val else None, "stopped_early": bool(stop),
            "best_summary": best_summary}


def run_training(cfg: dict, env: dict, config_path: str | None = None, extra_tags: dict | None = None) -> dict:
    set_seed(cfg["seed"])
    device = select_device(cfg["training"]["device"])
    ds = build_datasets(cfg)
    loaders = build_loaders(cfg, ds, device)
    classes = ds["classes"]
    model = build_model(cfg["model"], len(classes))

    mlflow.set_tracking_uri(env["tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])
    with mlflow.start_run(run_name=cfg.get("run_name")) as run:
        run_id = run.info.run_id
        out_dir = ROOT / "reports" / "runs" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        mlflow.log_params(flatten(cfg))
        mlflow.log_params({"num_parameters": count_params(model), "num_trainable_parameters":
                           count_params(model, True)})
        mlflow.set_tags({"phase": cfg["phase"], "task": "object-detection", "git_commit": git_commit(),
                         "data_version": ds["data_version"], "metric": cfg["metric"]["name"],
                         "metric_target": str(cfg["metric"]["target_value"]), "test_evaluated": "false",
                         **{f"env.{a}": str(b) for a, b in env_info(device).items()}, **(extra_tags or {})})
        print(f"run {run_id} | device {device} | params {count_params(model):,} "
              f"(trainable {count_params(model, True):,}) | data {ds['data_version']}")

        if cfg["sanity"]["enabled"]:
            report = diagnose.run_sanity_checks(cfg, model, loaders["train"], len(classes), device)
            (out_dir / "sanity.json").write_text(json.dumps(report, indent=2))
            mlflow.log_dict(report, "diagnostics/sanity.json")
            mlflow.log_metrics({"sanity_initial_loss_classifier": report["initial_loss_classifier"],
                                "sanity_overfit_ratio": report["overfit_ratio"]})
            mlflow.set_tag("sanity_passed", str(report["passed"]).lower())
            if cfg["sanity"]["fail_hard"] and not report["passed"]:
                raise RuntimeError(f"sanity checks failed: {json.dumps(report)}")
        set_seed(cfg["seed"])  # sanity checks consume RNG; restart from the configured seed

        res = fit(cfg, loaders, classes, device, out_dir, model)
        mlflow.set_tags({"best_epoch": str(res["best_epoch"]), "stopped_early": str(res["stopped_early"]).lower()})
        mlflow.log_metric("best_epoch", res["best_epoch"])
        has_val = loaders["val"] is not None
        best_val = res["best_val_metric"]
        if has_val:
            mlflow.log_metric("best_val_metric", best_val)

        # best checkpoint -> detailed validation artifacts
        state = torch.load(res["best_path"], map_location="cpu", weights_only=True)
        best = build_model(cfg["model"], len(classes), pretrained="none")
        best.load_state_dict(state["state_dict"])
        best.to(device).eval()
        if has_val:
            vm, preds, tgts = evaluate_map(best, loaders["val"], device, classes)
            mlflow.log_metrics({f"final_val_{k}": vm[k] for k in ("map50_voc07", "map50", "map75", "map", "mar100")})
            (out_dir / "val_per_class_ap50.json").write_text(json.dumps(vm["ap50_per_class"], indent=2))
            errs = error_breakdown(preds, tgts, cfg["diagnostics"]["error_score_thr"])
            (out_dir / "val_error_breakdown.json").write_text(json.dumps(errs, indent=2))
            mlflow.log_dict(errs, "diagnostics/val_error_breakdown.json")
            diagnose.qualitative_grid(best, ds["val"], classes, device, out_dir / "qualitative_val.png")
            ref = diagnose.reference_stats(best, ds["val"], device, cfg["diagnostics"]["error_score_thr"])
            (out_dir / "reference_stats.json").write_text(json.dumps(ref, indent=2))
            diag_json, curves_png = out_dir / "diagnosis.json", out_dir / "curves.png"
            subprocess.run([sys.executable, str(ROOT / "scripts" / "diagnose_curves.py"), str(res["csv_path"]),
                            "--out", str(diag_json), "--plot", str(curves_png),
                            "--target", str(cfg["metric"]["target_value"])], check=False)
            if diag_json.exists():
                advice = diagnose.next_steps(json.loads(diag_json.read_text()))
                (out_dir / "next_steps.json").write_text(json.dumps(advice, indent=2))
                print("diagnosis:", advice["verdict"])
        (out_dir / "class_names.json").write_text(json.dumps(classes))
        for p in out_dir.glob("*"):
            if p.is_file():
                mlflow.log_artifact(str(p))
        mlflow.log_artifact(str(res["best_path"]), "checkpoints")
        if config_path and Path(config_path).exists():
            mlflow.log_artifact(config_path, "config")
        req = ROOT / "requirements.txt"
        if req.exists():
            mlflow.log_artifact(str(req), "config")
        mlflow.log_dict({p: metadata.version(p) for p in ("torch", "torchvision")}, "config/versions.json")
        target = cfg["metric"]["target_value"]
        print(f"best epoch {res['best_epoch']} | best val {cfg['metric']['name']}: {best_val} (target {target}) "
              f"| artifacts: {out_dir}")
        return {"run_id": run_id, "best_val_metric": best_val, "best_epoch": res["best_epoch"],
                "out_dir": str(out_dir), "target_met": bool(best_val is not None and best_val >= target),
                "stopped_early": res["stopped_early"]}


# ------------------------------------------------------------------ benchmark
def benchmark(cfg: dict, steps: int = 3, image_hw=(375, 500)) -> dict:
    """Time training iterations and single-image inference on this machine (no weights are downloaded)."""
    device = select_device(cfg["training"]["device"])
    set_seed(cfg["seed"])
    k = cfg["data"]["num_classes"]
    model = build_model(cfg["model"], k, pretrained="none").to(device)
    opt = make_optimizer(model, cfg["training"])
    bs = cfg["data"]["batch_size"]

    def batch():
        imgs = [torch.rand(3, *image_hw, device=device) for _ in range(bs)]
        tg = [{"boxes": torch.tensor([[30., 40., 220., 300.], [250., 60., 450., 330.]], device=device),
               "labels": torch.tensor([1, min(2, k)], device=device)} for _ in range(bs)]
        return imgs, tg

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
    model.train()
    for i in range(steps + 1):
        imgs, tg = batch()
        if i == 1:
            sync(); t0 = time.time()  # noqa: E702
        opt.zero_grad()
        sum(model(imgs, tg).values()).backward()
        opt.step()
    sync()
    sec_it = (time.time() - t0) / steps
    model.eval()
    with torch.inference_mode():
        model([torch.rand(3, *image_hw, device=device)])
        sync(); t0 = time.time()  # noqa: E702
        for _ in range(3):
            model([torch.rand(3, *image_hw, device=device)])
        sync()
    sec_img = (time.time() - t0) / 3
    n_train = 5011 if cfg["data"]["dataset"] == "voc2007" else cfg["data"].get("synthetic_size", 100)
    n_train = int(n_train * (1 - cfg["data"]["val_fraction"]))
    epoch_min = sec_it * math.ceil(n_train / bs) / 60
    return {"device": str(device), "batch_size": bs, "image_hw": list(image_hw), "sec_per_train_iter": round(sec_it, 3),
            "est_train_minutes_per_epoch": round(epoch_min, 1),
            "est_hours_for_max_epochs": round(epoch_min * cfg["training"]["max_epochs"] / 60, 2),
            "inference_ms_per_image": round(sec_img * 1000, 1),
            "note": "random images, weights not loaded; data loading and per-epoch evaluation add time"}
