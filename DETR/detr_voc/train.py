"""DETR training (paper section 4 / A.4): AdamW, backbone LR 1e-5, transformer LR 1e-4, weight decay 1e-4, gradient clipping 0.1,
one LR step-down, plus validation-based early stopping, EMA option and MLflow logging."""
import contextlib
import csv
import json
import math
import shutil
import time
from collections import defaultdict
from importlib import metadata
from pathlib import Path

import mlflow
import numpy as np
import torch

from detr_voc.config import flatten, on_databricks
from detr_voc.curves import curve_read_rows, diagnose_curves, plot_curves
from detr_voc.data import DetrDataset, SyntheticSource, build_data, build_loaders, collate_train
from detr_voc.diagnose import next_steps, qualitative_grid, run_sanity_checks
from detr_voc.ema import ModelEMA
from detr_voc.infer import postprocess
from detr_voc.loss import HungarianMatcher, SetCriterion
from detr_voc.metrics import DetectionEvaluator, error_breakdown
from detr_voc.model import build_model, count_params
from detr_voc.schedule import PhaseEarlyStopper, PhaseSchedule

CSV_FIELDS = ["epoch", "iteration", "phase", "train_loss", "val_loss", "train_metric", "val_metric", "lr", "train_ce",
              "train_bbox", "train_giou", "val_ce", "val_bbox", "val_giou", "val_map50", "val_map50_voc07", "grad_norm",
              "imgs_per_s", "elapsed_s"]


def git_commit() -> str:
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def select_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        dev = torch.device(pref)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but PyTorch sees no GPU (CPU-only runtime?). On Databricks pick "
                               "the 'Machine Learning' GPU runtime.")
        return dev
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_mlflow(cfg: dict, paths: dict) -> str:
    """Tracking store: Databricks workspace (artifacts forced into <dest>) or SQLite/file under <dest>."""
    m = cfg["mlflow"]
    backend = m.get("backend", "auto")
    if backend == "auto":
        backend = "databricks" if on_databricks() else "sqlite"
    art = str(paths["mlflow_artifacts"])
    if backend == "databricks":
        uri = "databricks"
        art_loc = f"dbfs:{art}" if art.startswith("/Volumes/") else art
    elif backend == "sqlite":
        uri, art_loc = f"sqlite:///{paths['mlflow'] / 'mlflow.db'}", art
    elif backend == "file":
        uri, art_loc = f"file:{paths['mlflow'] / 'mlruns'}", art
    else:
        uri, art_loc = backend, art
    mlflow.set_tracking_uri(uri)
    name = m["experiment_name"]
    if backend == "databricks" and not name.startswith("/"):
        name = "/Shared/" + name
    if mlflow.get_experiment_by_name(name) is None:
        try:
            mlflow.create_experiment(name, artifact_location=art_loc)
        except Exception as e:  # e.g. the experiment path does not exist in the workspace
            raise RuntimeError(f"could not create MLflow experiment {name!r} at {uri}: {e}") from e
    mlflow.set_experiment(name)
    return uri


def amp_context(device, mode):
    if device.type != "cuda" or mode == "none":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if mode == "bf16" else torch.float16)




def targets_to(targets, device):
    return [{"boxes": t["boxes"].to(device, non_blocking=True), "labels": t["labels"].to(device, non_blocking=True)} for t in targets]


def make_criterion(cfg: dict, num_classes: int) -> SetCriterion:
    lc = cfg["loss"]
    w = {"loss_ce": lc["w_ce"], "loss_bbox": lc["w_bbox"], "loss_giou": lc["w_giou"]}
    return SetCriterion(num_classes, HungarianMatcher(lc["cost_class"], lc["cost_bbox"], lc["cost_giou"]), w, lc["eos_coef"])


def evaluate_model(model, loader, device, criterion, cfg, classes, fast=True, keep=False):
    """One pass: loss components + detections -> VOC-protocol metrics (IoU 0.5 only when fast)."""
    model.eval()
    ev = DetectionEvaluator(classes, [0.5] if fast else None)
    sums, n = defaultdict(float), 0
    preds, gts = [], []
    with torch.inference_mode():
        for images, mask, targets, metas in loader:
            images, mask = images.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            with amp_context(device, cfg["schedule"]["amp"]):
                out = model(images, mask)
            ls = criterion(out, targets_to(targets, device))
            for k in ("total", "loss_ce", "loss_bbox", "loss_giou"):
                sums[k] += float(ls[k]) * len(metas)
            n += len(metas)
            dets = postprocess(out, torch.tensor([m["size"] for m in metas]))
            for d, m in zip(dets, metas, strict=True):
                p = {"boxes": d["boxes"].cpu().numpy(), "scores": d["scores"].cpu().numpy(), "labels": d["labels"].cpu().numpy()}
                g = {"boxes": m["boxes"], "labels": m["labels"], "difficult": m["difficult"]}
                ev.update([p], [g])
                if keep:
                    preds.append(p)
                    gts.append(g)
    res = ev.compute()
    res.update({f"loss_{k}" if not k.startswith("loss") else k: v / max(n, 1) for k, v in sums.items()})
    return (res, preds, gts) if keep else res


def save_checkpoint(path, model_state, cfg, classes, epoch, val_metric, ema_used):
    torch.save({"state_dict": model_state, "cfg": cfg, "classes": classes, "num_foreground": len(classes), "epoch": epoch,
                "val_metric": val_metric, "metric_name": cfg["metric"]["name"], "ema": ema_used, "stage": cfg["stage"]}, path)


def fit(cfg, data, loaders, model, criterion, device, out_dir: Path):
    s, es_cfg = cfg["schedule"], cfg["schedule"]["early_stopping"]
    classes, metric_name = data["classes"], cfg["metric"]["name"]
    phases = [(p["lr"], p["epochs"]) for p in s["phases"]]
    optimizer = torch.optim.AdamW(model.param_groups(phases[0][0], s["lr_backbone"], s["weight_decay"]))
    sch = PhaseSchedule(optimizer, phases, s["scale"], 0)
    has_val = loaders["val"] is not None
    es = PhaseEarlyStopper(es_cfg["patience"], es_cfg["min_delta"], bool(es_cfg["enabled"] and has_val))
    ema = ModelEMA(model, s["ema"]["decay"], s["ema"]["tau"]) if s["ema"]["enabled"] else None
    eval_model = ema.module if ema is not None else model
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and s["amp"] == "fp16") else None
    ck = out_dir / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    best_path, last_path, csv_path = ck / "best.pt", ck / "last.pt", out_dir / "metrics.csv"
    epoch = iteration = 0
    if cfg.get("resume_from"):
        st = torch.load(cfg["resume_from"], map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        sch.load_state_dict(st["sched"])
        es.load_state_dict(st["es"])
        if ema is not None and st.get("ema"):
            ema.load_state_dict(st["ema"])
        epoch, iteration = st["epoch"], st["iteration"]
        prev = Path(cfg["resume_from"]).parent.parent   # carry the best checkpoint and the history into this run
        for name, dst in (("checkpoints/best.pt", best_path), ("metrics.csv", csv_path)):
            if (prev / name).exists() and not dst.exists():
                shutil.copy2(prev / name, dst)
    best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
    if best_path.exists():
        best_state = torch.load(best_path, map_location="cpu", weights_only=False)["state_dict"]
    params = [p for p in model.parameters() if p.requires_grad]
    win, n_win, n_imgs, gn_sum = defaultdict(float), 0, 0, 0.0
    t_win = t_start = time.time()
    stop_reason, last_row = "schedule_complete", {}
    resumed = csv_path.exists()
    f = open(csv_path, "a" if resumed else "w", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if not resumed:
        writer.writeheader()
    try:
        while True:
            lr = sch.step()                       # LR for this epoch (the phase LR; backbone gets its own multiple)
            model.train()
            for images, mask, targets in loaders["train"]:
                images, mask, targets = images.to(device, non_blocking=True), mask.to(device, non_blocking=True), targets_to(targets, device)
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, s["amp"]):
                    out = model(images, mask)
                ls = criterion(out, targets)
                loss = ls["total"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at iteration {iteration}: lower schedule.phases[0].lr")
                clip = s["grad_clip"] or float("inf")
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    gn = torch.nn.utils.clip_grad_norm_(params, clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_(params, clip)
                    optimizer.step()
                if ema is not None:
                    ema.update(model)
                iteration += 1
                win["total"] += float(loss.detach())
                for k in ("loss_ce", "loss_bbox", "loss_giou"):
                    win[k] += float(ls[k])
                gn_sum += float(gn)
                n_win += 1
                n_imgs += len(images)
                if iteration % s["log_every"] == 0:
                    mlflow.log_metrics({"train_loss_iter": win["total"] / n_win, "grad_norm": gn_sum / n_win}, step=iteration)
            epoch += 1
            mlflow.log_metrics({"lr": lr, "train_loss_epoch": win["total"] / max(n_win, 1)}, step=epoch)
            if not has_val:
                if sch.phase_done():
                    if sch.is_last_phase:
                        break
                    sch.advance()
                continue
            if epoch % s["eval_every"] != 0 and not sch.phase_done():
                continue
            dt = max(time.time() - t_win, 1e-9)
            row = {"epoch": epoch, "iteration": iteration, "phase": sch.phase, "lr": lr, "train_loss": win["total"] / n_win,
                   "train_ce": win["loss_ce"] / n_win, "train_bbox": win["loss_bbox"] / n_win, "train_giou": win["loss_giou"] / n_win,
                   "grad_norm": gn_sum / n_win, "imgs_per_s": n_imgs / dt, "elapsed_s": time.time() - t_start}
            v = evaluate_model(eval_model, loaders["val"], device, criterion, cfg, classes)
            value = v[metric_name]
            row.update({"val_loss": v["loss_total"], "val_ce": v["loss_ce"], "val_bbox": v["loss_bbox"], "val_giou": v["loss_giou"],
                        "val_metric": value, "val_map50": v["map50"], "val_map50_voc07": v["map50_voc07"]})
            if loaders.get("train_eval") is not None:
                row["train_metric"] = evaluate_model(eval_model, loaders["train_eval"], device, criterion, cfg, classes)[metric_name]
            writer.writerow(row)
            f.flush()
            mlflow.log_metrics({k: x for k, x in row.items() if k not in ("epoch", "iteration") and x is not None
                                and not (isinstance(x, float) and math.isnan(x))}, step=epoch)
            last_row = row
            gap = "" if "train_metric" not in row else f" | gap {row['train_metric'] - value:+.3f}"
            print(f"epoch {epoch:4d} ph {sch.phase} | loss {row['train_loss']:.3f} | val loss {row['val_loss']:.3f} | val {metric_name} "
                  f"{value:.4f}{gap} | lr {lr:.6f} | {row['imgs_per_s']:.0f} img/s", flush=True)
            improved = es.update(value, epoch)
            if improved:
                best_state = {k: x.detach().cpu().clone() for k, x in eval_model.state_dict().items()}
                save_checkpoint(best_path, best_state, cfg, classes, epoch, value, ema is not None)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "sched": sch.state_dict(),
                        "es": es.state_dict(), "ema": ema.state_dict() if ema is not None else None, "epoch": epoch,
                        "iteration": iteration}, last_path)
            win.clear()
            n_win = n_imgs = 0
            gn_sum = 0.0
            t_win = time.time()
            if es.exhausted or sch.phase_done():
                plateau = es.exhausted
                if sch.is_last_phase:
                    stop_reason = "early_stop_plateau" if plateau else "schedule_complete"
                    break
                if plateau and not es_cfg["advance_phase_on_plateau"] and not sch.phase_done():
                    continue
                sch.advance()
                es.reset_phase()
                mlflow.log_metric("phase_change_epoch", epoch, step=epoch)
                if es_cfg["restore_best_on_decay"]:
                    model.load_state_dict(best_state)
                    if ema is not None:
                        ema.module.load_state_dict(best_state)
                    optimizer.state.clear()
                print(f"  -> LR phase {sch.phase} ({'plateau' if plateau else 'phase complete'}), lr {sch.phases[sch.phase][0]}", flush=True)
    finally:
        f.close()
    if not has_val:
        save_checkpoint(best_path, {k: x.detach().cpu() for k, x in eval_model.state_dict().items()}, cfg, classes, epoch, None, ema is not None)
    return {"best_path": best_path, "csv_path": csv_path, "epochs": epoch, "iterations": iteration, "stop_reason": stop_reason,
            "best_val_metric": es.best if has_val else None, "best_epoch": es.best_it, "last_row": last_row}


def run_training(cfg: dict, paths: dict) -> dict:
    set_seed(cfg["seed"])
    device = select_device(cfg["device"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    data = build_data(cfg, paths)
    loaders = build_loaders(cfg, data, device)
    model = build_model(cfg).to(device)
    criterion = make_criterion(cfg, cfg["data"]["num_foreground"]).to(device)
    uri = setup_mlflow(cfg, paths)
    with mlflow.start_run(run_name=cfg.get("run_label") or cfg["stage"]) as run:
        run_id = run.info.run_id
        out_dir = Path(paths["runs"]) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "stage.json").write_text(json.dumps({"stage": cfg["stage"], "status": "running"}))
        mlflow.log_params({k: str(v)[:490] for k, v in flatten({kk: vv for kk, vv in cfg.items() if kk != "dest"}).items()})
        mlflow.log_params({"num_parameters": count_params(model), "num_trainable": count_params(model, True)})
        mlflow.set_tags({"stage": cfg["stage"], "task": "object-detection", "model": "DETR", "git_commit": git_commit(),
                         "data_version": data["data_version"], "metric": cfg["metric"]["name"],
                         "metric_target": str(cfg["metric"]["target_value"]), "test_evaluated": "false", "device": str(device),
                         "tracking_uri": uri, "dest": str(paths["dest"]), "pipeline_step": str(cfg.get("run_label", ""))})
        print(f"run {run_id} | stage {cfg['stage']} | device {device} | params {count_params(model):,} "
              f"(trainable {count_params(model, True):,}) | data {data['data_version']}")
        if cfg["sanity"]["enabled"]:
            rep = run_sanity_checks(cfg, model, criterion, next(iter(loaders["train"])), device)
            (out_dir / "sanity.json").write_text(json.dumps(rep, indent=2))
            mlflow.log_dict(rep, "diagnostics/sanity.json")
            mlflow.set_tag("sanity_passed", str(rep["passed"]).lower())
            print("sanity:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items()})
            if cfg["sanity"]["fail_hard"] and not rep["passed"]:
                raise RuntimeError(f"sanity checks failed: {json.dumps(rep)}")
        set_seed(cfg["seed"])
        res = fit(cfg, data, loaders, model, criterion, device, out_dir)
        mlflow.set_tags({"stop_reason": res["stop_reason"], "best_epoch": str(res["best_epoch"])})
        mlflow.log_metrics({"total_epochs": res["epochs"], "total_iterations": res["iterations"]})
        summary = {"run_id": run_id, "stage": cfg["stage"], "epochs": res["epochs"], "iterations": res["iterations"],
                   "stop_reason": res["stop_reason"], "best_epoch": res["best_epoch"], "best_val_metric": res["best_val_metric"],
                   "out_dir": str(out_dir)}
        state = torch.load(res["best_path"], map_location="cpu", weights_only=False)
        best = build_model(cfg, pretrained=False)
        best.load_state_dict(state["state_dict"])
        best.to(device).eval()
        if loaders["val"] is not None:
            v, preds, gts = evaluate_model(best, loaders["val"], device, criterion, cfg, data["classes"], fast=False, keep=True)
            mlflow.log_metrics({f"final_val_{k}": v[k] for k in ("map50", "map50_voc07", "map75", "map", "mar100")})
            mlflow.log_metric("best_val_metric", res["best_val_metric"])
            (out_dir / "val_per_class_ap50.json").write_text(json.dumps(v["ap50_per_class"], indent=2))
            errs = error_breakdown(preds, gts, 0.5)
            (out_dir / "val_error_breakdown.json").write_text(json.dumps(errs, indent=2))
            mlflow.log_dict(errs, "diagnostics/val_error_breakdown.json")
            if loaders.get("train_eval") is not None:
                tm = evaluate_model(best, loaders["train_eval"], device, criterion, cfg, data["classes"])[cfg["metric"]["name"]]
                gap = tm - v[cfg["metric"]["name"]]
                flag = bool(gap > cfg["schedule"]["overfit_gap_warn"])
                mlflow.log_metrics({"final_train_metric": tm, "generalisation_gap": gap})
                mlflow.set_tag("overfit_warning", str(flag).lower())
                summary.update({"train_metric_at_best": tm, "generalisation_gap": gap, "overfit_warning": flag})
                if flag:
                    print(f"WARNING: train - val {cfg['metric']['name']} gap {gap:.3f} exceeds {cfg['schedule']['overfit_gap_warn']}: "
                          "see next_steps.json (overfitting levers)")
            try:
                qualitative_grid(best, data["val"], data["classes"], device, out_dir / "qualitative_val.png")
            except Exception as e:  # plotting must never fail a finished training run
                print("qualitative grid skipped:", e)
            rows = curve_read_rows(res["csv_path"])
            diag = diagnose_curves(rows, target=cfg["metric"]["target_value"])
            (out_dir / "diagnosis.json").write_text(json.dumps(diag, indent=2))
            (out_dir / "next_steps.json").write_text(json.dumps(next_steps(diag), indent=2))
            try:
                plot_curves(rows, str(out_dir / "curves.png"), cfg["metric"]["target_value"])
            except Exception as e:
                print("curve plot skipped:", e)
            summary["diagnosis"] = diag["verdict"]
            print("diagnosis:", diag["verdict"])
        (out_dir / "class_names.json").write_text(json.dumps(data["classes"]))
        (out_dir / "stage.json").write_text(json.dumps({"stage": cfg["stage"], "status": "complete"}))
        stable = Path(paths["checkpoints"]) / f"{cfg['stage']}_best.pt"
        shutil.copy2(res["best_path"], stable)
        for p in out_dir.glob("*"):
            if p.is_file():
                mlflow.log_artifact(str(p))
        mlflow.log_artifact(str(res["best_path"]), "checkpoints")
        mlflow.log_dict({p: metadata.version(p) for p in ("torch", "torchvision")}, "config/versions.json")
        summary["stable_checkpoint"] = str(stable)
        summary["target_met"] = bool(res["best_val_metric"] is not None and res["best_val_metric"] >= cfg["metric"]["target_value"])
        print(json.dumps(summary, indent=2))
        return summary


def benchmark(cfg: dict, paths: dict, steps: int = 6, loader_batches: int = 12) -> dict:
    """Model step time and augmentation throughput on THIS machine (random images, no downloads)."""
    device = select_device(cfg["device"])
    set_seed(cfg["seed"])
    model = build_model(cfg, pretrained=False).to(device).train()
    criterion = make_criterion(cfg, cfg["data"]["num_foreground"]).to(device)
    opt = torch.optim.AdamW(model.param_groups(1e-4, 1e-5, 1e-4))
    bs, size, mx = cfg["data"]["batch_size"], cfg["aug"]["test_size"], cfg["aug"]["max_size"]
    h, w = size, min(int(size * 4 / 3), mx)
    x, mask = torch.randn(bs, 3, h, w, device=device), torch.zeros(bs, h, w, dtype=torch.bool, device=device)
    tg = [{"boxes": torch.tensor([[0.3, 0.4, 0.2, 0.3], [0.7, 0.6, 0.3, 0.2]], device=device), "labels": torch.tensor([0, 1], device=device)}
          for _ in range(bs)]

    def step():
        opt.zero_grad()
        with amp_context(device, cfg["schedule"]["amp"]):
            out = model(x, mask)
        criterion(out, tg)["total"].backward()
        opt.step()
    for _ in range(2):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = (time.time() - t0) / steps
    ds = DetrDataset(SyntheticSource(max(bs * 6, 48), 20, 500, cfg["seed"]), cfg["aug"], True)
    ld = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=int(cfg["data"]["num_workers"]), collate_fn=collate_train)
    t1, n = None, 0
    for i, _ in enumerate(ld):
        if i == 2:
            t1 = time.time()
        if i >= 2:
            n += bs
        if i >= loader_batches:
            break
    loader_ips = n / max(time.time() - (t1 or time.time()), 1e-9) if t1 else float("nan")
    ips_model = bs / sec
    n_train = {"voc2012": 10386, "coco2017": 118287}.get(cfg["data"]["dataset"], cfg["data"]["synthetic"]["size"])
    epoch_s = n_train / min(ips_model, loader_ips if loader_ips == loader_ips else ips_model)
    sch = cfg["schedule"]
    epochs = sum(round(p["epochs"] * sch["scale"]) for p in sch["phases"])
    out = {"device": str(device), "batch_size": bs, "image_hw": [h, w], "model_step_s": round(sec, 3), "model_img_per_s": round(ips_model, 1),
           "augmentation_img_per_s": round(loader_ips, 1), "bottleneck": "data loading" if loader_ips < ips_model else "model",
           "est_minutes_per_epoch": round(epoch_s / 60, 2), "schedule_epochs_max": epochs,
           "est_hours_if_run_to_the_end": round(epoch_s * epochs / 3600, 2),
           "note": "estimate; early stopping usually ends a run sooner; evaluation time is extra"}
    if device.type == "cuda":
        out["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return out
