"""One-time test-set evaluation (logged into the training run)."""
from __future__ import annotations

import csv
import json

import mlflow
import torch
from mlflow.tracking import MlflowClient

from src.data import build_datasets, build_loaders
from src.model import build_model
from src.train import evaluate_model
from src.utils import ROOT, select_device


def run_test_evaluation(cfg: dict, env: dict, run_id: str, force: bool = False) -> dict:
    mlflow.set_tracking_uri(env["tracking_uri"])
    client = MlflowClient()
    run = client.get_run(run_id)
    count = int(run.data.tags.get("test_evaluation_count", "0"))
    if count >= 1 and not force:
        raise RuntimeError(
            f"run {run_id} was already evaluated on the test set {count} time(s). Tuning on the test set "
            "invalidates it; pass --force only to re-verify, and the count is recorded.")

    device = select_device(cfg["training"]["device"])
    ds = build_datasets(cfg)
    loaders = build_loaders(cfg, ds, device)
    ckpt = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="checkpoints/best.pt")
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model = build_model(state["model_cfg"], state["num_classes"]).to(device)
    model.load_state_dict(state["state_dict"])
    res = evaluate_model(model, loaders["test"], device, collect=True)

    per_class = {}
    for c, name in enumerate(ds["classes"]):
        mask = res["target"] == c
        if mask.any():
            per_class[name] = (res["pred"][mask] == c).float().mean().item()
    worst = sorted(per_class.items(), key=lambda kv: kv[1])[:10]
    out_dir = ROOT / "reports" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "test_per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "accuracy"])
        w.writerows(sorted(per_class.items(), key=lambda kv: kv[1]))

    target = cfg["metric"]["target_value"]
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({"test_top1": res["top1"], "test_top5": res["top5"], "test_loss": res["loss"]})
        mlflow.set_tags({"test_evaluated": "true", "test_evaluation_count": str(count + 1),
                         "test_target_met": str(res["top1"] >= target).lower()})
        mlflow.log_artifact(str(out_dir / "test_per_class.csv"), "test")
        mlflow.log_dict({"worst_10_classes": worst}, "test/worst_classes.json")
    summary = {"run_id": run_id, "test_top1": res["top1"], "test_top5": res["top5"], "test_loss": res["loss"],
               "target": target, "target_met": res["top1"] >= target, "evaluation_count": count + 1,
               "worst_classes": worst[:5]}
    print(json.dumps(summary, indent=2))
    return summary
