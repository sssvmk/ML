"""One-time test-set evaluation (logged into the training run)."""
from __future__ import annotations

import json

import mlflow
import torch
from mlflow.tracking import MlflowClient

from src.data import build_datasets, build_loaders
from src.metrics import error_breakdown
from src.model import build_model
from src.train import evaluate_map
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
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = build_model(state["model_cfg"], state["num_foreground"], pretrained="none").to(device)
    model.load_state_dict(state["state_dict"])
    res, preds, tgts = evaluate_map(model, loaders["test"], device, ds["classes"])
    errs = error_breakdown(preds, tgts, cfg["diagnostics"]["error_score_thr"])

    out_dir = ROOT / "reports" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_per_class_ap50.json").write_text(json.dumps(res["ap50_per_class"], indent=2))
    metric = cfg["metric"]["name"]
    target = cfg["metric"]["target_value"]
    worst = sorted(((k, v) for k, v in res["ap50_per_class"].items() if v is not None), key=lambda kv: kv[1])[:5]
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"test_{k}": res[k] for k in ("map50_voc07", "map50", "map75", "map", "mar100")})
        mlflow.set_tags({"test_evaluated": "true", "test_evaluation_count": str(count + 1),
                         "test_target_met": str(res[metric] >= target).lower()})
        mlflow.log_artifact(str(out_dir / "test_per_class_ap50.json"), "test")
        mlflow.log_dict(errs, "test/error_breakdown.json")
    summary = {"run_id": run_id, f"test_{metric}": res[metric], "test_map50": res["map50"], "test_map": res["map"],
               "test_mar100": res["mar100"], "target": target, "target_met": res[metric] >= target,
               "evaluation_count": count + 1, "worst_classes_ap50": worst, "error_breakdown": errs}
    print(json.dumps(summary, indent=2))
    return summary
