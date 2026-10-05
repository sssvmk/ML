"""One-time evaluation of the best checkpoint on the untouched test set (logged into the training run)."""
import json
from pathlib import Path

import mlflow
import torch
from mlflow.tracking import MlflowClient

from detr_voc.data import build_data, build_loaders
from detr_voc.metrics import error_breakdown
from detr_voc.model import build_model
from detr_voc.train import evaluate_model, make_criterion, select_device, setup_mlflow


def run_test_evaluation(cfg: dict, paths: dict, run_id: str, force: bool = False) -> dict:
    setup_mlflow(cfg, paths)
    client = MlflowClient()
    count = int(client.get_run(run_id).data.tags.get("test_evaluation_count", "0"))
    if count >= 1 and not force:
        raise RuntimeError(f"run {run_id} was already evaluated on the test set {count} time(s). Tuning on the test set "
                           "invalidates it; pass force=True only to re-verify (the count is recorded).")
    if not cfg["data"].get("test_set") and cfg["data"]["dataset"] != "synthetic":
        raise ValueError("this stage has no test set (COCO stage: use the validation metrics)")
    device = select_device(cfg["device"])
    data = build_data(cfg, paths)
    loaders = build_loaders(cfg, data, device)
    local = Path(paths["runs"]) / run_id / "checkpoints" / "best.pt"
    ckpt = local if local.exists() else Path(mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="checkpoints/best.pt", dst_path=str(Path(paths["tmp"]))))
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()
    criterion = make_criterion(cfg, cfg["data"]["num_foreground"]).to(device)
    res, preds, gts = evaluate_model(model, loaders["test"], device, criterion, cfg, data["classes"], fast=False, keep=True)
    errs = error_breakdown(preds, gts, 0.5)
    out_dir = Path(paths["runs"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_per_class_ap50.json").write_text(json.dumps(res["ap50_per_class"], indent=2))
    metric, target = cfg["metric"]["name"], cfg["metric"]["target_value"]
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"test_{k}": res[k] for k in ("map50", "map50_voc07", "map75", "map", "mar100", "loss_total")})
        mlflow.set_tags({"test_evaluated": "true", "test_evaluation_count": str(count + 1),
                         "test_target_met": str(res[metric] >= target).lower()})
        mlflow.log_artifact(str(out_dir / "test_per_class_ap50.json"), "test")
        mlflow.log_dict(errs, "test/error_breakdown.json")
    worst = sorted(((k, v) for k, v in res["ap50_per_class"].items() if v is not None), key=lambda kv: kv[1])[:5]
    summary = {"run_id": run_id, "test_set": cfg["data"].get("test_set") or "synthetic", f"test_{metric}": res[metric],
               "test_map50_voc07": res["map50_voc07"], "test_map": res["map"], "test_map75": res["map75"], "test_mar100": res["mar100"],
               "target": target, "target_met": res[metric] >= target, "evaluation_count": count + 1,
               "worst_classes_ap50": worst, "error_breakdown": errs}
    print(json.dumps(summary, indent=2))
    return summary
