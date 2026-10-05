"""Package the best checkpoint as an MLflow model, register it, verify it reloads, set the `candidate` alias."""
from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
from mlflow.models import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types import ColSpec, ParamSchema, ParamSpec, Schema
from PIL import Image

from src.data import build_datasets
from src.model import build_model
from src.pyfunc_model import DetectorModel
from src.utils import ROOT, registered_name


def _ver(*names):
    for n in names:
        try:
            return f"{n}=={metadata.version(n).split('+')[0]}"  # drop +rocm/+cu/+cpu local tag: not on PyPI
        except metadata.PackageNotFoundError:
            continue
    raise RuntimeError(f"none of {names} installed")


def _example_png_b64(cfg) -> tuple[str, torch.Tensor]:
    ds = build_datasets(cfg)
    src = ds["val"] if len(ds["val"]) else ds["test"]
    img, _ = src[0]
    arr = (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode(), torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)


def expected_detections(ckpt_path: str, image: torch.Tensor) -> list[dict]:
    """Reference output computed straight from the checkpoint (independent of the pyfunc wrapper), floor 0."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    m = build_model(state["model_cfg"], state["num_foreground"], pretrained="none")
    m.load_state_dict(state["state_dict"])
    m.eval()
    m.roi_heads.score_thresh = 0.0
    with torch.inference_mode():
        out = m([image])[0]
    return [{"box": [round(float(v), 2) for v in b], "score": round(float(s), 5)}
            for b, s in zip(out["boxes"], out["scores"], strict=True)]


def verify_in_fresh_process(uri: str, example_b64: str, expected: list[dict], atol=0.02) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        pd.DataFrame({"image_b64": [example_b64]}).to_csv(Path(tmp) / "x.csv", index=False)
        (Path(tmp) / "y.json").write_text(json.dumps(expected))
        proc = subprocess.run([sys.executable, str(ROOT / "src" / "verify_load.py"), uri, str(Path(tmp) / "x.csv"),
                               str(Path(tmp) / "y.json"), str(atol)], cwd=tmp, capture_output=True, text=True)
    print(proc.stdout.strip(), proc.stderr.strip()[-500:] if proc.returncode else "")
    return proc.returncode == 0


def register_candidate(cfg: dict, env: dict, run_id: str) -> dict:
    mlflow.set_tracking_uri(env["tracking_uri"])
    client = MlflowClient()
    name = registered_name(cfg, env)
    run = client.get_run(run_id)
    ckpt = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="checkpoints/best.pt")
    b64, tensor = _example_png_b64(cfg)
    expected = expected_detections(ckpt, tensor)

    signature = ModelSignature(
        inputs=Schema([ColSpec("string", "image_b64")]), outputs=Schema([ColSpec("string", "detections")]),
        params=ParamSchema([ParamSpec("score_floor", "float", 0.05)]))
    with mlflow.start_run(run_id=run_id):
        info = mlflow.pyfunc.log_model(
            name="model", python_model=DetectorModel(), artifacts={"checkpoint": ckpt},
            code_paths=[str(ROOT / "src")], signature=signature,
            input_example=pd.DataFrame({"image_b64": [b64]}),
            pip_requirements=[_ver("mlflow", "mlflow-skinny"), _ver("torch"), _ver("torchvision"), _ver("numpy"),
                              _ver("pillow"), _ver("pandas")])
    ok = verify_in_fresh_process(info.model_uri, b64, expected)
    mv = mlflow.register_model(info.model_uri, name)
    m = run.data.metrics
    tags = {"source_run_id": run_id, "data_version": run.data.tags.get("data_version", ""),
            "git_commit": run.data.tags.get("git_commit", ""), "load_verified": str(ok).lower(), "validated": "false",
            "best_val_metric": f"{m['best_val_metric']:.4f}" if "best_val_metric" in m else "n/a (refit)",
            "test_metric": f"{m[f'test_{cfg['metric']['name']}']:.4f}" if f"test_{cfg['metric']['name']}" in m
            else "not evaluated"}
    for key, val in tags.items():
        client.set_model_version_tag(name, mv.version, key, val)
    if ok:
        client.set_registered_model_alias(name, "candidate", mv.version)
    else:
        print("load-and-predict check FAILED: alias `candidate` not set")
    out = {"name": name, "version": mv.version, "model_uri": info.model_uri, "load_verified": ok}
    print(out)
    return out
