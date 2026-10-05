"""Package the best checkpoint as an MLflow model, register it, verify it reloads, set the `candidate` alias."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import torch
from mlflow.models import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types import Schema, TensorSpec

from src.data import denormalize
from src.pyfunc_model import CifarClassifier
from src.utils import ROOT, registered_name


def _ver(*names):
    for n in names:
        try:
            return f"{n}=={metadata.version(n).split('+')[0]}"  # drop +rocm/+cu local tag: not on PyPI
        except metadata.PackageNotFoundError:
            continue
    raise RuntimeError(f"none of {names} installed")


def _example_images(cfg, n=5):
    from src.data import build_datasets
    ds = build_datasets(cfg)
    imgs = [denormalize(ds["val"][i][0][None], *ds["norm"])[0].permute(1, 2, 0).mul(255).round().byte().numpy()
            for i in range(n)]
    return np.stack(imgs)


def verify_in_fresh_process(uri: str, example: np.ndarray, expected: np.ndarray, atol=1e-4) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        np.save(Path(tmp) / "x.npy", example)
        np.save(Path(tmp) / "y.npy", expected)
        proc = subprocess.run(
            [sys.executable, str(ROOT / "src" / "verify_load.py"), uri, str(Path(tmp) / "x.npy"),
             str(Path(tmp) / "y.npy"), str(atol)],
            cwd=tmp, capture_output=True, text=True)  # neutral cwd: `src` is importable only via bundled code
    print(proc.stdout.strip(), proc.stderr.strip()[-500:] if proc.returncode else "")
    return proc.returncode == 0


def register_candidate(cfg: dict, env: dict, run_id: str) -> dict:
    mlflow.set_tracking_uri(env["tracking_uri"])
    client = MlflowClient()
    name = registered_name(cfg, env)
    run = client.get_run(run_id)
    ckpt = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="checkpoints/best.pt")
    k = cfg["data"]["num_classes"]
    example = _example_images(cfg)

    # expected output from the original checkpoint, computed independently of the pyfunc wrapper
    from src.data import MEAN, STD, preprocess_uint8_batch
    from src.model import build_model
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    ref = build_model(state["model_cfg"], state["num_classes"])
    ref.load_state_dict(state["state_dict"])
    ref.eval()
    norm = state.get("norm") or {"mean": MEAN, "std": STD}
    with torch.inference_mode():
        expected = ref(preprocess_uint8_batch(example, norm["mean"], norm["std"])).softmax(1).numpy().astype(np.float32)

    signature = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("uint8"), (-1, 32, 32, 3))]),
        outputs=Schema([TensorSpec(np.dtype("float32"), (-1, k))]))
    with mlflow.start_run(run_id=run_id):
        info = mlflow.pyfunc.log_model(
            name="model", python_model=CifarClassifier(), artifacts={"checkpoint": ckpt},
            code_paths=[str(ROOT / "src")], signature=signature, input_example=example,
            pip_requirements=[_ver("mlflow", "mlflow-skinny"), _ver("torch"), _ver("torchvision"),
                              _ver("numpy"), _ver("pillow")])
    ok = verify_in_fresh_process(info.model_uri, example, expected)
    mv = mlflow.register_model(info.model_uri, name)
    tags = {
        "source_run_id": run_id, "data_version": run.data.tags.get("data_version", ""),
        "git_commit": run.data.tags.get("git_commit", ""), "load_verified": str(ok).lower(),
        "validated": "false", "best_val_top1": f"{run.data.metrics.get('best_val_top1', float('nan')):.4f}",
        "test_top1": f"{run.data.metrics['test_top1']:.4f}" if "test_top1" in run.data.metrics else "not evaluated",
    }
    for key, val in tags.items():
        client.set_model_version_tag(name, mv.version, key, val)
    if ok:
        client.set_registered_model_alias(name, "candidate", mv.version)
    else:
        print("load-and-predict check FAILED: alias `candidate` not set")
    out = {"name": name, "version": mv.version, "model_uri": info.model_uri, "load_verified": ok}
    print(out)
    return out
