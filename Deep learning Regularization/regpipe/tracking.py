"""MLflow helpers: tracking setup, run metadata, model logging, registry and promotion gates."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from .serving_model import MNISTRegModel

ROOT = Path(__file__).resolve().parent.parent


def setup(cfg: dict, out_dir: Path) -> str:
    uri = cfg["tracking"]["uri"] or f"sqlite:///{(out_dir / 'mlflow.db').resolve()}"
    mlflow.set_tracking_uri(uri)
    name = cfg["tracking"]["experiment"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(out_dir / "mlartifacts").resolve().as_uri())
    mlflow.set_experiment(name)
    return uri


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=ROOT, timeout=5).stdout.strip() or "not-a-git-repo"
    except Exception:
        return "unknown"


def env_tags() -> dict:
    import sklearn
    return {"git_commit": git_commit(), "python": platform.python_version(), "torch": torch.__version__,
            "mlflow": mlflow.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()}


def log_model(bundle_dir: Path, x_example: np.ndarray, name: str = "model"):
    """Log the inference unit (bundle + wrapper + code) as an MLflow pyfunc model with a signature."""
    from .bundle import Predictor
    pred = Predictor(bundle_dir)
    import pandas as pd
    out = pd.DataFrame(pred.predict_proba(x_example), columns=[f"prob_{i}" for i in range(10)])
    out.insert(0, "confidence", out.max(axis=1))
    out.insert(0, "label", out.drop(columns=["confidence"]).values.argmax(1))
    signature = infer_signature(x_example, out)
    return mlflow.pyfunc.log_model(
        name=name, python_model=MNISTRegModel(), artifacts={"bundle": str(bundle_dir)},
        signature=signature, input_example=x_example, code_paths=[str(ROOT / "regpipe")],
        pip_requirements=[f"torch=={torch.__version__.split('+')[0]}", "numpy", "pandas", "mlflow"],
    )


def verify_reload(model_uri: str, x_example: np.ndarray, expected: np.ndarray, tracking_uri: str, tol=1e-5) -> dict:
    """Load-and-predict in a FRESH process; fails if loading breaks or outputs differ from the original."""
    tmp = ROOT / ".verify_tmp"
    tmp.mkdir(exist_ok=True)
    np.save(tmp / "x.npy", x_example)
    code = (
        "import sys, mlflow, numpy as np;"
        f"mlflow.set_tracking_uri({tracking_uri!r});"
        f"m = mlflow.pyfunc.load_model({model_uri!r});"
        f"p = m.predict(np.load({str(tmp / 'x.npy')!r}));"
        "cols=[c for c in p.columns if c.startswith('prob_')];"
        f"np.save({str(tmp / 'p.npy')!r}, p[cols].to_numpy())"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, cwd=str(tmp))
    if r.returncode != 0:
        return {"passed": False, "reason": "load/predict failed in fresh process", "stderr": r.stderr[-800:]}
    got = np.load(tmp / "p.npy")
    diff = float(np.abs(got - expected).max())
    return {"passed": diff <= tol, "max_abs_diff": diff}


def register_candidate(cfg: dict, model_info, tags: dict) -> tuple[str, str]:
    name = cfg["tracking"]["registered_model_name"]
    mv = mlflow.register_model(model_info.model_uri, name)
    client = MlflowClient()
    for k, v in tags.items():
        client.set_model_version_tag(name, mv.version, k, str(v))
    client.set_registered_model_alias(name, "candidate", mv.version)
    return name, str(mv.version)


def evaluate_gates(cfg: dict, name: str, version: str, candidate: dict, reload_check: dict) -> dict:
    """Promotion gates for `champion`.  A human approver is a separate, mandatory requirement."""
    gates = {"reload_check": bool(reload_check.get("passed"))}
    target = cfg["promotion"].get("min_accuracy")
    gates["meets_accuracy_target"] = True if target is None else candidate["test_accuracy"] >= target
    gates["accuracy_target"] = target
    client = MlflowClient()
    try:
        champ = client.get_model_version_by_alias(name, "champion")
        tags = champ.tags
        if tags.get("data_hash") == candidate["data_hash"] and "val_accuracy" in tags:
            gates["beats_champion_on_validation"] = candidate["val_accuracy"] >= float(tags["val_accuracy"])
        else:
            gates["beats_champion_on_validation"] = True
            gates["note_champion"] = "champion was trained on different data/metadata; not comparable"
    except Exception:
        gates["beats_champion_on_validation"] = True   # no champion yet
    gates["all_passed"] = all(v for k, v in gates.items() if k in
                              ("reload_check", "meets_accuracy_target", "beats_champion_on_validation"))
    return gates


def promote(cfg: dict, name: str, version: str, approver: str, gates: dict, out_dir: Path) -> bool:
    client = MlflowClient()
    ok = gates["all_passed"] and bool(approver)
    record = {"model": name, "version": version, "approver": approver, "gates": gates, "promoted": ok}
    approvals = out_dir / "approvals"
    approvals.mkdir(exist_ok=True)
    (approvals / f"{name}_v{version}.json").write_text(json.dumps(record, indent=2))
    if ok:
        client.set_registered_model_alias(name, "champion", version)
        client.set_model_version_tag(name, version, "approved_by", approver)
    return ok
