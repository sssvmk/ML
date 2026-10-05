"""Packaging: a model folder (checkpoint + pyfunc), an MLflow pyfunc logged in the run, and a fresh-process reload check."""
import base64
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
import torch
from mlflow.models import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types import ColSpec, ParamSchema, ParamSpec, Schema
from PIL import Image

from detr_voc.config import on_databricks, registered_name
from detr_voc.infer import Predictor
from detr_voc.train import setup_mlflow

# MLflow "model from code": loading it needs this project's code, which MLflow copies next to the model (code_paths).
PYFUNC_TEMPLATE = '''
import base64
import io
import json

import mlflow
import pandas as pd
import torch
from PIL import Image

from {module} import Predictor


class DetrDetector(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.predictor = Predictor(context.artifacts["checkpoint"])

    def predict(self, context, model_input, params=None):
        params = params or {{}}
        thr, k = float(params.get("score_threshold", 0.5)), int(params.get("max_detections", 100))
        rows = []
        for b64 in model_input["image_b64"].tolist():
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            rows.append(json.dumps(self.predictor.predict([img], thr, k)[0]))
        return pd.DataFrame({{"detections": rows}})


mlflow.models.set_model(DetrDetector())
'''

VERIFY_SNIPPET = '''
import json, sys
import mlflow, numpy as np, pandas as pd
uri, tracking, example_csv, expected_json, atol = sys.argv[1:6]
mlflow.set_tracking_uri(tracking)
model = mlflow.pyfunc.load_model(uri)
got = json.loads(model.predict(pd.read_csv(example_csv), params={"score_threshold": 0.0})["detections"].iloc[0])["detections"]
want = json.load(open(expected_json))["detections"]
if len(got) != len(want):
    print("FAIL: %d detections vs %d expected" % (len(got), len(want))); sys.exit(1)
if not got:
    print("OK (no detections on either side)"); sys.exit(0)
err = max(float(np.abs(np.array([d["box"] for d in got]) - np.array([d["box"] for d in want])).max()),
          float(np.abs(np.array([d["score"] for d in got]) - np.array([d["score"] for d in want])).max()))
print("max abs diff %.2e over %d detections" % (err, len(got)))
sys.exit(0 if err <= float(atol) else 1)
'''


def code_reference() -> tuple[str, str]:
    """(path MLflow must ship, module name that provides Predictor): the package folder, or the single-file script."""
    here = Path(__file__).resolve()
    if here.name == "serve.py" and (here.parent / "infer.py").exists():
        return str(here.parent), "detr_voc.infer"
    return str(here), here.stem


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def package_model(cfg: dict, paths: dict, run_id: str, example_image: Image.Image, out_dir) -> dict:
    """Save checkpoint + pyfunc source into out_dir, log the pyfunc model to the run, verify it in a FRESH process."""
    uri = setup_mlflow(cfg, paths)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    local = Path(paths["runs"]) / run_id / "checkpoints" / "best.pt"
    ckpt = local if local.exists() else Path(mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="checkpoints/best.pt", dst_path=str(Path(paths["tmp"]) / f"pkg_{run_id}")))
    shutil.copy2(ckpt, out / "checkpoint.pt")
    code_path, module = code_reference()
    (out / "detr_pyfunc.py").write_text(PYFUNC_TEMPLATE.format(module=module))
    b64 = _png_b64(example_image)
    pd.DataFrame({"image_b64": [b64]}).to_csv(out / "example.csv", index=False)
    (out / "expected.json").write_text(json.dumps(Predictor(out / "checkpoint.pt").predict([example_image], score_threshold=0.0)[0]))
    signature = ModelSignature(inputs=Schema([ColSpec("string", "image_b64")]), outputs=Schema([ColSpec("string", "detections")]),
                               params=ParamSchema([ParamSpec("score_threshold", "float", 0.5), ParamSpec("max_detections", "integer", 100)]))
    import torchvision
    reqs = [f"torch=={torch.__version__.split('+')[0]}",
            f"torchvision=={torchvision.__version__.split('+')[0]}", f"mlflow=={mlflow.__version__}", "numpy", "pandas", "pillow"]
    code_dir = str(Path(code_path).parent)              # MLflow imports the model file while logging: make the module importable
    added = code_dir not in sys.path
    if added:
        sys.path.insert(0, code_dir)
    try:
        with mlflow.start_run(run_id=run_id):
            info = mlflow.pyfunc.log_model(name="model", python_model=str(out / "detr_pyfunc.py"),
                                           artifacts={"checkpoint": str(out / "checkpoint.pt")}, code_paths=[code_path],
                                           signature=signature, input_example=pd.DataFrame({"image_b64": [b64]}), pip_requirements=reqs)
    finally:
        if added:
            sys.path.remove(code_dir)
    proc = subprocess.run([sys.executable, "-c", VERIFY_SNIPPET, info.model_uri, uri, str(out / "example.csv"),
                           str(out / "expected.json"), "0.05"], cwd=str(out), capture_output=True, text=True)
    ok = proc.returncode == 0
    print(proc.stdout.strip(), proc.stderr.strip()[-400:] if not ok else "")
    (out / "README.txt").write_text(
        "DETR model folder\n  checkpoint.pt  full checkpoint (config + classes): Predictor('checkpoint.pt').predict([image])\n"
        f"  MLflow pyfunc  logged in run {run_id} as 'model' (input column image_b64, output column detections)\n")
    return {"model_dir": str(out), "model_uri": info.model_uri, "load_verified": ok, "tracking_uri": uri}


def register_candidate(cfg: dict, paths: dict, run_id: str, example_image: Image.Image, model_name: str | None = None) -> dict:
    """package_model + registry: set the alias `candidate` if the fresh-process check passed."""
    setup_mlflow(cfg, paths)
    if on_databricks() and cfg["mlflow"].get("backend", "auto") in ("auto", "databricks"):
        mlflow.set_registry_uri("databricks-uc")
    name = model_name or registered_name(cfg)
    if on_databricks() and name.count(".") != 2:
        raise ValueError(f"Unity Catalog needs a three-level model name catalog.schema.model, got {name!r}")
    pkg = package_model(cfg, paths, run_id, example_image, Path(paths["tmp"]) / f"register_{run_id}")
    mv = mlflow.register_model(pkg["model_uri"], name)
    client = MlflowClient()
    client.set_model_version_tag(name, mv.version, "load_verified", str(pkg["load_verified"]).lower())
    client.set_model_version_tag(name, mv.version, "source_run_id", run_id)
    if pkg["load_verified"]:
        client.set_registered_model_alias(name, "candidate", mv.version)
    return {"name": name, "version": mv.version, "model_uri": pkg["model_uri"], "load_verified": pkg["load_verified"]}
