"""Inference and packaging: checkpoint predictor, TorchScript export, self-contained MLflow pyfunc, registry."""
import base64
import copy
import io
import json
import shutil
import subprocess
import sys
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

from ssd_voc.boxes import postprocess
from ssd_voc.config import on_databricks, registered_name
from ssd_voc.data import IMAGENET_MEAN, IMAGENET_STD, eval_transform
from ssd_voc.model import build_model
from ssd_voc.train import setup_mlflow

# A standalone MLflow "model from code" file: only torch / torchvision / numpy / pandas / PIL / mlflow are needed to
# load the registered model, never this package. Written to <dest>/tmp when registering.
PYFUNC_SOURCE = '''
import base64
import io
import json

import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.ops import batched_nms


class SSDDetector(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.net = torch.jit.load(context.artifacts["torchscript"], map_location="cpu").eval()
        self.meta = json.load(open(context.artifacts["meta"]))
        self.priors = torch.from_numpy(np.load(context.artifacts["priors"])).float()
        self.mean = torch.tensor(self.meta["mean"]).view(3, 1, 1)
        self.std = torch.tensor(self.meta["std"]).view(3, 1, 1)

    def _decode(self, loc):
        v0, v1 = self.meta["variances"]
        p = self.priors
        cxcy = p[:, :2] + loc[:, :2] * v0 * p[:, 2:]
        wh = p[:, 2:] * torch.exp((loc[:, 2:] * v1).clamp(max=4.0))
        return torch.cat([cxcy - wh / 2, cxcy + wh / 2], 1).clamp(0, 1)

    def predict(self, context, model_input, params=None):
        params = params or {}
        thr = float(params.get("score_threshold", self.meta["score_thresh"]))
        size, rows = self.meta["input_size"], []
        with torch.inference_mode():
            for b64 in model_input["image_b64"].tolist():
                img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                w, h = img.size
                x = torch.from_numpy(np.asarray(img.resize((size, size), Image.BILINEAR)).copy()).permute(2, 0, 1)
                x = ((x.float() / 255.0 - self.mean) / self.std)[None]
                loc, conf = self.net(x)
                probs = conf[0].float().softmax(-1)[:, 1:]
                boxes = self._decode(loc[0].float())
                pi, ci = (probs > thr).nonzero(as_tuple=True)
                sc = probs[pi, ci]
                if sc.numel() > self.meta["pre_nms_top_k"]:
                    top = sc.topk(self.meta["pre_nms_top_k"]).indices
                    pi, ci, sc = pi[top], ci[top], sc[top]
                bx = boxes[pi]
                keep = batched_nms(bx, sc, ci, self.meta["nms_iou"])[: self.meta["max_detections"]]
                scale = torch.tensor([w, h, w, h], dtype=torch.float32)
                dets = [{"box": [round(float(v), 2) for v in (b * scale)], "score": round(float(s), 5),
                         "label_id": int(c) + 1, "label": self.meta["classes"][int(c)]}
                        for b, s, c in zip(bx[keep], sc[keep], ci[keep])]
                rows.append(json.dumps({"width": w, "height": h, "detections": dets}))
        return pd.DataFrame({"detections": rows})


mlflow.models.set_model(SSDDetector())
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


def load_checkpoint(path, device="cpu"):
    st = torch.load(path, map_location="cpu", weights_only=False)
    cfg = st["cfg"]
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(st["state_dict"])
    return model.to(device).eval(), cfg, st["classes"]


class Predictor:
    """Run a trained checkpoint on images (paths, PIL images or HxWx3 uint8 arrays). No MLflow, no downloads."""

    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device)
        self.model, self.cfg, self.classes = load_checkpoint(checkpoint, self.device)
        self.tf = eval_transform(self.cfg["model"]["input_size"])

    @staticmethod
    def _pil(item) -> Image.Image:
        if isinstance(item, (str, Path)):
            return Image.open(item).convert("RGB")
        if isinstance(item, Image.Image):
            return item.convert("RGB")
        arr = np.asarray(item)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError("array input must be uint8 with shape (H, W, 3)")
        return Image.fromarray(arr)

    @torch.inference_mode()
    def predict(self, images, score_threshold=0.5, max_detections=200) -> list[dict]:
        pp, var = self.cfg["postprocess"], tuple(self.cfg["loss"]["variances"])
        out = []
        for item in images:
            img = self._pil(item)
            w, h = img.size
            loc, conf = self.model(self.tf(img)[None].to(self.device))
            d = postprocess(loc, conf, self.model.priors, var, min(score_threshold, pp["score_thresh"]), pp["nms_iou"],
                            max_detections, pp["pre_nms_top_k"])[0]
            keep = d["scores"] >= score_threshold
            scale = torch.tensor([w, h, w, h], dtype=torch.float32, device=d["boxes"].device)
            dets = [{"box": [round(float(v), 2) for v in (b * scale)], "score": round(float(s), 5), "label_id": int(c),
                     "label": self.classes[int(c) - 1]}
                    for b, s, c in zip(d["boxes"][keep], d["scores"][keep], d["labels"][keep], strict=True)]
            out.append({"width": w, "height": h, "detections": dets})
        return out


def export_torchscript(model, path) -> None:
    """Trace the raw network (images -> loc, conf); box decoding and NMS live in the pyfunc."""
    m = copy.deepcopy(model).cpu().eval()
    size = m.input_size
    with torch.no_grad():
        ts = torch.jit.trace(m, torch.zeros(1, 3, size, size), check_trace=False)
    ts.save(str(path))


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def package_model(cfg: dict, paths: dict, run_id: str, example_image: Image.Image, out_dir) -> dict:
    """Save a self-contained model folder (checkpoint, TorchScript, priors, meta, pyfunc source), log it to the run as an
    MLflow pyfunc model and verify that it reloads and predicts identically in a FRESH process."""
    uri = setup_mlflow(cfg, paths)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    local = Path(paths["runs"]) / run_id / "checkpoints" / "best.pt"
    ckpt = local if local.exists() else Path(mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="checkpoints/best.pt", dst_path=str(Path(paths["tmp"]) / f"pkg_{run_id}")))
    shutil.copy2(ckpt, out / "checkpoint.pt")
    model, ccfg, classes = load_checkpoint(out / "checkpoint.pt")
    export_torchscript(model, out / "torchscript.pt")
    np.save(out / "priors.npy", model.priors.numpy())
    pp = ccfg["postprocess"]
    (out / "meta.json").write_text(json.dumps({
        "classes": classes, "input_size": ccfg["model"]["input_size"], "variances": ccfg["loss"]["variances"],
        "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD), "score_thresh": pp["score_thresh"],
        "nms_iou": pp["nms_iou"], "max_detections": pp["max_detections"], "pre_nms_top_k": pp["pre_nms_top_k"]}))
    (out / "ssd_pyfunc.py").write_text(PYFUNC_SOURCE)
    b64 = _png_b64(example_image)
    pd.DataFrame({"image_b64": [b64]}).to_csv(out / "example.csv", index=False)
    (out / "expected.json").write_text(json.dumps(Predictor(out / "checkpoint.pt").predict([example_image], score_threshold=0.0)[0]))
    signature = ModelSignature(inputs=Schema([ColSpec("string", "image_b64")]),
                               outputs=Schema([ColSpec("string", "detections")]),
                               params=ParamSchema([ParamSpec("score_threshold", "float", float(pp["score_thresh"]))]))
    import torchvision
    reqs = [f"torch=={torch.__version__.split('+')[0]}", f"torchvision=={torchvision.__version__.split('+')[0]}",
            f"mlflow=={mlflow.__version__}", "numpy", "pandas", "pillow"]
    with mlflow.start_run(run_id=run_id):
        info = mlflow.pyfunc.log_model(
            name="model", python_model=str(out / "ssd_pyfunc.py"),
            artifacts={"torchscript": str(out / "torchscript.pt"), "priors": str(out / "priors.npy"),
                       "meta": str(out / "meta.json")},
            signature=signature, input_example=pd.DataFrame({"image_b64": [b64]}), pip_requirements=reqs)
    proc = subprocess.run([sys.executable, "-c", VERIFY_SNIPPET, info.model_uri, uri, str(out / "example.csv"),
                           str(out / "expected.json"), "0.05"], cwd=str(out), capture_output=True, text=True)
    ok = proc.returncode == 0
    print(proc.stdout.strip(), proc.stderr.strip()[-400:] if not ok else "")
    (out / "README.txt").write_text(
        "SSD model folder\n  checkpoint.pt   full checkpoint (config + classes); Predictor('checkpoint.pt').predict([image])\n"
        "  torchscript.pt  traced network (images -> loc, conf); priors.npy are the default boxes; meta.json the decoding settings\n"
        f"  MLflow pyfunc   logged in run {run_id} as 'model' (input column image_b64, output column detections)\n")
    return {"model_dir": str(out), "model_uri": info.model_uri, "load_verified": ok, "tracking_uri": uri}


def register_candidate(cfg: dict, paths: dict, run_id: str, example_image: Image.Image, model_name: str | None = None) -> dict:
    """package_model + registry: set the alias `candidate` if the fresh-process check passed."""
    setup_mlflow(cfg, paths)
    if on_databricks() and cfg["mlflow"].get("backend", "auto") in ("auto", "databricks"):
        mlflow.set_registry_uri("databricks-uc")
    name = model_name or registered_name(cfg)
    if on_databricks() and name.count(".") != 2:
        raise ValueError(f"Unity Catalog needs a three-level model name catalog.schema.model, got {name!r}: set "
                         "mlflow.registered_model_name")
    pkg = package_model(cfg, paths, run_id, example_image, Path(paths["tmp"]) / f"register_{run_id}")
    client = MlflowClient()
    mv = mlflow.register_model(pkg["model_uri"], name)
    run = client.get_run(run_id)
    for key, val in {"source_run_id": run_id, "stage": cfg["stage"], "data_version": run.data.tags.get("data_version", ""),
                     "load_verified": str(pkg["load_verified"]).lower(),
                     "best_val_metric": f"{run.data.metrics.get('best_val_metric', float('nan')):.4f}",
                     "test_metric": f"{run.data.metrics[f'test_{cfg['metric']['name']}']:.4f}"
                     if f"test_{cfg['metric']['name']}" in run.data.metrics else "not evaluated"}.items():
        client.set_model_version_tag(name, mv.version, key, val)
    if pkg["load_verified"]:
        client.set_registered_model_alias(name, "candidate", mv.version)
    else:
        print("load-and-predict check FAILED: alias `candidate` not set")
    out = {"name": name, "version": mv.version, "model_uri": pkg["model_uri"], "load_verified": pkg["load_verified"]}
    print(out)
    return out
