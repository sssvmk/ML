"""Real-time service:  uvicorn src.api:app --port 8000   (env: MODEL_URI, MLFLOW_TRACKING_URI)."""
from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.serve import Predictor
from src.utils import ROOT, load_yaml

log = logging.getLogger("vgg-api")
SERVE_CFG = load_yaml(ROOT / "config" / "serve.yaml")
MAX_BYTES = SERVE_CFG.get("max_upload_bytes", 5 * 1024 * 1024)
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    name = os.environ.get("REGISTERED_MODEL_NAME", SERVE_CFG["model_name"])
    uri = os.environ.get("MODEL_URI", f"models:/{name}@{SERVE_CFG['model_alias']}")
    state["predictor"] = Predictor(uri, os.environ.get("MLFLOW_TRACKING_URI"))
    state["uri"] = uri
    log.info("loaded %s", uri)
    yield
    state.clear()


app = FastAPI(title="CIFAR image classifier", lifespan=lifespan)


@app.get("/health")
def health():
    ok = "predictor" in state
    return {"status": "ok" if ok else "loading", "model_uri": state.get("uri")}


@app.post("/predict")
async def predict(file: UploadFile = File(...), topk: int = 5):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "file too large")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "not a valid image") from None
    res = state["predictor"].predict([img], topk=max(1, min(topk, 20)))[0]
    log.info("predict top1=%s latency_ms=%s", res["top_k"][0]["label"], res["latency_ms_per_image"])
    return {"model_uri": state["uri"], **res}
