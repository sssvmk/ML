"""Real-time service:  uvicorn src.api:app --port 8000   (env: MODEL_URI or REGISTERED_MODEL_NAME, MLFLOW_TRACKING_URI)."""
from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.serve import Predictor
from src.utils import ROOT, load_yaml

log = logging.getLogger("detector-api")
CFG = load_yaml(ROOT / "config" / "serve.yaml")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    name = os.environ.get("REGISTERED_MODEL_NAME", CFG["model_name"])
    uri = os.environ.get("MODEL_URI", f"models:/{name}@{CFG['model_alias']}")
    state["predictor"] = Predictor(uri, os.environ.get("MLFLOW_TRACKING_URI"), CFG["max_image_side"])
    state["uri"] = uri
    log.info("loaded %s", uri)
    yield
    state.clear()


app = FastAPI(title="Pascal VOC object detector (Faster R-CNN)", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok" if "predictor" in state else "loading", "model_uri": state.get("uri")}


@app.post("/predict")
async def predict(file: UploadFile = File(...), score_threshold: float | None = None, max_detections: int | None = None):
    data = await file.read()
    if len(data) > CFG["max_upload_bytes"]:
        raise HTTPException(413, "file too large")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "not a valid image") from None
    thr = CFG["default_score_threshold"] if score_threshold is None else min(max(score_threshold, 0.0), 1.0)
    k = CFG["max_detections"] if max_detections is None else max(1, min(max_detections, 300))
    res = state["predictor"].predict([img], thr, k)[0]
    log.info("predict n=%d latency_ms=%s", len(res["detections"]), res["latency_ms_per_image"])
    return {"model_uri": state["uri"], "score_threshold": thr, **res}
