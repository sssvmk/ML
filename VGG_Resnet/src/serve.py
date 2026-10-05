"""Inference entry point: load a registered model and classify images (library + CLI)."""
from __future__ import annotations

import time
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
from PIL import Image

from src.data import pil_to_uint8


def to_uint8_array(item) -> np.ndarray:
    """Accept a path, PIL image or HxWx3 uint8 array."""
    if isinstance(item, (str, Path)):
        item = Image.open(item)
    if isinstance(item, Image.Image):
        return pil_to_uint8(item)
    arr = np.asarray(item)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("array input must be uint8 with shape (H, W, 3)")
    return arr if arr.shape[:2] == (32, 32) else pil_to_uint8(Image.fromarray(arr))


class Predictor:
    def __init__(self, model_uri: str, tracking_uri: str | None = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.model_uri = model_uri
        self.model = mlflow.pyfunc.load_model(model_uri)
        self.class_names = self.model.unwrap_python_model().class_names

    def predict(self, images, topk: int = 5) -> list[dict]:
        batch = np.stack([to_uint8_array(i) for i in images])
        t0 = time.perf_counter()
        probs = np.asarray(self.model.predict(batch))
        latency_ms = (time.perf_counter() - t0) * 1000 / len(batch)
        out = []
        for p in probs:
            idx = np.argsort(-p)[:topk]
            out.append({
                "top_k": [{"class_id": int(i), "label": self.class_names[i], "prob": round(float(p[i]), 5)} for i in idx],
                "latency_ms_per_image": round(latency_ms, 2),
            })
        return out
