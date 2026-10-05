"""Inference entry point: load a registered detector and run it on images (library + CLI)."""
from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from PIL import Image


def to_pil(item) -> Image.Image:
    """Accept a path, PIL image or HxWx3 uint8 array."""
    if isinstance(item, (str, Path)):
        return Image.open(item).convert("RGB")
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    arr = np.asarray(item)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("array input must be uint8 with shape (H, W, 3)")
    return Image.fromarray(arr)


def encode(img: Image.Image, max_side: int = 4096) -> tuple[str, float]:
    """PNG + base64; images larger than max_side are shrunk (returns the scale applied) to bound memory and latency."""
    scale = 1.0
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode(), scale


class Predictor:
    def __init__(self, model_uri: str, tracking_uri: str | None = None, max_side: int = 4096):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.model_uri, self.max_side = model_uri, max_side
        self.model = mlflow.pyfunc.load_model(model_uri)

    def predict(self, images, score_threshold: float = 0.5, max_detections: int = 100) -> list[dict]:
        pil = [to_pil(i) for i in images]
        enc = [encode(p, self.max_side) for p in pil]
        df = pd.DataFrame({"image_b64": [e[0] for e in enc]})
        floor = min(0.05, float(score_threshold))
        t0 = time.perf_counter()
        raw = self.model.predict(df, params={"score_floor": floor})["detections"].tolist()
        ms = (time.perf_counter() - t0) * 1000 / max(len(pil), 1)
        out = []
        for p, (_, scale), r in zip(pil, enc, raw, strict=True):
            dets = [d for d in json.loads(r)["detections"] if d["score"] >= score_threshold][:max_detections]
            if scale != 1.0:  # boxes were computed on the shrunken image: map back to the original pixel grid
                for d in dets:
                    d["box"] = [round(v / scale, 2) for v in d["box"]]
            out.append({"width": p.width, "height": p.height, "detections": dets,
                        "latency_ms_per_image": round(ms, 1)})
        return out
