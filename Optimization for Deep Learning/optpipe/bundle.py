"""Model bundle (directory with model.pt + bundle.json) and the stand-alone Predictor.

A bundle is everything inference needs: the architecture spec, the weights (standardisation is a
layer inside the model) and metadata.  No training code, config or data is needed to load it."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import DIM, IMG
from .model import build_model


def save_bundle(path, model, meta: dict) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    (path / "bundle.json").write_text(json.dumps({"spec": model.spec, "meta": meta}, indent=2, default=str))
    return path


def load_bundle(path):
    path = Path(path)
    info = json.loads((path / "bundle.json").read_text())
    model = build_model(info["spec"])
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
    return model.eval(), info["meta"]


def read_image(path, invert: str = "auto") -> np.ndarray:
    """PNG/JPG -> float32 [784] in [0,1], white digit on black background (MNIST convention)."""
    from PIL import Image
    img = Image.open(path).convert("L")
    if img.size != (IMG, IMG):
        img = img.resize((IMG, IMG), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32) / 255.0
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]).mean()
    if invert == "yes" or (invert == "auto" and border > 0.5):
        a = 1.0 - a
    return a.reshape(DIM)


class Predictor:
    """Load a bundle directory and classify images."""

    def __init__(self, bundle_dir):
        self.model, self.meta = load_bundle(bundle_dir)

    @torch.no_grad()
    def predict_proba(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1, DIM)
        if x.min() < 0.0 or x.max() > 1.0:
            raise ValueError("pixels must be in [0, 1] (white digit on black background)")
        return F.softmax(self.model(torch.from_numpy(x)), dim=-1).numpy()

    def predict(self, x) -> list[dict]:
        p = self.predict_proba(x)
        out = []
        for row in p:
            top = np.argsort(-row)[:3]
            out.append({"label": int(top[0]), "confidence": float(row[top[0]]),
                        "top3": [{"label": int(i), "p": float(row[i])} for i in top],
                        "probabilities": [float(v) for v in row]})
        return out
