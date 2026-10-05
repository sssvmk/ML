import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image


def test_api_validation(monkeypatch):
    from src import api

    class Fake:
        def predict(self, imgs, topk=5):
            return [{"top_k": [{"class_id": 0, "label": "x", "prob": 1.0}], "latency_ms_per_image": 1.0}]

    monkeypatch.setattr(api, "Predictor", lambda *a, **k: Fake())
    with TestClient(api.app) as c:
        assert c.get("/health").json()["status"] == "ok"
        buf = io.BytesIO()
        Image.fromarray(np.zeros((40, 40, 3), np.uint8)).save(buf, "PNG")
        r = c.post("/predict", files={"file": ("a.png", buf.getvalue(), "image/png")})
        assert r.status_code == 200 and r.json()["top_k"][0]["label"] == "x"
        assert c.post("/predict", files={"file": ("a.txt", b"not an image", "text/plain")}).status_code == 400
