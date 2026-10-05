import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image


def test_api_validation_and_response_shape(monkeypatch):
    from src import api

    seen = {}

    class Fake:
        def predict(self, imgs, thr, k):
            seen.update(thr=thr, k=k, size=imgs[0].size)
            return [{"width": 40, "height": 30, "latency_ms_per_image": 1.0,
                     "detections": [{"box": [1, 2, 3, 4], "score": 0.9, "label_id": 1, "label": "aeroplane"}]}]

    monkeypatch.setattr(api, "Predictor", lambda *a, **k: Fake())
    with TestClient(api.app) as c:
        assert c.get("/health").json()["status"] == "ok"
        buf = io.BytesIO()
        Image.fromarray(np.zeros((30, 40, 3), np.uint8)).save(buf, "PNG")
        r = c.post("/predict?score_threshold=0.7&max_detections=5", files={"file": ("a.png", buf.getvalue(), "image/png")})
        assert r.status_code == 200 and r.json()["detections"][0]["label"] == "aeroplane"
        assert seen["thr"] == 0.7 and seen["k"] == 5 and seen["size"] == (40, 30)
        assert c.post("/predict", files={"file": ("a.txt", b"not an image", "text/plain")}).status_code == 400
        big = io.BytesIO(b"0" * (api.CFG["max_upload_bytes"] + 1))
        assert c.post("/predict", files={"file": ("big.bin", big.getvalue(), "application/octet-stream")}).status_code == 413
