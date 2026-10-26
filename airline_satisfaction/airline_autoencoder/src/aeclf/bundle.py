"""ModelBundle: raw rows in -> probability / embedding / reconstruction out.

Stored as a directory (no pickled torch objects): encoder.joblib (numpy/pandas only), weights as plain tensors
(torch.save of state_dicts, loadable with weights_only=True), meta.json. Same code path for training-time evaluation,
batch inference and the HTTP service, so preprocessing cannot drift between them.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .models import SatisfactionClassifier
from .schema import Schema
from .lgbm_head import build_features
from .training import build_ae, predict_proba, reconstruct, to_tensors
from .utils import dump_json, load_json
from .validation import validate_input


def compute_reference_stats(df: pd.DataFrame, schema: Schema, bins: int = 10) -> dict:
    ref = {"numeric": {}, "categorical": {}}
    for c in schema.continuous + schema.ordinal:
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        edges = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        ref["numeric"][c] = {"edges": [float(e) for e in edges], "proportions": (np.histogram(v, bins=edges)[0] / len(v)).tolist()}
    for c in schema.nominal:
        ref["categorical"][c] = df[c].astype(str).value_counts(normalize=True).to_dict()
    return ref


class ModelBundle:
    def __init__(self, kind: str, mode: str, encoder, schema: Schema, ae_arch: dict, ae_state: dict, head_params: dict,
                 clf_state: dict | None, threshold: float, metadata: dict, reference_stats: dict,
                 head_type: str = "torch", lgbm=None):
        self.kind, self.mode, self.encoder, self.schema = kind, mode, encoder, schema
        self.ae_arch, self.ae_state, self.head_params, self.clf_state = ae_arch, ae_state, head_params, clf_state
        self.threshold, self.metadata, self.reference_stats = float(threshold), metadata, reference_stats
        self.head_type, self.lgbm = head_type, lgbm
        self._clf = None
        self._ae = None

    # -- models (built lazily) ---------------------------------------------------------------------------------------
    @property
    def autoencoder(self):
        if self.ae_state is None:
            raise RuntimeError("this bundle has no pretrained autoencoder (control model): embed/reconstruct unavailable")
        if self._ae is None:
            self._ae = build_ae(self.ae_arch, self.ae_state).eval()
        return self._ae

    @property
    def classifier(self) -> SatisfactionClassifier:
        if self.head_type != "torch":
            raise RuntimeError(f"this bundle uses a {self.head_type} head, not a neural classification layer")
        if self._clf is None:
            clf = SatisfactionClassifier(build_ae(self.ae_arch), self.head_params["head_hidden"], self.head_params["head_dropout"])
            clf.load_state_dict(self.clf_state)
            self._clf = clf.eval()
        return self._clf

    # -- inference -----------------------------------------------------------------------------------------------------
    def _tensors(self, X_raw: pd.DataFrame):
        clean, _ = validate_input(X_raw, self.schema)
        return clean, to_tensors(self.encoder, clean)

    def predict_proba(self, X_raw: pd.DataFrame) -> np.ndarray:
        t = self._tensors(X_raw)[1]
        if self.head_type == "torch":
            return predict_proba(self.classifier, t)
        ae = self.autoencoder if self.head_type != "lgbm_raw" else None
        return self.lgbm.predict_proba(build_features(ae, t, self.head_type))[:, 1]

    def predict(self, X_raw: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X_raw) >= self.threshold).astype(int)

    @torch.no_grad()
    def embed(self, X_raw: pd.DataFrame) -> np.ndarray:
        """Latent code from the PRETRAINED autoencoder encoder (the representation, independent of the classifier)."""
        t = self._tensors(X_raw)[1]
        mu, _ = self.autoencoder.encode(t.xc, t.xk)
        return mu.numpy()

    def reconstruct(self, X_raw: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Decode rows back to the original columns/units; also returns a per-row reconstruction error (anomaly score)."""
        clean, _ = self._tensors(X_raw)
        return reconstruct(self.autoencoder, self.encoder, clean)

    # -- persistence -----------------------------------------------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.encoder, d / "encoder.joblib", compress=3)
        torch.save({"autoencoder": self.ae_state, "classifier": self.clf_state}, d / "weights.pt")
        dump_json({"kind": self.kind, "mode": self.mode, "ae_arch": self.ae_arch, "head_params": self.head_params,
                   "threshold": self.threshold, "schema": self.schema.to_dict(), "metadata": self.metadata, "head_type": self.head_type}, d / "meta.json")
        if self.lgbm is not None:
            joblib.dump(self.lgbm, d / "lgbm.joblib", compress=3)
        dump_json(self.reference_stats, d / "reference_stats.json")
        return d

    @staticmethod
    def load(directory: str | Path) -> "ModelBundle":
        d = Path(directory)
        meta = load_json(d / "meta.json")
        w = torch.load(d / "weights.pt", map_location="cpu", weights_only=True)
        lg = joblib.load(d / "lgbm.joblib") if (d / "lgbm.joblib").exists() else None
        return ModelBundle(meta["kind"], meta["mode"], joblib.load(d / "encoder.joblib"), Schema.from_dict(meta["schema"]),
                           meta["ae_arch"], w["autoencoder"], meta["head_params"], w["classifier"], meta["threshold"],
                           meta["metadata"], load_json(d / "reference_stats.json"), meta.get("head_type", "torch"), lg)

    @staticmethod
    def read_meta(directory: str | Path) -> dict:
        return load_json(Path(directory) / "meta.json")
