"""ModelBundle: raw rows in -> probabilities out. Bundles validation + feature engineering + encoding + model +
threshold, so training and serving can never drift apart (no separate preprocessing code to keep in sync)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import FeatureEngineer, FeatureSet
from .schema import Schema
from .utils import dump_json, load_json
from .validation import validate_input


def compute_reference_stats(fe_train: pd.DataFrame, fs: FeatureSet, bins: int = 10) -> dict:
    """Training distribution summary used later for drift monitoring (PSI)."""
    ref = {"numeric": {}, "categorical": {}}
    for c in fs.continuous + fs.ordinal + fs.binary:
        v = pd.to_numeric(fe_train[c], errors="coerce").dropna()
        edges = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        props = np.histogram(v, bins=edges)[0] / len(v)
        ref["numeric"][c] = {"edges": [float(e) for e in edges], "proportions": props.tolist()}
    for c in fs.nominal:
        ref["categorical"][c] = fe_train[c].astype(str).value_counts(normalize=True).to_dict()
    return ref


class ModelBundle:
    def __init__(self, algorithm: str, schema: Schema, feature_engineer: FeatureEngineer, preprocessor, estimator,
                 threshold: float, metadata: dict, reference_stats: dict):
        self.algorithm = algorithm
        self.schema = schema
        self.feature_engineer = feature_engineer
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.threshold = float(threshold)
        self.metadata = metadata
        self.reference_stats = reference_stats

    @property
    def feature_set(self) -> FeatureSet:
        return self.feature_engineer.feature_set

    def transform(self, X_raw: pd.DataFrame) -> np.ndarray:
        clean, _ = validate_input(X_raw, self.schema)
        fe = self.feature_engineer.transform(clean)
        return self.preprocessor.transform(fe[self.feature_set.model_columns])

    def predict_proba(self, X_raw: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(self.transform(X_raw)))[:, 1]

    def predict(self, X_raw: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X_raw) >= self.threshold).astype(int)

    # -- persistence -------------------------------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, d / "bundle.joblib", compress=3)
        dump_json({"algorithm": self.algorithm, "threshold": self.threshold, "schema": self.schema.to_dict(),
                   "feature_set": self.feature_set.to_dict(), "metadata": self.metadata}, d / "bundle_meta.json")
        dump_json(self.reference_stats, d / "reference_stats.json")
        return d

    @staticmethod
    def load(directory: str | Path) -> "ModelBundle":
        return joblib.load(Path(directory) / "bundle.joblib")

    @staticmethod
    def read_meta(directory: str | Path) -> dict:
        return load_json(Path(directory) / "bundle_meta.json")
