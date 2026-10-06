"""Step 3 building blocks: encoding, imputation and normalisation, assembled per algorithm family.

Which family needs what (and why):
  linear / MLP : one-hot nominals + standardised numerics (gradient methods and penalties are scale sensitive)
  tree/boosting: ordinal-coded nominals + raw numerics (splits are invariant to monotone transforms)
All fitted on TRAIN only (inside the algorithm module), then reused unchanged for validation/test/inference.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .features import FeatureSet


def build_preprocessor(fs: FeatureSet, *, nominal: str = "onehot", scale: bool = True) -> ColumnTransformer:
    """nominal: 'onehot' | 'ordinal'.  scale: standardise continuous + ordinal columns."""

    def numeric(strategy="median", do_scale=scale):
        steps = [("impute", SimpleImputer(strategy=strategy))]
        if do_scale:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps)

    if nominal == "onehot":
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    elif nominal == "ordinal":
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
    else:
        raise ValueError(nominal)

    parts = []
    if fs.continuous:
        parts.append(("cont", numeric(), fs.continuous))
    if fs.ordinal:
        parts.append(("ord", numeric(), fs.ordinal))
    if fs.binary:
        parts.append(("bin", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0))]), fs.binary))
    if fs.nominal:
        parts.append(("nom", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                                       ("encode", enc)]), fs.nominal))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=True, sparse_threshold=0.0)


def feature_names(pre: ColumnTransformer) -> list[str]:
    return [str(n) for n in pre.get_feature_names_out()]
