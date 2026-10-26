"""Step 3: encoding / normalisation of the mixed-type table, fitted on TRAIN only.

Continuous columns -> (log1p if non-negative and heavily skewed) -> standardise; missing -> train median.
Categorical columns (nominal, and by default the 0-5 ratings) -> integer codes; the LAST code of every column is
reserved for 'unknown / missing', so unseen categories at inference never crash the network.

Why ratings are categorical by default: a rating of 0 means 'not applicable', not 'worst', so treating the scale
as a number would make the autoencoder reconstruct a meaningless average. As classes they are embedded and
reconstructed with cross-entropy. `encoding.ordinal_as: numeric` switches this off.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import Schema

UNKNOWN = "__unknown__"


def _cat_key(s: pd.Series, ordinal: bool) -> pd.Series:
    if ordinal:
        num = pd.to_numeric(s, errors="coerce")
        return num.round().astype("Int64").astype(str).where(num.notna(), None)
    return s.astype("object").where(s.notna(), None).map(lambda v: v if v is None else str(v).strip())


class TabularEncoder:
    def __init__(self, schema: Schema, ordinal_as: str = "categorical", skew_threshold: float = 2.0):
        self.schema, self.ordinal_as, self.skew_threshold = schema, ordinal_as, skew_threshold
        self.cont_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.ordinal_cats: set[str] = set()
        self.cont_stats: dict = {}
        self.levels: dict = {}
        self.cards: list[int] = []

    def fit(self, df: pd.DataFrame) -> "TabularEncoder":
        s = self.schema
        self.cont_cols = list(s.continuous) + (list(s.ordinal) if self.ordinal_as == "numeric" else [])
        self.cat_cols = list(s.nominal) + (list(s.ordinal) if self.ordinal_as == "categorical" else [])
        self.ordinal_cats = set(s.ordinal) if self.ordinal_as == "categorical" else set()
        for c in self.cont_cols:
            v = pd.to_numeric(df[c], errors="coerce")
            log = bool(v.min() >= 0 and abs(v.skew()) > self.skew_threshold)
            t = np.log1p(v.clip(lower=0)) if log else v
            std = float(t.std())
            self.cont_stats[c] = {"log": log, "mean": float(t.mean()), "std": std if std > 0 else 1.0,
                                  "fill": float(t.median()), "orig_mean": float(v.mean())}
        for c in self.cat_cols:
            key = _cat_key(df[c], c in self.ordinal_cats).dropna()
            vals = list(key.unique())
            vals.sort(key=(lambda x: float(x)) if c in self.ordinal_cats else str)
            counts = key.value_counts()
            self.levels[c] = vals
            self.cont_stats.setdefault("__majority__", {})[c] = vals.index(counts.index[0]) if len(vals) else 0
        self.cards = [len(self.levels[c]) + 1 for c in self.cat_cols]
        return self

    @property
    def unknown_codes(self) -> list[int]:
        return [c - 1 for c in self.cards]

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n = len(df)
        xc = np.zeros((n, len(self.cont_cols)), dtype=np.float32)
        for j, c in enumerate(self.cont_cols):
            st = self.cont_stats[c]
            v = pd.to_numeric(df[c], errors="coerce")
            t = np.log1p(v.clip(lower=0)) if st["log"] else v
            xc[:, j] = ((t.fillna(st["fill"]) - st["mean"]) / st["std"]).to_numpy(np.float32)
        xk = np.zeros((n, len(self.cat_cols)), dtype=np.int64)
        for j, c in enumerate(self.cat_cols):
            idx = {lv: i for i, lv in enumerate(self.levels[c])}
            key = _cat_key(df[c], c in self.ordinal_cats)
            xk[:, j] = key.map(idx).fillna(len(self.levels[c])).to_numpy(np.int64)
        return xc, xk

    # -- decoding back to the original table ---------------------------------------------------------------
    def inverse_cont(self, xc: np.ndarray) -> pd.DataFrame:
        out = {}
        for j, c in enumerate(self.cont_cols):
            st = self.cont_stats[c]
            t = xc[:, j] * st["std"] + st["mean"]
            out[c] = np.clip(np.expm1(t), 0, None) if st["log"] else t
        return pd.DataFrame(out)

    def decode_cat(self, codes: np.ndarray) -> pd.DataFrame:
        out = {}
        for j, c in enumerate(self.cat_cols):
            lv = self.levels[c] + [UNKNOWN]
            labels = np.array(lv, dtype=object)[codes[:, j]]
            out[c] = pd.to_numeric(pd.Series(labels), errors="coerce") if c in self.ordinal_cats else labels
        return pd.DataFrame(out)

    def majority_codes(self) -> np.ndarray:
        return np.array([self.cont_stats["__majority__"][c] for c in self.cat_cols])
