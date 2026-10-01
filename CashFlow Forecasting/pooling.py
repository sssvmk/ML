"""
Cross-segment pooling (PRD Overview §1; gap G-03).

One pooled model instance may be trained on MANY segments at once. This module is model-agnostic
plumbing: it assembles the multi-series batch every pooling-capable module receives.

The system is aware of exactly three things about each series:
    segment_id   — the series key (opaque string)
    series_role  — endogenous | exogenous (structural)
    value        — the numeric observation

Three contract fields are carried as opaque static grouping attributes for pooled neural embeddings:
    company_code, currency, dataset

What those values MEAN is the adapter's business. The system never derives business concepts
(direction, inflow/outflow, asset class) from them — it treats them as opaque grouping keys.

  * Series in different currencies are NOT converted (G-03 decision): each series is scaled by its
    own per-series factor v_i = 1 + mean|y| (DeepAR-style).
  * Eligibility is derived from what is actually supplied (n_series, total_obs).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from algorithms.utils import endogenous_series


def series_scale(y: pd.Series) -> float:
    """Per-series scale factor v_i = 1 + mean(|y|) (DeepAR, Salinas et al.)."""
    return float(1.0 + np.abs(np.asarray(y, dtype=float)).mean())


@dataclass
class PooledBatch:
    segments:   dict[str, pd.DataFrame]  # segment_id -> canonical contract rows
    attributes: dict[str, dict]           # segment_id -> {company_code, currency, dataset}
    scales:     dict[str, float]          # segment_id -> per-series scale factor

    @property
    def segment_ids(self) -> list[str]:
        return sorted(self.segments)

    @property
    def n_series(self) -> int:
        return len(self.segments)

    def obs(self, segment_id: str) -> int:
        return int(endogenous_series(self.segments[segment_id]).shape[0])

    @property
    def total_obs(self) -> int:
        """Pooled N: endogenous observations summed over all series."""
        return sum(self.obs(s) for s in self.segments)

    @property
    def min_series_obs(self) -> int:
        return min((self.obs(s) for s in self.segments), default=0)

    def categories(self) -> dict[str, dict[str, int]]:
        """Integer codes for each static attribute (for embeddings in the neural modules).
        Keys are the three generic contract grouping fields: company_code, currency, dataset.
        The system never interprets their values — they are opaque grouping keys."""
        out: dict[str, dict[str, int]] = {}
        for key in ("company_code", "currency", "dataset"):
            vals = sorted({a[key] for a in self.attributes.values()})
            out[key] = {v: i for i, v in enumerate(vals)}
        return out


def assemble_pool(segments: dict[str, pd.DataFrame]) -> PooledBatch:
    """Builds the batch: static attributes (opaque grouping keys) and per-series scales."""
    attrs, scales = {}, {}
    for sid, df in segments.items():
        attrs[sid] = {
            "company_code": str(df["company_code"].iloc[0]),
            "currency":     str(df["currency"].iloc[0]),
            "dataset":      str(df.loc[df["series_role"] == "endogenous", "dataset"].iloc[0]
                                if (df["series_role"] == "endogenous").any() else ""),
        }
        scales[sid] = series_scale(endogenous_series(df))
    return PooledBatch(segments=dict(segments), attributes=attrs, scales=scales)


def slice_pool(segments: dict[str, pd.DataFrame], train_dates: set, train_end,
               future_known: dict | None = None, through=None) -> PooledBatch:
    """
    Training slice of every segment for one backtest fold. Historical exogenous rows are limited
    to <= train_end (no leakage); a segment's FUTURE-KNOWN datasets (inferred from structure) are
    kept through `through` (the fold's test end).
    """
    future_known = future_known or {}
    sliced = {}
    for sid, df in segments.items():
        is_exog = df["series_role"] == "exogenous"
        is_fk   = is_exog & df["dataset"].isin(future_known.get(sid, []))
        keep    = (df["date"].isin(train_dates)
                   | (is_exog & ~is_fk & (df["date"] <= train_end)))
        if through is not None:
            keep = keep | (is_fk & (df["date"] <= through))
        part = df[keep]
        if not part.empty and (part["series_role"] == "endogenous").any():
            sliced[sid] = part
    return assemble_pool(sliced)
