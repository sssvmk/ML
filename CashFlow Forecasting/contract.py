"""
Canonical data contract (PRD Section 5.2) and the universal pre-fit
validator (PRD Section 3.2, step 1).

This is the ONLY interface between the Data Module (adapters) and the
Algorithm Module. Adapters produce rows in this shape; nothing else
crosses the boundary. Algorithm modules consume only these rows -- they
never see BUKRS, EBS vs IHC, or any source-specific field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
import pandas as pd
import numpy as np


class SeriesRole(str, Enum):
    ENDOGENOUS = "endogenous"
    EXOGENOUS = "exogenous"


class Dataset(str, Enum):
    """
    Well-known dataset labels used by the cashflow adapters (SyntheticAdapter, SAPAdapter, CSVFixtureAdapter).
    These are OPTIONAL CONSTANTS — any adapter for a non-cashflow source can pass any string as the `dataset`
    field (the contract validator accepts any non-empty string). Adding a new source never requires touching
    this class; the cashflow labels live here purely so existing adapters don't hard-code strings.
    """
    AR_ACTUAL_CLEARED  = "AR_Actual_Cleared"
    AR_EXPECTED_UNPAID = "AR_Expected_Unpaid"
    BANK_INFLOW_EBS    = "Bank_Inflow_EBS"
    AP_ACTUAL_CLEARED  = "AP_Actual_Cleared"
    AP_EXPECTED_UNPAID = "AP_Expected_Unpaid"
    BANK_OUTFLOW_EBS   = "Bank_Outflow_EBS"
    # Generic label for non-cashflow adapters — adapters may also pass any arbitrary string directly.
    GENERIC            = "Generic"


#: PRD §4.2: provenance is ONE nested `lineage` object per row, not three top-level columns (G-14).
LINEAGE_FIELDS = ("source_adapter", "rule_version", "extraction_batch_id")

CONTRACT_COLUMNS = [
    "segment_id",
    "company_code",
    "currency",
    "date",
    "series_role",
    "dataset",
    "value",
    "lineage",
]


@dataclass
class Lineage:
    """Provenance of a row (PRD §4.2): which adapter produced it, under which rule version, in which batch."""

    source_adapter: str
    rule_version: str
    extraction_batch_id: str

    def as_dict(self) -> dict:
        return {"source_adapter": self.source_adapter, "rule_version": self.rule_version,
                "extraction_batch_id": self.extraction_batch_id}


@dataclass
class ContractRow:
    """One row of the canonical contract (PRD Section 5.2)."""

    segment_id: str
    company_code: str
    currency: str
    date: date
    series_role: SeriesRole
    dataset: "str | Dataset"
    value: float
    lineage: Lineage


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings}


def rows_to_frame(rows: list[ContractRow]) -> pd.DataFrame:
    """Convert a list of ContractRow to a DataFrame with CONTRACT_COLUMNS."""
    if not rows:
        return pd.DataFrame(columns=CONTRACT_COLUMNS)
    df = pd.DataFrame(
        [
            {
                "segment_id": r.segment_id,
                "company_code": r.company_code,
                "currency": r.currency,
                "date": pd.Timestamp(r.date),
                "series_role": r.series_role.value,
                "dataset": r.dataset.value if isinstance(r.dataset, Dataset) else str(r.dataset),
                "value": r.value,
                "lineage": r.lineage.as_dict(),
            }
            for r in rows
        ]
    )
    return df.sort_values(["segment_id", "dataset", "date"]).reset_index(drop=True)


def lineage_field(df: pd.DataFrame, key: str) -> pd.Series:
    """One sub-field of the nested `lineage` object as a flat Series (e.g. lineage_field(df, "rule_version"))."""
    if key not in LINEAGE_FIELDS:
        raise KeyError(f"unknown lineage field {key!r}; expected one of {LINEAGE_FIELDS}")
    return df["lineage"].map(lambda d: d.get(key) if isinstance(d, dict) else None)


def upgrade_flat_lineage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a frame stored in the superseded flat form (top-level source_adapter / rule_version /
    extraction_batch_id columns) into the §4.2 nested form. Explicit migration helper only: the validator
    does NOT silently accept the flat form.
    """
    if "lineage" in df.columns or not set(LINEAGE_FIELDS) <= set(df.columns):
        return df
    out = df.copy()
    out["lineage"] = [dict(zip(LINEAGE_FIELDS, vals)) for vals in zip(*(out[f] for f in LINEAGE_FIELDS))]
    return out.drop(columns=list(LINEAGE_FIELDS))


def validate_contract(
    df: pd.DataFrame,
    *,
    min_observations: int = 30,
    require_exogenous_alignment: bool = True,
) -> ValidationResult:
    """
    Universal pre-fit validation (PRD Section 3.2, step 1). Applied
    identically regardless of which algorithm will eventually train on
    this segment's data -- this is the contract validator gate (Section
    5.2): a row set that fails here never reaches an algorithm module.
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing_cols = set(CONTRACT_COLUMNS) - set(df.columns)
    if missing_cols:
        return ValidationResult(ok=False, errors=[f"missing columns: {sorted(missing_cols)}"])

    # dataset: any non-empty string accepted — the six cashflow Dataset enum values are well-known constants
    # but any adapter for a non-cashflow source may use arbitrary labels (the adapter owns its mapping).
    bad_dataset = df["dataset"].isna() | (df["dataset"].astype(str).str.strip() == "")
    if bad_dataset.any():
        errors.append(f"{int(bad_dataset.sum())} rows with missing/empty dataset label")

    if df.empty:
        return ValidationResult(ok=False, errors=["no rows for this segment/batch"])

    # lineage: every row carries the full nested provenance object (PRD §4.2)
    bad_lineage = df["lineage"].map(
        lambda d: not (isinstance(d, dict) and all(isinstance(d.get(k), str) and d.get(k) for k in LINEAGE_FIELDS))
    )
    if bad_lineage.any():
        errors.append(f"{int(bad_lineage.sum())} rows with missing/incomplete lineage "
                      f"(need non-empty {list(LINEAGE_FIELDS)})")

    # target exists and is numeric
    if not pd.api.types.is_numeric_dtype(df["value"]):
        errors.append("value column is not numeric")

    # timestamps valid, correctly ordered, de-duplicated
    if df["date"].isna().any():
        errors.append("null timestamps present")
    dup_key = ["segment_id", "dataset", "series_role", "date"]
    dupes = df.duplicated(subset=dup_key, keep=False)
    if dupes.any():
        errors.append(f"{int(dupes.sum())} duplicated (segment, dataset, date) rows")

    for (seg, ds), g in df.groupby(["segment_id", "dataset"]):
        if not g["date"].is_monotonic_increasing:
            g_sorted_ok = g["date"].sort_values().reset_index(drop=True).equals(
                g["date"].reset_index(drop=True)
            )
            if not g_sorted_ok:
                warnings.append(f"{seg}/{ds}: dates not pre-sorted (re-sorted for use)")

    # missingness
    endog = df[df["series_role"] == SeriesRole.ENDOGENOUS.value]
    if endog["value"].isna().mean() > 0.05:
        errors.append("endogenous missingness exceeds 5%")

    # sufficient temporal coverage / sufficient observations
    for (seg, ds), g in df.groupby(["segment_id", "dataset"]):
        if len(g) < min_observations:
            errors.append(
                f"{seg}/{ds}: only {len(g)} observations, need >= {min_observations}"
            )

    # no future-data leakage: exogenous rows must not extend past the
    # latest endogenous timestamp for the same segment (a stricter check
    # belongs to whichever algorithm needs a future-known feed, Section 4.3)
    if not endog.empty:
        for seg, g in df.groupby("segment_id"):
            seg_endog = g[g["series_role"] == SeriesRole.ENDOGENOUS.value]
            if seg_endog.empty:
                continue
            cutoff = seg_endog["date"].max()
            exog_future = g[
                (g["series_role"] == SeriesRole.EXOGENOUS.value) & (g["date"] > cutoff)
            ]
            if not exog_future.empty and require_exogenous_alignment:
                warnings.append(
                    f"{seg}: exogenous rows extend past the last endogenous date "
                    f"({len(exog_future)} rows) -- confirm these are known-future, "
                    f"not leaked actuals (Section 4.3)"
                )

    # currency / entity consistency
    for seg, g in df.groupby("segment_id"):
        if g["currency"].nunique() > 1:
            errors.append(f"{seg}: multiple currencies in one segment ({g['currency'].unique()})")
        if g["company_code"].nunique() > 1:
            errors.append(f"{seg}: multiple company codes in one segment")

    return ValidationResult(ok=len(errors) == 0, errors=errors, warnings=warnings)
