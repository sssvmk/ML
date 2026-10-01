"""
Generic wide-format multivariate time series -> canonical contract (this request: plug in an open, non-cashflow
benchmark dataset). Reads a headerless CSV/TXT of one column per series, one row per period (the shape used by the
standard Exchange-Rate / Electricity / Traffic / Solar-Energy multivariate forecasting benchmarks), and maps each
COLUMN onto its own segment: one endogenous dataset, zero exogenous (the data contract's floor -- see contract.py).

The canonical `Dataset` field is a closed enum of 6 cashflow-specific labels (contract.py); nothing in any algorithm
module branches on what the label MEANS, only on `series_role` and the numeric values, so a non-cashflow series is
mapped onto one of the 6 labels purely as a container, chosen once and used consistently, and never claimed to BE a
cashflow. `company_code` gets a placeholder shared by every series (there is no "entity" concept in this kind of
dataset); `currency` carries the actual, meaningful per-series identity (e.g. a real currency code) so pooling's
static per-series attributes are genuinely informative rather than constant.
"""
from __future__ import annotations
import pandas as pd

from .base import SourceAdapter
from contract import ContractRow, Lineage, SeriesRole, Dataset, rows_to_frame


class GenericMultivariateAdapter(SourceAdapter):
    name = "generic_multivariate"
    rule_version = "generic-multivariate-v1"

    def extract(self, csv_path: str, series_names: list[str], start: str = "1990-01-01", freq: str = "D",
                company_code: str = "GENERIC", dataset: "str | Dataset" = Dataset.GENERIC, sep: str = ",",
                segment_prefix: str = "") -> pd.DataFrame:
        raw = pd.read_csv(csv_path, header=None, sep=sep)
        if raw.shape[1] != len(series_names):
            raise ValueError(f"{csv_path} has {raw.shape[1]} columns, but {len(series_names)} series_names were given")
        dates = pd.date_range(start, periods=len(raw), freq=freq)
        batch_id = f"generic-{csv_path}"
        rows: list[ContractRow] = []
        for col_idx, series_name in enumerate(series_names):
            segment_id = f"{segment_prefix}{series_name}"
            for d, v in zip(dates, raw.iloc[:, col_idx]):
                rows.append(ContractRow(
                    segment_id=segment_id, company_code=company_code, currency=str(series_name),
                    date=d.date(), series_role=SeriesRole.ENDOGENOUS, dataset=dataset, value=float(v),
                    lineage=Lineage(self.name, self.rule_version, batch_id),
                ))
        return rows_to_frame(rows)
