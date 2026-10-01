"""
Source adapter for the Exchange Rate benchmark dataset (Lai et al. 2018 / LSTNet).

The dataset:
    Daily exchange rates of 8 currencies vs USD, covering 1990-01-01 to 2016-10-10
    (7,588 trading days). File is a headerless CSV with one row per day and one
    column per currency in this fixed order:
        0  AUD  Australian Dollar
        1  GBP  British Pound
        2  CAD  Canadian Dollar
        3  CHF  Swiss Franc
        4  CNY  Chinese Yuan
        5  JPY  Japanese Yen
        6  NZD  New Zealand Dollar
        7  SGD  Singapore Dollar

Source:
    Guokun Lai, Wei-Cheng Chang, Yiming Yang, Hanxiao Liu.
    "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks."
    SIGIR 2018. https://github.com/laiguokun/multivariate-time-series-data

Adapter contract output:
    - One segment per currency: segment_id = "FX-<CURRENCY_CODE>"
    - series_role = "endogenous" for every row (no exogenous series in this dataset)
    - dataset label = "FX_Rate" (free string, not a cashflow enum)
    - company_code = "EXCHANGE_RATE_BENCHMARK" (placeholder; no entity concept in this dataset)
    - currency = the ISO 4217 code (AUD, GBP, ...) — meaningful per-series identity for pooling
    - date = calendar date inferred from start + row index, configurable
    - value = the raw FX rate (float, ratio vs USD, no transformation applied here)

No transformations, no imputation. Missing/NaN values in the source will fail contract
validation (>5% missingness threshold) and the segment will be excluded cleanly.
The transformation step is the algorithm's responsibility, not the adapter's.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import SourceAdapter
from contract import ContractRow, Lineage, SeriesRole, rows_to_frame

# Canonical currency order as shipped in the file (do not reorder)
CURRENCIES: list[str] = ["AUD", "GBP", "CAD", "CHF", "CNY", "JPY", "NZD", "SGD"]

# Conventional start date used by every paper that cites this dataset
DEFAULT_START: str = "1990-01-01"

# Exchange-rate-specific dataset label (free string — the contract accepts any non-empty string)
DATASET_LABEL: str = "FX_Rate"


class ExchangeRateAdapter(SourceAdapter):
    """
    Adapter for the 8-currency daily exchange rate benchmark.

    Usage:
        df = ExchangeRateAdapter().extract("path/to/exchange_rate.txt")

    Config-driven usage (config.json -> data_source):
        {
            "adapter": "exchange_rate",
            "params": {
                "file_path": "/data/exchange_rate.txt",
                "start":     "1990-01-01",   // optional, default shown
                "freq":      "D",            // optional, default daily
                "currencies": ["AUD", "GBP"] // optional, default all 8
            }
        }
    """

    name = "exchange_rate"
    rule_version = "exchange-rate-v1"

    def extract(
        self,
        file_path: str,
        start: str = DEFAULT_START,
        freq: str = "D",
        currencies: list[str] | None = None,
        segment_prefix: str = "FX-",
    ) -> pd.DataFrame:
        """
        Args:
            file_path:       Path to the exchange_rate.txt file (headerless, comma-separated).
            start:           ISO date string for the first row (default "1990-01-01").
            freq:            Pandas frequency alias for date generation (default "D" = daily).
                             The source is daily; change only if you have resampled the file.
            currencies:      Subset of CURRENCIES to extract (default: all 8).
                             Must be a subset of ["AUD","GBP","CAD","CHF","CNY","JPY","NZD","SGD"].
            segment_prefix:  Prepended to the currency code to form segment_id (default "FX-").
        Returns:
            Canonical contract DataFrame with CONTRACT_COLUMNS.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Exchange rate file not found: {file_path}\n"
                f"Download from: https://github.com/laiguokun/multivariate-time-series-data"
            )

        raw = pd.read_csv(path, header=None, dtype=float)
        if raw.shape[1] != len(CURRENCIES):
            raise ValueError(
                f"Expected {len(CURRENCIES)} columns ({CURRENCIES}), "
                f"got {raw.shape[1]} — is this the right file?"
            )

        selected = currencies if currencies is not None else CURRENCIES
        unknown = set(selected) - set(CURRENCIES)
        if unknown:
            raise ValueError(
                f"Unknown currency codes: {sorted(unknown)}. "
                f"Valid codes for this dataset: {CURRENCIES}"
            )

        dates = pd.date_range(start=start, periods=len(raw), freq=freq)
        batch_id = f"{path.name}@{start}"

        rows: list[ContractRow] = []
        for code in selected:
            col_idx = CURRENCIES.index(code)
            segment_id = f"{segment_prefix}{code}"
            for date_ts, value in zip(dates, raw.iloc[:, col_idx]):
                rows.append(ContractRow(
                    segment_id=segment_id,
                    company_code="EXCHANGE_RATE_BENCHMARK",
                    currency=code,
                    date=date_ts.date(),
                    series_role=SeriesRole.ENDOGENOUS,
                    dataset=DATASET_LABEL,
                    value=float(value),
                    lineage=Lineage(
                        source_adapter=self.name,
                        rule_version=self.rule_version,
                        extraction_batch_id=batch_id,
                    ),
                ))

        return rows_to_frame(rows)
