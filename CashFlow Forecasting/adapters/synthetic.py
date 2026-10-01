from __future__ import annotations
import numpy as np
import pandas as pd

from .base import SourceAdapter
from contract import ContractRow, Lineage, SeriesRole, Dataset, rows_to_frame

# PRD §3.1 dataset assignment per process: the bank movement is the
# endogenous target; the invoice-side datasets of the same process are
# exogenous.
PROCESS_DATASETS = {
    "AR": {
        "endogenous": Dataset.BANK_INFLOW_EBS,
        "exogenous": [Dataset.AR_ACTUAL_CLEARED, Dataset.AR_EXPECTED_UNPAID],
        "suffix": "inflow",
    },
    "AP": {
        "endogenous": Dataset.BANK_OUTFLOW_EBS,
        "exogenous": [Dataset.AP_ACTUAL_CLEARED, Dataset.AP_EXPECTED_UNPAID],
        "suffix": "outflow",
    },
}


class SyntheticAdapter(SourceAdapter):
    """
    Generates synthetic bank + invoice rows for the AR process (bank inflow
    endogenous; AR_Actual_Cleared / AR_Expected_Unpaid exogenous) and the AP
    process (bank outflow endogenous; AP_* exogenous), per PRD §3.1 -- with
    weekly seasonality, trend and noise. Owns its own generation logic;
    produces nothing but canonical contract rows.
    """

    name = "synthetic"
    rule_version = "synthetic-v1"

    def extract(
        self,
        segment_id: str | None = None,
        company_code: str = "1000",
        currency: str = "INR",
        start: str = "2023-01-01",
        n_days: int = 900,
        seed: int = 42,
        process: str = "AR",
        future_days: int = 0,
        freq: str = "D",
    ) -> pd.DataFrame:
        if process not in PROCESS_DATASETS:
            raise ValueError(f"process must be one of {sorted(PROCESS_DATASETS)}, got {process!r}")
        spec = PROCESS_DATASETS[process]
        segment_id = segment_id or f"{company_code}-{currency}-{spec['suffix']}"
        # AP uses a different random stream so the two processes are not copies of each other
        rng = np.random.default_rng(seed + (0 if process == "AR" else 1000))
        dates = pd.date_range(start, periods=n_days, freq=freq)
        t = np.arange(n_days)
        weekly = 5000 * np.sin(2 * np.pi * t / 7)
        trend = 20 * t
        noise = rng.normal(0, 2000, n_days)
        endog_values = 50000 + trend + weekly + noise
        exog_values = {
            spec["exogenous"][0]: 100 + 10 * np.sin(2 * np.pi * t / 30) + rng.normal(0, 5, n_days),
            spec["exogenous"][1]: 80 + 8 * np.cos(2 * np.pi * t / 30) + rng.normal(0, 4, n_days),
        }
        batch_id = f"synthetic-batch-{process.lower()}-0001"
        rows: list[ContractRow] = []

        def add(ds: Dataset, role: SeriesRole, values) -> None:
            for d, v in zip(dates, values):
                rows.append(ContractRow(
                    segment_id=segment_id, company_code=company_code, currency=currency, date=d.date(),
                    series_role=role, dataset=ds, value=float(v),
                    lineage=Lineage(self.name, self.rule_version, batch_id),
                ))

        add(spec["endogenous"], SeriesRole.ENDOGENOUS, endog_values)
        for ds, vals in exog_values.items():
            add(ds, SeriesRole.EXOGENOUS, vals)
        if future_days > 0:
            # The *_Expected_Unpaid series is aggregated by DUE date, so it is known beyond the last actual: it gets
            # `future_days` extra dated values (drawn after everything else, so future_days=0 output is unchanged).
            ds_fk = spec["exogenous"][1]
            t2 = np.arange(n_days, n_days + future_days)
            extra = 80 + 8 * np.cos(2 * np.pi * t2 / 30) + rng.normal(0, 4, future_days)
            from algorithms.utils import future_dates as _future_dates
            fut_dates = _future_dates(dates[-1], future_days, freq=freq)
            for d, v in zip(fut_dates, extra):
                rows.append(ContractRow(
                    segment_id=segment_id, company_code=company_code, currency=currency, date=d.date(),
                    series_role=SeriesRole.EXOGENOUS, dataset=ds_fk, value=float(v),
                    lineage=Lineage(self.name, self.rule_version, batch_id)))
        return rows_to_frame(rows)

    def extract_all_processes(self, **kwargs) -> dict[str, pd.DataFrame]:
        """One segment per PRD §3.1 process (AR inflow and AP outflow), keyed by segment_id."""
        out = {}
        for proc in PROCESS_DATASETS:
            df = self.extract(process=proc, **kwargs)
            out[df["segment_id"].iloc[0]] = df
        return out
