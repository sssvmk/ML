from __future__ import annotations
import pandas as pd

from .base import SourceAdapter
from contract import ContractRow, Lineage, SeriesRole, Dataset, rows_to_frame


class CSVFixtureAdapter(SourceAdapter):
    """
    Reads a test-fixture CSV in its OWN column names and maps them onto
    the canonical contract. This is deliberately a different column
    vocabulary than SAP's, to prove the contract carries no source-
    specific terms: segment,company,ccy,as_of,role,series,amount
    """

    name = "csv_fixture"
    rule_version = "csv-fixture-v1"

    REQUIRED_COLUMNS = ["segment", "company", "ccy", "as_of", "role", "series", "amount"]

    def extract(self, csv_path: str) -> pd.DataFrame:
        raw = pd.read_csv(csv_path)
        missing = set(self.REQUIRED_COLUMNS) - set(raw.columns)
        if missing:
            raise ValueError(f"CSV fixture missing columns: {missing}")

        rows: list[ContractRow] = []
        batch_id = f"csv-{csv_path}"
        for _, r in raw.iterrows():
            rows.append(
                ContractRow(
                    segment_id=str(r["segment"]),
                    company_code=str(r["company"]),
                    currency=str(r["ccy"]),
                    date=pd.Timestamp(r["as_of"]).date(),
                    series_role=SeriesRole(r["role"]),
                    dataset=Dataset(r["series"]),
                    value=float(r["amount"]),
                    lineage=Lineage(self.name, self.rule_version, batch_id),
                )
            )
        return rows_to_frame(rows)
