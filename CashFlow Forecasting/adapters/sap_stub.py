from __future__ import annotations
import pandas as pd

from .base import SourceAdapter


class SAPAdapter(SourceAdapter):
    """
    Documents the SAP field mapping onto the canonical contract (PRD
    §4.2, resolved questions #1-#5). Raw SAP RFC/HANA connectivity is out
    of scope for this system (Design doc §5.1) and unavailable in this
    environment, so `extract()` is a stub -- the mapping itself is real
    and is what a live implementation would apply after extraction.

    Field mapping (SAP source -> canonical contract):
        BUKRS                        -> company_code
        HWAER (AR/AP items)          -> currency
        WAERS (FEBKO, bank items)    -> currency (validity vs HWAER is
                                         the still-open item, PRD §4.3)
        ZFBDT + ZBD1T/2T/3T cascade  -> date (net due date, recomputed;
                                         swappable via config)
        DMBTR / KWERT                -> value, sign-adjusted by SHKZG
        BUKRS+BELNR+GJAHR+BUZEI      -> item-level dedup key for the
                                         open->cleared transition
                                         (never exposed past this adapter)
    None of these SAP field names ever appear on the canonical contract
    rows this adapter produces (Design doc §5.2 guarantee).
    """

    name = "sap"
    rule_version = "sap-v1"

    def extract(self, **kwargs) -> pd.DataFrame:
        raise NotImplementedError(
            "No live SAP connectivity in this environment. This adapter documents "
            "the field mapping (see class docstring); a deployed version would "
            "apply it to RFC/HANA extraction output and return canonical rows "
            "identical in shape to the synthetic and CSV-fixture adapters."
        )
