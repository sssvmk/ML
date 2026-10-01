"""
Source adapter base interface (Design doc §5.1). Each adapter owns its
own field mapping and transformation rules; the only thing it produces
is canonical contract rows (contract.CONTRACT_COLUMNS). An adapter never
sees algorithm-specific logic, and no adapter references another.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class SourceAdapter(ABC):
    name: str = "base"
    rule_version: str = "v1"

    @abstractmethod
    def extract(self, **kwargs) -> pd.DataFrame:
        """Return rows already conforming to contract.CONTRACT_COLUMNS."""
        raise NotImplementedError
