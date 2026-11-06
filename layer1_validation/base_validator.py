"""
Base Validator Interface for Layer 1: Validation Layer.
Defines the standard result contract adhered to by all validators.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class BaseValidator(ABC):
    """Abstract base class for all data validators."""

    @abstractmethod
    def validate(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the input dataframe against configuration rules.

        Returns:
            dict: {
                "validator": str,
                "status": "PASS" | "WARNING" | "FAIL",
                "total_issues": int,
                "issues": list[dict]
            }
        """
        pass
