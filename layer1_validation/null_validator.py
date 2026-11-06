"""
NullValidator: Evaluates missing values against allowed thresholds.
"""
from typing import Dict, Any, List
import pandas as pd
from .base_validator import BaseValidator


class NullValidator(BaseValidator):
    def validate(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        issues: List[Dict[str, Any]] = []
        status = "PASS"

        if df is None or len(df) == 0:
            return {
                "validator": "NullValidator",
                "status": "PASS",
                "total_issues": 0,
                "issues": []
            }

        missing_threshold = config.get("missing_threshold", 0.05)
        total_rows = len(df)

        for col in df.columns:
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                null_pct = float(null_count / total_rows)
                if null_pct == 1.0:
                    status = "FAIL"
                    issues.append({
                        "column": col,
                        "type": "all_null_column",
                        "severity": "FAIL",
                        "missing_count": null_count,
                        "missing_percent": round(null_pct * 100, 2),
                        "limit": round(missing_threshold * 100, 2),
                        "message": f"Column '{col}' is 100% empty."
                    })
                elif null_pct > missing_threshold:
                    if status != "FAIL":
                        status = "WARNING"
                    issues.append({
                        "column": col,
                        "type": "excessive_nulls",
                        "severity": "WARNING",
                        "missing_count": null_count,
                        "missing_percent": round(null_pct * 100, 2),
                        "limit": round(missing_threshold * 100, 2),
                        "message": f"Column '{col}' has {round(null_pct*100, 2)}% missing values (threshold is {round(missing_threshold*100, 2)}%)."
                    })

        return {
            "validator": "NullValidator",
            "status": status,
            "total_issues": len(issues),
            "issues": issues
        }
