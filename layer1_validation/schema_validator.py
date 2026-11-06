"""
SchemaValidator: Checks required columns, column duplication, empty dataframes, and data types.
"""
from typing import Dict, Any, List
import pandas as pd
from .base_validator import BaseValidator


class SchemaValidator(BaseValidator):
    def validate(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        issues: List[Dict[str, Any]] = []
        status = "PASS"

        # Check empty dataframe
        if df is None or len(df) == 0:
            return {
                "validator": "SchemaValidator",
                "status": "FAIL",
                "total_issues": 1,
                "issues": [{"type": "empty_dataframe", "message": "Dataset contains 0 rows."}]
            }

        # Check duplicate column names
        cols = list(df.columns)
        duplicates = [col for col in set(cols) if cols.count(col) > 1]
        if duplicates:
            status = "FAIL"
            for dup in duplicates:
                issues.append({
                    "column": dup,
                    "type": "duplicate_column",
                    "severity": "FAIL",
                    "message": f"Duplicate column name detected: '{dup}'"
                })

        # Check required columns
        required_cols = config.get("required_columns", [])
        for req in required_cols:
            if req not in df.columns:
                status = "FAIL"
                issues.append({
                    "column": req,
                    "type": "missing_required_column",
                    "severity": "FAIL",
                    "message": f"Required column '{req}' is missing from the dataset."
                })

        # Check data types if specified
        col_types = config.get("column_types", {})
        for col, expected_type in col_types.items():
            if col in df.columns:
                series = df[col]
                mismatch = False
                found_type = str(series.dtype)

                if expected_type in ["float", "numeric", "number"]:
                    if not pd.api.types.is_numeric_dtype(series):
                        mismatch = True
                elif expected_type == "integer":
                    if not pd.api.types.is_integer_dtype(series) and not (
                        pd.api.types.is_float_dtype(series) and (series.dropna() % 1 == 0).all()
                    ):
                        mismatch = True
                elif expected_type == "datetime":
                    if not pd.api.types.is_datetime64_any_dtype(series):
                        # Attempt soft parse
                        try:
                            pd.to_datetime(series.dropna().head(10))
                        except Exception:
                            mismatch = True

                if mismatch:
                    issues.append({
                        "column": col,
                        "type": "dtype_mismatch",
                        "severity": "WARNING",
                        "expected": expected_type,
                        "found": found_type,
                        "message": f"Column '{col}' expected type '{expected_type}', found '{found_type}'"
                    })
                    if status != "FAIL":
                        status = "WARNING"

        return {
            "validator": "SchemaValidator",
            "status": status,
            "total_issues": len(issues),
            "issues": issues
        }
