"""
DomainValidator: Verifies values against domain validation rules and physical constraints.
"""
from typing import Dict, Any, List
import pandas as pd
from .base_validator import BaseValidator


class DomainValidator(BaseValidator):
    def validate(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        issues: List[Dict[str, Any]] = []
        status = "PASS"

        if df is None or len(df) == 0:
            return {
                "validator": "DomainValidator",
                "status": "PASS",
                "total_issues": 0,
                "issues": []
            }

        # 1. Check Value Ranges from Validation Rules
        value_ranges = config.get("value_ranges", {})
        for col, r_spec in value_ranges.items():
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                series = df[col].dropna()
                min_val = r_spec.get("min")
                max_val = r_spec.get("max")
                unit = r_spec.get("unit", "")

                if min_val is not None:
                    viol_low = series[series < min_val]
                    if len(viol_low) > 0:
                        issues.append({
                            "column": col,
                            "type": "range_underflow",
                            "severity": "WARNING",
                            "violations_count": int(len(viol_low)),
                            "bound": min_val,
                            "min_found": float(series.min()),
                            "unit": unit,
                            "message": f"Column '{col}' has {len(viol_low)} rows below minimum expected range {min_val} {unit}."
                        })
                        if status != "FAIL":
                            status = "WARNING"

                if max_val is not None:
                    viol_high = series[series > max_val]
                    if len(viol_high) > 0:
                        issues.append({
                            "column": col,
                            "type": "range_overflow",
                            "severity": "WARNING",
                            "violations_count": int(len(viol_high)),
                            "bound": max_val,
                            "max_found": float(series.max()),
                            "unit": unit,
                            "message": f"Column '{col}' has {len(viol_high)} rows above maximum expected range {max_val} {unit}."
                        })
                        if status != "FAIL":
                            status = "WARNING"

        # 2. Check Physical Constraints (Hard rules)
        constraints = config.get("physical_constraints", [])
        for c in constraints:
            col = c.get("column")
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                series = df[col].dropna()
                rule = c.get("rule")

                if rule == "non_negative":
                    negatives = series[series < 0]
                    if len(negatives) > 0:
                        status = "FAIL"
                        issues.append({
                            "column": col,
                            "type": "physical_constraint_violation",
                            "severity": "FAIL",
                            "violations_count": int(len(negatives)),
                            "message": f"Physical constraint violated in '{col}': values cannot be negative ({len(negatives)} violations found)."
                        })

                if "min" in c and rule != "non_negative":
                    min_b = c["min"]
                    below = series[series < min_b]
                    if len(below) > 0:
                        issues.append({
                            "column": col,
                            "type": "physical_boundary_below",
                            "severity": "WARNING",
                            "violations_count": int(len(below)),
                            "message": f"Values in '{col}' breached physical lower bound {min_b} ({len(below)} rows)."
                        })
                        if status != "FAIL":
                            status = "WARNING"

                if "max" in c:
                    max_b = c["max"]
                    above = series[series > max_b]
                    if len(above) > 0:
                        issues.append({
                            "column": col,
                            "type": "physical_boundary_above",
                            "severity": "WARNING",
                            "violations_count": int(len(above)),
                            "message": f"Values in '{col}' breached physical upper bound {max_b} ({len(above)} rows)."
                        })
                        if status != "FAIL":
                            status = "WARNING"

        return {
            "validator": "DomainValidator",
            "status": status,
            "total_issues": len(issues),
            "issues": issues
        }
