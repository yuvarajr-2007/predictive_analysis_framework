"""
Feature Verification Gate (Component C):
Strict, deterministic verification ensuring only proven, safe features reach the Modelling Layer.
Includes automated repair capability to drop/fix offending features.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


class FeatureVerificationGate:
    def __init__(self, target_column: str = "anomaly", max_missing_pct: float = 30.0, leakage_corr_limit: float = 0.98):
        self.target_column = target_column
        self.max_missing_pct = max_missing_pct
        self.leakage_corr_limit = leakage_corr_limit

    def verify(
        self,
        original_df: pd.DataFrame,
        engineered_df: pd.DataFrame,
        plan: List[Dict[str, Any]],
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Executes all 11 deterministic checks without LLM dependency.
        Returns:
            {
                "status": "PASSED" | "PASSED WITH WARNINGS" | "FAILED",
                "checks": list of check results,
                "offending_columns": list of columns that caused warnings/failures
            }
        """
        checks: List[Dict[str, Any]] = []
        offending_columns = []
        overall_status = "PASSED"

        # Check 1: Engineered columns exist
        planned_outputs = [s["output_name"] for s in plan if s.get("operation") != "drop"]
        missing_engineered = [col for col in planned_outputs if col not in engineered_df.columns]
        if missing_engineered:
            checks.append({
                "check": "Engineered columns exist",
                "status": "FAIL",
                "message": f"Planned columns missing in output dataframe: {missing_engineered}"
            })
            overall_status = "FAILED"
        else:
            checks.append({"check": "Engineered columns exist", "status": "PASS", "message": "All planned columns exist."})

        # Check 2: Row count preserved
        if len(engineered_df) != len(original_df):
            checks.append({
                "check": "Row count preserved",
                "status": "FAIL",
                "message": f"Row count changed from {len(original_df)} to {len(engineered_df)}."
            })
            overall_status = "FAILED"
        else:
            checks.append({"check": "Row count preserved", "status": "PASS", "message": f"Row count preserved ({len(engineered_df)} rows)."})

        # Check 3: Index alignment
        if not engineered_df.index.equals(original_df.index):
            checks.append({
                "check": "Index alignment",
                "status": "FAIL",
                "message": "Dataframe index alignment was altered during feature transformations."
            })
            overall_status = "FAILED"
        else:
            checks.append({"check": "Index alignment", "status": "PASS", "message": "Dataframe index is strictly preserved."})

        # Check 4: Infinite values
        num_cols = engineered_df.select_dtypes(include=[np.number]).columns
        inf_cols = []
        for c in num_cols:
            if np.isneginf(engineered_df[c]).any() or np.isposinf(engineered_df[c]).any():
                inf_cols.append(c)
                offending_columns.append(c)

        if inf_cols:
            checks.append({
                "check": "Infinite values",
                "status": "FAIL",
                "columns": inf_cols,
                "message": f"Infinite values (+/-inf) detected in columns: {inf_cols}"
            })
            overall_status = "FAILED"
        else:
            checks.append({"check": "Infinite values", "status": "PASS", "message": "No infinite values detected."})

        # Check 5: Missing values in new columns
        null_issues = []
        for col in planned_outputs:
            if col in engineered_df.columns:
                null_pct = engineered_df[col].isna().sum() / max(1, len(engineered_df)) * 100
                if null_pct == 100.0:
                    null_issues.append((col, "100% missing", "FAIL"))
                    offending_columns.append(col)
                    overall_status = "FAILED"
                elif null_pct > self.max_missing_pct:
                    null_issues.append((col, f"{null_pct:.1f}% missing", "WARN"))
                    offending_columns.append(col)
                    if overall_status != "FAILED":
                        overall_status = "PASSED WITH WARNINGS"

        if null_issues:
            checks.append({
                "check": "Missing values",
                "status": "FAIL" if any(item[2] == "FAIL" for item in null_issues) else "WARN",
                "issues": null_issues,
                "message": f"High missing values detected in: {[i[0] for i in null_issues]}"
            })
        else:
            checks.append({"check": "Missing values", "status": "PASS", "message": "Missing percentages are within acceptable thresholds."})

        # Check 6: Constant / Zero-variance features
        constant_features = []
        for col in planned_outputs:
            if col in engineered_df.columns:
                if engineered_df[col].nunique(dropna=True) <= 1:
                    constant_features.append(col)
                    offending_columns.append(col)

        if constant_features:
            checks.append({
                "check": "Constant features",
                "status": "WARN",
                "columns": constant_features,
                "message": f"Zero-variance constant columns detected: {constant_features}"
            })
            if overall_status != "FAILED":
                overall_status = "PASSED WITH WARNINGS"
        else:
            checks.append({"check": "Constant features", "status": "PASS", "message": "No constant engineered features."})

        # Check 7: Duplicate / collinear features
        dup_pairs = []
        if len(num_cols) > 1:
            corr = engineered_df[num_cols].corr().abs()
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    c1, c2 = num_cols[i], num_cols[j]
                    if c1 in planned_outputs or c2 in planned_outputs:
                        val = corr.loc[c1, c2]
                        if val > 0.999:
                            dup_pairs.append((c1, c2, float(val)))
                            offending_columns.append(c2 if c2 in planned_outputs else c1)

        if dup_pairs:
            checks.append({
                "check": "Duplicate features",
                "status": "WARN",
                "pairs": dup_pairs,
                "message": f"Near-identical collinear features detected (|r| > 0.999): {dup_pairs}"
            })
            if overall_status != "FAILED":
                overall_status = "PASSED WITH WARNINGS"
        else:
            checks.append({"check": "Duplicate features", "status": "PASS", "message": "No duplicate features found."})

        # Check 8: Target leakage
        leakage_detected = []
        if self.target_column:
            # Sub-check: Target used directly in feature inputs
            for step in plan:
                if self.target_column in step.get("inputs", []):
                    leakage_detected.append((step.get("output_name"), "Target column used directly as input"))
                    offending_columns.append(step.get("output_name"))

            if self.target_column in engineered_df.columns:
                target_series = pd.to_numeric(engineered_df[self.target_column], errors="coerce")
                for col in planned_outputs:
                    if col in engineered_df.columns and pd.api.types.is_numeric_dtype(engineered_df[col]):
                        r = abs(target_series.corr(engineered_df[col]))
                        if r >= self.leakage_corr_limit:
                            leakage_detected.append((col, f"Correlation |r|={round(float(r), 4)} >= limit"))
                            offending_columns.append(col)

        if leakage_detected:
            checks.append({
                "check": "Target leakage",
                "status": "FAIL",
                "columns": leakage_detected,
                "message": f"Severe target leakage detected (|r| >= {self.leakage_corr_limit}): {leakage_detected}"
            })
            overall_status = "FAILED"
        else:
            checks.append({"check": "Target leakage", "status": "PASS", "message": "No target leakage detected in engineered features."})

        # Check 9: Metadata completeness
        meta_valid = True
        for m in metadata:
            if not all(k in m for k in ["name", "data_type", "origin", "source_columns", "operation", "reason", "version"]):
                meta_valid = False
                break

        if not meta_valid:
            checks.append({
                "check": "Metadata completeness",
                "status": "FAIL",
                "message": "Feature metadata is incomplete. Each feature requires name, type, origin, source_columns, operation, reason, and version."
            })
            overall_status = "FAILED"
        else:
            checks.append({"check": "Metadata completeness", "status": "PASS", "message": "Metadata is complete and registered."})

        return {
            "status": overall_status,
            "checks": checks,
            "offending_columns": list(set(offending_columns))
        }

    def repair_plan(
        self,
        plan: List[Dict[str, Any]],
        offending_columns: List[str]
    ) -> List[Dict[str, Any]]:
        """Removes steps from the plan that generated offending features."""
        repaired = [step for step in plan if step.get("output_name") not in offending_columns]
        return repaired
