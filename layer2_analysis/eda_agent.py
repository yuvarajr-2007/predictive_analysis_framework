"""
EDA Agent: Concurrent profiling of validated dataframes, statistical analysis,
and insight generation with rule-based and LLM fallback.
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class EDAAgent:
    def __init__(self, target_column: Optional[str] = "anomaly", time_column: Optional[str] = "timestamp"):
        self.target_column = target_column
        self.time_column = time_column

    def compute_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Dataset overview: rows, columns, memory, dtypes, column roles."""
        roles = {}
        for col in df.columns:
            if col == self.target_column:
                roles[col] = "target"
            elif col == self.time_column or pd.api.types.is_datetime64_any_dtype(df[col]):
                roles[col] = "datetime"
            elif pd.api.types.is_numeric_dtype(df[col]):
                roles[col] = "numeric"
            elif pd.api.types.is_bool_dtype(df[col]):
                roles[col] = "boolean"
            else:
                # Check if it could be an identifier
                if df[col].nunique() / max(1, len(df)) > 0.9:
                    roles[col] = "identifier"
                else:
                    roles[col] = "categorical"

        return {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
            "duplicate_rows": int(df.duplicated().sum()),
            "column_roles": roles,
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }

    def compute_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Missing count & %, constant columns, zero-variance, high-cardinality."""
        quality = {}
        constant_cols = []
        zero_var_cols = []
        high_card_cols = []

        total_rows = len(df)
        for col in df.columns:
            null_count = int(df[col].isna().sum())
            null_pct = round(null_count / total_rows * 100, 2)
            n_unique = int(df[col].nunique(dropna=True))

            quality[col] = {
                "missing_count": null_count,
                "missing_percent": null_pct,
                "unique_values": n_unique
            }

            if n_unique <= 1:
                constant_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                if float(df[col].std()) == 0.0:
                    zero_var_cols.append(col)

            if n_unique > 500 and not pd.api.types.is_numeric_dtype(df[col]):
                high_card_cols.append(col)

        return {
            "per_column": quality,
            "constant_columns": constant_cols,
            "zero_variance_columns": zero_var_cols,
            "high_cardinality_columns": high_card_cols
        }

    def compute_numerical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Numeric summary: mean, median, std, min, max, skewness, kurtosis, quantiles."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        stats = {}
        for col in num_cols:
            if col == self.target_column:
                continue
            s = df[col].dropna()
            if len(s) == 0:
                continue
            skew = float(s.skew()) if len(s) > 2 else 0.0
            kurt = float(s.kurt()) if len(s) > 3 else 0.0
            q25, q50, q75 = s.quantile([0.25, 0.50, 0.75]).tolist()

            stats[col] = {
                "mean": round(float(s.mean()), 3),
                "std": round(float(s.std()), 3),
                "median": round(float(q50), 3),
                "min": round(float(s.min()), 3),
                "max": round(float(s.max()), 3),
                "q25": round(float(q25), 3),
                "q75": round(float(q75), 3),
                "skewness": round(skew, 3),
                "kurtosis": round(kurt, 3),
                "highly_skewed": abs(skew) > 1.5
            }
        return stats

    def compute_categorical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Categorical summary: unique counts and top-5 frequencies."""
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
        res = {}
        for col in cat_cols:
            if col == self.time_column:
                continue
            s = df[col].dropna()
            top_vals = s.value_counts().head(5).to_dict()
            res[col] = {
                "unique_count": int(s.nunique()),
                "top_categories": {str(k): int(v) for k, v in top_vals.items()}
            }
        return res

    def compute_target_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect target type, class distribution, and imbalance ratio."""
        if not self.target_column or self.target_column not in df.columns:
            return {"detected": False, "message": f"Target column '{self.target_column}' not found."}

        target_series = df[self.target_column].dropna()
        n_unique = target_series.nunique()

        if n_unique == 2:
            prob_type = "binary_classification"
            counts = target_series.value_counts().to_dict()
            counts_str = {str(k): int(v) for k, v in counts.items()}
            min_class_count = min(counts.values())
            imbalance_ratio = round(min_class_count / len(target_series), 4)
            is_imbalanced = imbalance_ratio < 0.15
        elif n_unique < 15 and pd.api.types.is_integer_dtype(target_series):
            prob_type = "multiclass_classification"
            counts = target_series.value_counts().to_dict()
            counts_str = {str(k): int(v) for k, v in counts.items()}
            imbalance_ratio = round(min(counts.values()) / len(target_series), 4)
            is_imbalanced = imbalance_ratio < 0.05
        else:
            prob_type = "regression"
            counts_str = {}
            imbalance_ratio = 1.0
            is_imbalanced = False

        return {
            "detected": True,
            "target_column": self.target_column,
            "problem_type": prob_type,
            "class_distribution": counts_str,
            "imbalance_ratio": imbalance_ratio,
            "is_imbalanced": is_imbalanced
        }

    def compute_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Pearson correlation and highly correlated pairs (|r| > 0.85)."""
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            return {"corr_matrix": {}, "high_correlations": []}

        corr = num_df.corr().round(3)
        high_corr_pairs = []

        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col1, col2 = cols[i], cols[j]
                val = corr.loc[col1, col2]
                if abs(val) >= 0.85 and not np.isnan(val):
                    high_corr_pairs.append({
                        "col1": col1,
                        "col2": col2,
                        "r": float(val),
                        "interpretation": "Strong multi-collinearity or physical coupling"
                    })

        return {
            "corr_matrix": corr.to_dict(),
            "high_correlations": high_corr_pairs
        }

    def compute_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """IQR-based outlier count per numeric column."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        outlier_res = {}
        for col in num_cols:
            if col == self.target_column:
                continue
            s = df[col].dropna()
            if len(s) < 10:
                continue
            q25 = s.quantile(0.25)
            q75 = s.quantile(0.75)
            iqr = q75 - q25
            if iqr > 0:
                lower = q25 - 1.5 * iqr
                upper = q75 + 1.5 * iqr
                count = int(((s < lower) | (s > upper)).sum())
                pct = round(count / len(s) * 100, 2)
                outlier_res[col] = {
                    "outlier_count": count,
                    "outlier_percent": pct,
                    "lower_bound": round(lower, 2),
                    "upper_bound": round(upper, 2)
                }
        return outlier_res

    def compute_time_series_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check monotonic order, time step frequency, and gaps."""
        time_col = None
        if self.time_column and self.time_column in df.columns:
            time_col = self.time_column
        else:
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]) or "time" in c.lower() or "date" in c.lower():
                    time_col = c
                    break

        if not time_col:
            return {"is_time_series": False, "message": "No temporal/datetime column identified."}

        ts = pd.to_datetime(df[time_col], errors="coerce")
        is_monotonic = bool(ts.is_monotonic_increasing)
        gaps_detected = False
        if len(ts.dropna()) > 2 and is_monotonic:
            diffs = ts.diff().dropna()
            median_diff = diffs.median()
            gaps_detected = bool((diffs > 3 * median_diff).any())

        return {
            "is_time_series": True,
            "time_column": time_col,
            "monotonic_increasing": is_monotonic,
            "gaps_detected": gaps_detected,
            "start_time": str(ts.min()),
            "end_time": str(ts.max())
        }

    def generate_insights(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic insight generator with agentic suggestions."""
        target_info = profile.get("target_analysis", {})
        prob_type = target_info.get("problem_type", "anomaly_detection")
        is_imbalanced = target_info.get("is_imbalanced", False)
        imbalance_ratio = target_info.get("imbalance_ratio", 1.0)

        findings = []
        risks = []
        recommended_actions = []

        # Target finding
        if target_info.get("detected"):
            findings.append(f"Target column detected as {prob_type}. Imbalance ratio: {imbalance_ratio * 100:.1f}%.")
            if is_imbalanced:
                risks.append(f"Severe class imbalance ({imbalance_ratio*100:.1f}% positives). Standard accuracy will mislead; PR-AUC and Macro-F1 required.")
                recommended_actions.append({"action": "cv_strategy", "strategy": "StratifiedKFold", "reason": "Preserve anomaly ratio across CV splits."})

        # Skewness recommendations
        numerics = profile.get("numerical_analysis", {})
        for col, st in numerics.items():
            if st.get("highly_skewed"):
                findings.append(f"Feature '{col}' exhibits significant positive/negative skewness ({st.get('skewness')}).")
                recommended_actions.append({
                    "action": "transform",
                    "operation": "log1p",
                    "column": col,
                    "reason": f"Normalize heavy tail in {col} (skewness={st.get('skewness')})"
                })

        # High correlation
        high_corr = profile.get("correlations", {}).get("high_correlations", [])
        for hc in high_corr:
            col1, col2 = hc["col1"], hc["col2"]
            findings.append(f"Strong collinearity between '{col1}' and '{col2}' (r={hc['r']}).")
            recommended_actions.append({
                "action": "ratio",
                "operation": "ratio",
                "inputs": [col1, col2],
                "reason": f"Capture physical coupling ratio between {col1} and {col2}"
            })

        # Time series finding
        ts_info = profile.get("time_series_checks", {})
        if ts_info.get("is_time_series"):
            findings.append(f"Chronological dataset detected on '{ts_info.get('time_column')}'.")
            recommended_actions.append({"action": "cv_strategy", "strategy": "TimeSeriesSplit", "reason": "Strictly prevent future lookahead leakage."})
            recommended_actions.append({"action": "feature_engineering", "operation": "datetime_part", "column": ts_info.get("time_column"), "part": "hour", "reason": "Extract diurnal/hourly operational cycles."})

        return {
            "problem_type": prob_type,
            "key_findings": findings,
            "risks": risks,
            "recommended_actions": recommended_actions
        }

    def run_parallel(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Runs the independent profiling sections concurrently in a ThreadPool."""
        with ThreadPoolExecutor(max_workers=6) as executor:
            fut_overview = executor.submit(self.compute_overview, df)
            fut_quality = executor.submit(self.compute_data_quality, df)
            fut_numeric = executor.submit(self.compute_numerical_analysis, df)
            fut_categorical = executor.submit(self.compute_categorical_analysis, df)
            fut_target = executor.submit(self.compute_target_analysis, df)
            fut_corr = executor.submit(self.compute_correlations, df)
            fut_outliers = executor.submit(self.compute_outliers, df)
            fut_ts = executor.submit(self.compute_time_series_checks, df)

            profile = {
                "overview": fut_overview.result(),
                "data_quality": fut_quality.result(),
                "numerical_analysis": fut_numeric.result(),
                "categorical_analysis": fut_categorical.result(),
                "target_analysis": fut_target.result(),
                "correlations": fut_corr.result(),
                "outliers": fut_outliers.result(),
                "time_series_checks": fut_ts.result(),
            }

        # Generate insights
        profile["insights"] = self.generate_insights(profile)
        return profile
