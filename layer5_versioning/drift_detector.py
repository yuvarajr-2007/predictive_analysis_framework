"""
Drift Detection (ML-Divergence Monitor):
Monitors incoming/inference datasets against training baselines.
Computes non-parametric Kolmogorov-Smirnov (KS-test), Population Stability Index (PSI),
and Chi-Square categorical divergence.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency


class DriftDetector:
    def __init__(self, p_value_threshold: float = 0.05, psi_threshold: float = 0.20):
        self.p_value_threshold = p_value_threshold
        self.psi_threshold = psi_threshold

    @staticmethod
    def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
        """
        Calculates Population Stability Index (PSI) between reference and current samples.
        """
        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        # Create quantile break points from reference distribution
        percentiles = np.linspace(0, 100, num_buckets + 1)
        breakpoints = np.percentile(expected, percentiles)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        # Calculate counts in buckets
        expected_counts, _ = np.histogram(expected, bins=breakpoints)
        actual_counts, _ = np.histogram(actual, bins=breakpoints)

        # Convert to relative proportions with smoothing epsilon to prevent division by zero
        eps = 1e-4
        expected_pct = (expected_counts + eps) / (len(expected) + eps * num_buckets)
        actual_pct = (actual_counts + eps) / (len(actual) + eps * num_buckets)

        # Compute PSI summation
        psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(np.round(psi_value, 4))

    def detect_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates drift across all shared features between reference and current datasets.
        """
        if feature_columns is None:
            feature_columns = [c for c in reference_df.columns if c in current_df.columns]

        per_feature_drift = {}
        drifted_features = []

        for col in feature_columns:
            ref_col = reference_df[col].dropna()
            cur_col = current_df[col].dropna()

            if len(ref_col) < 5 or len(cur_col) < 5:
                continue

            # Numerical feature drift analysis
            if pd.api.types.is_numeric_dtype(ref_col) and pd.api.types.is_numeric_dtype(cur_col):
                ks_stat, ks_pval = ks_2samp(ref_col.values, cur_col.values)
                psi_val = self.calculate_psi(ref_col.values, cur_col.values)

                is_drifted = bool(ks_pval < self.p_value_threshold or psi_val >= self.psi_threshold)
                if is_drifted:
                    drifted_features.append(col)

                severity = "HIGH" if psi_val >= self.psi_threshold else ("MODERATE" if ks_pval < self.p_value_threshold else "NONE")

                per_feature_drift[col] = {
                    "type": "numeric",
                    "drift_detected": is_drifted,
                    "severity": severity,
                    "ks_statistic": round(float(ks_stat), 4),
                    "ks_p_value": round(float(ks_pval), 5),
                    "psi": round(float(psi_val), 4),
                    "ref_mean": round(float(ref_col.mean()), 3),
                    "cur_mean": round(float(cur_col.mean()), 3)
                }

            # Categorical feature drift analysis
            else:
                top_cats = set(ref_col.value_counts().head(10).index).union(cur_col.value_counts().head(10).index)
                ref_freq = [ref_col.isin([c]).sum() for c in top_cats]
                cur_freq = [cur_col.isin([c]).sum() for c in top_cats]

                try:
                    chi2, p_val, _, _ = chi2_contingency([ref_freq, cur_freq])
                    is_drifted = bool(p_val < self.p_value_threshold)
                    if is_drifted:
                        drifted_features.append(col)
                    per_feature_drift[col] = {
                        "type": "categorical",
                        "drift_detected": is_drifted,
                        "severity": "HIGH" if is_drifted else "NONE",
                        "chi2_stat": round(float(chi2), 4),
                        "p_value": round(float(p_val), 5)
                    }
                except Exception:
                    pass

        total_features = len(per_feature_drift)
        drift_ratio = len(drifted_features) / max(1, total_features)
        dataset_drift_alert = drift_ratio > 0.30

        return {
            "dataset_drift_alert": dataset_drift_alert,
            "drift_share_percent": round(drift_ratio * 100, 2),
            "drifted_features_count": len(drifted_features),
            "total_features_evaluated": total_features,
            "drifted_features": drifted_features,
            "per_feature_details": per_feature_drift
        }
