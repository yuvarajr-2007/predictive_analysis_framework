"""
Adaptive Baseline (Naive Predictor - Pathway A):
Fast mathematical anomaly boundaries using rolling Z-scores, Interquartile Range (IQR),
rolling medians, and Isolation Forest. Explainable and ultra low-latency.
"""
import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AdaptiveBaselinePredictor:
    def __init__(self, window_size: int = 20, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold

    def fit_predict(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = "anomaly"
    ) -> Dict[str, Any]:
        start_time = time.time()

        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in feature_cols:
                feature_cols.remove(target_col)

        sub_df = df[feature_cols].copy().fillna(df[feature_cols].median())

        # 1. Rolling Z-scores & Median Absolute Deviation (MAD)
        z_scores = pd.DataFrame(index=df.index)
        for col in feature_cols:
            series = sub_df[col]
            rolling_mean = series.rolling(window=self.window_size, min_periods=1).mean()
            rolling_std = series.rolling(window=self.window_size, min_periods=1).std().replace(0, 1e-6)
            z_scores[col] = (series - rolling_mean).abs() / rolling_std

        max_z = z_scores.max(axis=1).fillna(0)
        # Normalize anomaly scores to 0-1
        norm_scores = (max_z / (max_z.max() + 1e-6)).clip(0.0, 1.0)

        # 2. Isolation Forest boundary confirmation
        iso = IsolationForest(contamination=0.05, random_state=42)
        iso.fit(sub_df)
        iso_scores = -iso.decision_function(sub_df)
        iso_scores = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-6)

        # Blended anomaly score
        blended_scores = 0.6 * norm_scores + 0.4 * iso_scores
        threshold = float(np.percentile(blended_scores, 95))
        flags = (blended_scores >= threshold).astype(int)

        # Calculate metrics if ground truth target exists
        metrics = {}
        if target_col and target_col in df.columns:
            y_true = df[target_col].values
            from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, precision_score, recall_score
            try:
                metrics["ROC-AUC"] = round(float(roc_auc_score(y_true, blended_scores)), 4)
                p, r, _ = precision_recall_curve(y_true, blended_scores)
                metrics["PR-AUC"] = round(float(auc(r, p)), 4)
                metrics["F1-Score"] = round(float(f1_score(y_true, flags, zero_division=0)), 4)
                metrics["Precision"] = round(float(precision_score(y_true, flags, zero_division=0)), 4)
                metrics["Recall"] = round(float(recall_score(y_true, flags, zero_division=0)), 4)
            except Exception:
                pass

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "model_type": "Adaptive_Baseline_NaivePredictor",
            "anomaly_scores": blended_scores.round(4).tolist(),
            "flags": flags.tolist(),
            "threshold": round(threshold, 4),
            "metrics": metrics,
            "feature_importance": {col: round(float(z_scores[col].mean()), 4) for col in feature_cols},
            "latency_ms": latency_ms,
            "boundary_type": "Rolling_ZScore_Plus_IsolationForest"
        }
