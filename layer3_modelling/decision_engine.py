"""
Decision Engine (Block 3: Intelligent Routing Gate):
Evaluates dataset volume, velocity, variety, and domain constraints
to decide the algorithmic pathway:
  - Branch YES -> Adaptive Baseline (Naive Predictor: rolling Z-scores, IQR, medians)
  - Branch NO  -> AutoGluon Ensemble (Multi-model stacking: LightGBM, XGBoost, CatBoost, RF)
"""
from typing import Dict, Any
import numpy as np
import pandas as pd


class DecisionEngine:
    def evaluate_routing(
        self,
        df: pd.DataFrame,
        eda_report: Dict[str, Any],
        is_streaming_stream: bool = False,
        low_latency_override: bool = False
    ) -> Dict[str, Any]:
        """
        Determines the execution path based on:
        1. Volume: len(df)
        2. Velocity: streaming or continuous sensor monitoring
        3. Variety: categorical, numeric, text combinations
        4. Multi-collinearity & complexity
        """
        rows = len(df)
        cols = len(df.columns)
        has_text_or_cat = len(df.select_dtypes(include=["object", "category"]).columns) > 1
        high_corr_count = len(eda_report.get("correlations", {}).get("high_correlations", []))
        is_time_series = eda_report.get("time_series_checks", {}).get("is_time_series", False)

        reasons = []
        route_to_baseline = False

        if low_latency_override or is_streaming_stream:
            route_to_baseline = True
            reasons.append("Low-latency / streaming operational mode requested.")

        elif rows < 500 and not has_text_or_cat and is_time_series and high_corr_count == 0:
            route_to_baseline = True
            reasons.append("Data is highly uniform, continuous temporal stream with low complexity.")

        else:
            route_to_baseline = False
            if rows >= 500:
                reasons.append(f"Substantial dataset volume ({rows} rows).")
            if has_text_or_cat:
                reasons.append("Mixed data types (categoricals/text and numerics present).")
            if high_corr_count > 0:
                reasons.append(f"Non-linear multi-collinearity present ({high_corr_count} strong pairs).")
            reasons.append("Complex interactions require gradient boosted meta-ensembling.")

        pathway = "Adaptive_Baseline" if route_to_baseline else "AutoGluon_Ensemble"

        return {
            "pathway": pathway,
            "decision": "YES (Simple/Streaming Baseline)" if route_to_baseline else "NO (Ensemble Powerhouse)",
            "reasons": reasons,
            "metrics_to_optimize": ["PR-AUC", "F1-Score"] if eda_report.get("target_analysis", {}).get("is_imbalanced") else ["ROC-AUC", "F1-Score"]
        }
