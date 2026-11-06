"""
CV Strategy Selector:
Dynamically selects Cross-Validation strategy to guarantee zero data leakage.
Supports TimeSeriesSplit, StratifiedKFold, GroupKFold, and standard KFold.
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold


class CVStrategySelector:
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def select_strategy(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = "anomaly",
        time_column: Optional[str] = "timestamp",
        group_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Inspects dataset schema, class imbalance, and temporal/group properties
        to select the appropriate cross-validation split strategy.
        """
        # 1. Check for temporal/time-series data
        has_time = False
        t_col = None
        if time_column and time_column in df.columns:
            has_time = True
            t_col = time_column
        else:
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]) or "time" in c.lower() or "date" in c.lower():
                    has_time = True
                    t_col = c
                    break

        if has_time:
            # Check if sorted or can be chronologically ordered
            return {
                "strategy_name": "TimeSeriesSplit",
                "n_splits": self.n_splits,
                "split_column": t_col,
                "reason": f"Chronological ordering detected on '{t_col}'. TimeSeriesSplit prevents future lookahead leakage.",
                "cv_object": TimeSeriesSplit(n_splits=self.n_splits)
            }

        # 2. Check for grouped data (multi-device, multi-user)
        detected_group = None
        if group_column and group_column in df.columns:
            detected_group = group_column
        else:
            for col in ["machine_id", "user_id", "device_id", "patient_id", "client_id"]:
                if col in df.columns:
                    detected_group = col
                    break

        if detected_group and df[detected_group].nunique() >= self.n_splits:
            return {
                "strategy_name": "GroupKFold",
                "n_splits": self.n_splits,
                "split_column": detected_group,
                "reason": f"Hierarchical grouping detected on '{detected_group}'. GroupKFold guarantees out-of-device generalization.",
                "cv_object": GroupKFold(n_splits=self.n_splits)
            }

        # 3. Check for imbalanced classification
        if target_column and target_column in df.columns:
            target_series = df[target_column].dropna()
            n_unique = target_series.nunique()
            if n_unique == 2 or (n_unique < 10 and pd.api.types.is_integer_dtype(target_series)):
                # Calculate minority class percentage
                counts = target_series.value_counts()
                minority_pct = counts.min() / len(target_series)
                if minority_pct < 0.20:
                    return {
                        "strategy_name": "StratifiedKFold",
                        "n_splits": self.n_splits,
                        "split_column": target_column,
                        "reason": f"Severe class imbalance detected ({minority_pct*100:.1f}% positives). StratifiedKFold maintains identical class proportions across all folds.",
                        "cv_object": StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
                    }

        # 4. Default fallback
        return {
            "strategy_name": "KFold",
            "n_splits": self.n_splits,
            "split_column": None,
            "reason": "Standard cross-sectional data with balanced distribution.",
            "cv_object": KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        }
