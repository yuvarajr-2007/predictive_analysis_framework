"""
DSL Executor: Safe execution of whitelisted feature engineering operations.
Zero arbitrary code execution. Guarantees deterministic, reproducible transformations.
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


class DSLExecutor:
    ALLOWED_OPERATIONS = {
        "log1p", "sqrt", "square", "ratio", "multiply",
        "add", "subtract", "bin", "group_agg", "datetime_part", "clip", "drop"
    }

    def execute_step(self, df: pd.DataFrame, step: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes a single planned feature transformation.
        Returns:
            (updated_df, metadata_entry)
        """
        op = step.get("operation")
        if op not in self.ALLOWED_OPERATIONS:
            raise ValueError(f"Disallowed or unknown operation: '{op}'")

        out_name = step.get("output_name")
        inputs = step.get("inputs", [])
        params = step.get("params", {})
        reason = step.get("reason", "Engineered transformation")
        origin = step.get("origin", "automatic")

        # Working copy to preserve immutability if needed
        res_df = df.copy()

        # Handle 'drop'
        if op == "drop":
            cols_to_drop = [c for c in inputs if c in res_df.columns]
            if cols_to_drop:
                res_df.drop(columns=cols_to_drop, inplace=True)
            meta = {
                "name": ",".join(cols_to_drop),
                "data_type": "None",
                "origin": origin,
                "source_columns": inputs,
                "operation": "drop",
                "reason": reason,
                "version": 1
            }
            return res_df, meta

        # Check required input columns
        for inp in inputs:
            if inp not in res_df.columns:
                raise KeyError(f"Input column '{inp}' missing from dataframe.")

        new_series = None

        if op == "log1p":
            col = inputs[0]
            val = pd.to_numeric(res_df[col], errors="coerce").fillna(0)
            new_series = np.log1p(np.maximum(0, val))

        elif op == "sqrt":
            col = inputs[0]
            val = pd.to_numeric(res_df[col], errors="coerce").fillna(0)
            new_series = np.sqrt(np.maximum(0, val))

        elif op == "square":
            col = inputs[0]
            val = pd.to_numeric(res_df[col], errors="coerce").fillna(0)
            new_series = val ** 2

        elif op == "ratio":
            num_col, den_col = inputs[0], inputs[1]
            num = pd.to_numeric(res_df[num_col], errors="coerce").fillna(0)
            den = pd.to_numeric(res_df[den_col], errors="coerce").fillna(0)
            # Guard against division by zero with epsilon
            eps = params.get("epsilon", 1e-6)
            new_series = num / (den.abs() + eps)

        elif op == "multiply":
            col1, col2 = inputs[0], inputs[1]
            c1 = pd.to_numeric(res_df[col1], errors="coerce").fillna(0)
            c2 = pd.to_numeric(res_df[col2], errors="coerce").fillna(0)
            new_series = c1 * c2

        elif op == "add":
            col1, col2 = inputs[0], inputs[1]
            c1 = pd.to_numeric(res_df[col1], errors="coerce").fillna(0)
            c2 = pd.to_numeric(res_df[col2], errors="coerce").fillna(0)
            new_series = c1 + c2

        elif op == "subtract":
            col1, col2 = inputs[0], inputs[1]
            c1 = pd.to_numeric(res_df[col1], errors="coerce").fillna(0)
            c2 = pd.to_numeric(res_df[col2], errors="coerce").fillna(0)
            new_series = c1 - c2

        elif op == "bin":
            col = inputs[0]
            s = pd.to_numeric(res_df[col], errors="coerce")
            bins = params.get("bins")
            labels = params.get("labels", None)
            if bins:
                new_series = pd.cut(s, bins=bins, labels=labels, include_lowest=True).astype(str)
            else:
                q = params.get("quantiles", 5)
                new_series = pd.qcut(s, q=q, labels=labels, duplicates="drop").astype(str)

        elif op == "group_agg":
            val_col, grp_col = inputs[0], inputs[1]
            agg_type = params.get("agg", "mean")
            grouped = res_df.groupby(grp_col)[val_col].transform(agg_type)
            new_series = pd.to_numeric(grouped, errors="coerce")

        elif op == "datetime_part":
            col = inputs[0]
            part = params.get("part", "hour")
            ts = pd.to_datetime(res_df[col], errors="coerce")
            if part == "hour":
                new_series = ts.dt.hour
            elif part == "day":
                new_series = ts.dt.day
            elif part == "dayofweek":
                new_series = ts.dt.dayofweek
            elif part == "month":
                new_series = ts.dt.month
            else:
                new_series = ts.dt.hour

        elif op == "clip":
            col = inputs[0]
            val = pd.to_numeric(res_df[col], errors="coerce")
            lower = params.get("lower", val.min())
            upper = params.get("upper", val.max())
            new_series = val.clip(lower=lower, upper=upper)

        if new_series is not None:
            # Round numerical features cleanly
            if pd.api.types.is_float_dtype(new_series):
                new_series = new_series.round(4)
            res_df[out_name] = new_series

        metadata = {
            "name": out_name,
            "data_type": str(res_df[out_name].dtype) if out_name in res_df else "None",
            "origin": origin,
            "source_columns": inputs,
            "operation": op,
            "reason": reason,
            "version": 1
        }
        return res_df, metadata
