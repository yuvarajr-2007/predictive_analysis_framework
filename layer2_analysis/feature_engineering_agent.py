"""
Feature Engineering Agent:
Merges 4 priority sources (User overrides > Domain templates > LLM/EDA suggestions > Automatic defaults),
validates and plans transformations, executes via DSLExecutor, and builds feature metadata.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import yaml
import pandas as pd

from .dsl_executor import DSLExecutor


class FeatureEngineeringAgent:
    def __init__(self, domain_dir: Optional[str] = None, max_new_features: int = 10, target_column: str = "anomaly"):
        if domain_dir:
            self.domain_dir = Path(domain_dir)
        else:
            self.domain_dir = Path(__file__).resolve().parent.parent / "domain_knowledge"
        self.max_new_features = max_new_features
        self.target_column = target_column
        self.executor = DSLExecutor()

    def load_domain_templates(self, domain_name: str) -> Dict[str, Any]:
        """Loads domain feature templates (interactions, bins, groups)."""
        template_file = self.domain_dir / "feature_templates" / f"{domain_name}.yaml"
        if template_file.exists():
            with open(template_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def build_feature_plan(
        self,
        df: pd.DataFrame,
        domain_name: str,
        eda_report: Dict[str, Any],
        user_overrides: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Synthesizes the plan following strict 4-tier priority order:
        1. User overrides
        2. Domain feature templates
        3. LLM/EDA recommendations
        4. Automatic defaults
        """
        plan: List[Dict[str, Any]] = []
        planned_outputs = set(df.columns)

        # 1. User overrides (Priority 1)
        if user_overrides:
            for u in user_overrides:
                out_col = u.get("output_name")
                if out_col not in planned_outputs:
                    u_step = dict(u)
                    u_step["origin"] = "user_override"
                    plan.append(u_step)
                    planned_outputs.add(out_col)

        # 2. Domain Feature Templates (Priority 2)
        templates = self.load_domain_templates(domain_name)
        for inter in templates.get("interactions", []):
            out_col = inter.get("name")
            inputs = inter.get("inputs", [])
            # Skip if columns not in dataframe or already planned
            if out_col not in planned_outputs and all(col in df.columns for col in inputs):
                plan.append({
                    "output_name": out_col,
                    "operation": inter.get("operation"),
                    "inputs": inputs,
                    "reason": inter.get("reason", "Domain knowledge physical interaction"),
                    "origin": "domain_template",
                    "params": {}
                })
                planned_outputs.add(out_col)

        for b_spec in templates.get("bins", []):
            out_col = b_spec.get("output_name")
            col = b_spec.get("column")
            if out_col not in planned_outputs and col in df.columns:
                plan.append({
                    "output_name": out_col,
                    "operation": "bin",
                    "inputs": [col],
                    "reason": "Domain defined threshold binning",
                    "origin": "domain_template",
                    "params": {
                        "bins": b_spec.get("bins"),
                        "labels": b_spec.get("labels")
                    }
                })
                planned_outputs.add(out_col)

        # 3. LLM / EDA suggestions (Priority 3)
        eda_recs = eda_report.get("insights", {}).get("recommended_actions", [])
        for rec in eda_recs:
            if len(plan) >= self.max_new_features:
                break
            op = rec.get("operation")
            if op == "log1p":
                col = rec.get("column")
                if col == self.target_column:
                    continue
                out_col = f"{col}_log1p"
                if out_col not in planned_outputs and col in df.columns:
                    plan.append({
                        "output_name": out_col,
                        "operation": "log1p",
                        "inputs": [col],
                        "reason": rec.get("reason", "Normalize skewed distribution"),
                        "origin": "eda_suggestion",
                        "params": {}
                    })
                    planned_outputs.add(out_col)
            elif op == "ratio":
                inputs = rec.get("inputs", [])
                if self.target_column in inputs:
                    continue  # Strict prevention of target leakage
                if len(inputs) == 2 and all(c in df.columns for c in inputs):
                    out_col = f"{inputs[0]}_per_{inputs[1]}"
                    if out_col not in planned_outputs:
                        plan.append({
                            "output_name": out_col,
                            "operation": "ratio",
                            "inputs": inputs,
                            "reason": rec.get("reason", "Correlated feature interaction ratio"),
                            "origin": "eda_suggestion",
                            "params": {"epsilon": 1e-6}
                        })
                        planned_outputs.add(out_col)
            elif op == "datetime_part":
                col = rec.get("column")
                part = rec.get("part", "hour")
                out_col = f"{col}_{part}"
                if out_col not in planned_outputs and col in df.columns:
                    plan.append({
                        "output_name": out_col,
                        "operation": "datetime_part",
                        "inputs": [col],
                        "reason": rec.get("reason", "Extract cyclic temporal component"),
                        "origin": "eda_suggestion",
                        "params": {"part": part}
                    })
                    planned_outputs.add(out_col)

        # 4. Automatic defaults (Priority 4)
        if len(plan) < self.max_new_features:
            # Check for datetime column to extract hour if not done
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]) or "time" in c.lower():
                    out_col = f"{c}_hour"
                    if out_col not in planned_outputs:
                        plan.append({
                            "output_name": out_col,
                            "operation": "datetime_part",
                            "inputs": [c],
                            "reason": "Automatic diurnal feature extraction",
                            "origin": "automatic_default",
                            "params": {"part": "hour"}
                        })
                        planned_outputs.add(out_col)
                        break

        # Respect overall cap
        plan = plan[:self.max_new_features]
        return plan

    def execute_plan(self, df: pd.DataFrame, plan: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Executes the safe plan step-by-step and collects feature metadata."""
        current_df = df.copy()
        metadata_list = []

        for step in plan:
            try:
                current_df, meta = self.executor.execute_step(current_df, step)
                metadata_list.append(meta)
            except Exception as e:
                # Log step error and continue with remaining steps
                metadata_list.append({
                    "name": step.get("output_name", "unknown"),
                    "data_type": "error",
                    "origin": step.get("origin", "unknown"),
                    "source_columns": step.get("inputs", []),
                    "operation": step.get("operation"),
                    "reason": f"Execution failed: {str(e)}",
                    "version": 1,
                    "error": str(e)
                })

        return current_df, metadata_list
