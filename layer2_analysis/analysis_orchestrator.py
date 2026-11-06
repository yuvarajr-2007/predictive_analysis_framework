"""
Analysis Layer Orchestrator:
Coordinates EDA Agent, Feature Engineering Agent, and Feature Verification Gate with repair loop.
"""
import time
from typing import Dict, Any, List, Optional
import pandas as pd

from .eda_agent import EDAAgent
from .feature_engineering_agent import FeatureEngineeringAgent
from .verification_gate import FeatureVerificationGate


class AnalysisOrchestrator:
    def __init__(
        self,
        domain_dir: Optional[str] = None,
        target_column: str = "anomaly",
        time_column: str = "timestamp",
        max_new_features: int = 10
    ):
        self.target_column = target_column
        self.time_column = time_column
        self.eda_agent = EDAAgent(target_column=target_column, time_column=time_column)
        self.fe_agent = FeatureEngineeringAgent(domain_dir=domain_dir, max_new_features=max_new_features, target_column=target_column)
        self.gate = FeatureVerificationGate(target_column=target_column)

    def run(
        self,
        validated_df: pd.DataFrame,
        domain_name: str = "high_voltage_power",
        validation_package: Optional[Dict[str, Any]] = None,
        user_overrides: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        start_time = time.time()

        # Step 1: Pre-condition check from Validation Layer
        if validation_package and validation_package.get("overall_status") == "FAIL":
            return {
                "status": "STOPPED",
                "error": "Upstream Validation reported FAILED. Analysis layer execution halted to prevent dirty pipeline state.",
                "eda_report": None,
                "engineered_dataframe": None,
                "feature_metadata": None,
                "feature_plan": None,
                "gate_result": None
            }

        # Step 2: Run EDA Agent in parallel
        eda_report = self.eda_agent.run_parallel(validated_df)

        # Step 3: Build initial feature plan
        plan = self.fe_agent.build_feature_plan(
            df=validated_df,
            domain_name=domain_name,
            eda_report=eda_report,
            user_overrides=user_overrides
        )

        # Step 4 & 5: Execute plan and run Verification Gate with repair loop (up to 3 attempts)
        max_attempts = 3
        current_attempt = 1
        current_plan = list(plan)
        engineered_df = validated_df.copy()
        metadata: List[Dict[str, Any]] = []
        gate_result: Dict[str, Any] = {}

        while current_attempt <= max_attempts:
            engineered_df, metadata = self.fe_agent.execute_plan(validated_df, current_plan)
            gate_result = self.gate.verify(
                original_df=validated_df,
                engineered_df=engineered_df,
                plan=current_plan,
                metadata=metadata
            )

            if gate_result["status"] in ["PASSED", "PASSED WITH WARNINGS"]:
                break

            # If FAILED, attempt repair
            offending = gate_result.get("offending_columns", [])
            if not offending:
                break

            current_plan = self.gate.repair_plan(current_plan, offending)
            current_attempt += 1

        duration = round(time.time() - start_time, 4)

        return {
            "status": gate_result.get("status", "FAILED"),
            "attempts": current_attempt,
            "duration_seconds": duration,
            "eda_report": eda_report,
            "engineered_dataframe": engineered_df,
            "feature_metadata": metadata,
            "feature_plan": current_plan,
            "gate_result": gate_result
        }
