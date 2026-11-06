import pytest
import pandas as pd
from layer2_analysis.eda_agent import EDAAgent
from layer2_analysis.dsl_executor import DSLExecutor
from layer2_analysis.feature_engineering_agent import FeatureEngineeringAgent
from layer2_analysis.verification_gate import FeatureVerificationGate
from layer2_analysis.analysis_orchestrator import AnalysisOrchestrator


def test_eda_agent_overview_and_insights():
    df = pd.read_csv("d:/predictive_framework/data/sample_high_voltage_power.csv")
    agent = EDAAgent(target_column="anomaly")
    profile = agent.run_parallel(df)
    assert profile["overview"]["rows"] == len(df)
    assert "insights" in profile
    assert profile["target_analysis"]["detected"] is True


def test_dsl_executor_whitelisted_operations():
    executor = DSLExecutor()
    df = pd.DataFrame({"v": [10.0, 20.0], "i": [2.0, 5.0]})

    # Multiply
    res_df, meta = executor.execute_step(df, {
        "operation": "multiply",
        "inputs": ["v", "i"],
        "output_name": "p",
        "reason": "power"
    })
    assert "p" in res_df.columns
    assert list(res_df["p"]) == [20.0, 100.0]

    # Disallowed operation should raise ValueError
    with pytest.raises(ValueError):
        executor.execute_step(df, {"operation": "arbitrary_eval", "inputs": ["v"]})


def test_verification_gate_leakage_detection():
    gate = FeatureVerificationGate(target_column="anomaly")
    df = pd.DataFrame({"v": [1.0, 2.0], "anomaly": [0, 1]})
    # Engineered column that is identical to target
    eng_df = df.copy()
    eng_df["leaked_feature"] = eng_df["anomaly"]
    plan = [{"output_name": "leaked_feature", "operation": "copy", "inputs": ["anomaly"]}]
    metadata = [{"name": "leaked_feature", "data_type": "int", "origin": "test", "source_columns": ["anomaly"], "operation": "copy", "reason": "leak", "version": 1}]

    res = gate.verify(df, eng_df, plan, metadata)
    assert res["status"] == "FAILED"
    assert "leaked_feature" in res["offending_columns"]


def test_analysis_orchestrator():
    df = pd.read_csv("d:/predictive_framework/data/sample_high_voltage_power.csv")
    orch = AnalysisOrchestrator()
    out = orch.run(df, domain_name="high_voltage_power")
    assert out["status"] in ["PASSED", "PASSED WITH WARNINGS"]
    assert "engineered_dataframe" in out
    assert len(out["feature_plan"]) > 0
