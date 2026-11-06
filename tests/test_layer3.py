import pytest
import pandas as pd
from layer3_modelling.cv_strategy_selector import CVStrategySelector
from layer3_modelling.tabpfn_predictor import TabPFNZeroShotPredictor
from layer3_modelling.decision_engine import DecisionEngine
from layer3_modelling.adaptive_baseline import AdaptiveBaselinePredictor
from layer3_modelling.ensemble_predictor import EnsembleTabularPredictor
from layer3_modelling.modelling_orchestrator import ModellingOrchestrator
from layer2_analysis.analysis_orchestrator import AnalysisOrchestrator


def test_cv_strategy_time_series():
    selector = CVStrategySelector(n_splits=5)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=100, freq="1h"),
        "val": range(100),
        "anomaly": [0] * 95 + [1] * 5
    })
    res = selector.select_strategy(df, target_column="anomaly", time_column="timestamp")
    assert res["strategy_name"] == "TimeSeriesSplit"


def test_decision_engine_routing():
    engine = DecisionEngine()
    df_large = pd.DataFrame({"x": range(600), "y": range(600)})
    eda = {"correlations": {"high_correlations": []}, "target_analysis": {"is_imbalanced": False}}
    route = engine.evaluate_routing(df_large, eda)
    assert route["pathway"] == "AutoGluon_Ensemble"


def test_modelling_orchestrator_end_to_end():
    df = pd.read_csv("d:/predictive_framework/data/sample_high_voltage_power.csv")
    orch = AnalysisOrchestrator()
    an_res = orch.run(df, domain_name="high_voltage_power")

    mod_orch = ModellingOrchestrator(target_column="anomaly", time_column="timestamp")
    mod_res = mod_orch.run(
        engineered_df=an_res["engineered_dataframe"],
        eda_report=an_res["eda_report"],
        run_id="test-run-1"
    )
    assert mod_res["status"] == "COMPLETED"
    assert mod_res["champion_model"] in ["AutoGluon_Ensemble", "TabPFN_ZeroShot", "Adaptive_Baseline"]
    assert "unified_model_summary" in mod_res
    summary = mod_res["unified_model_summary"]
    assert summary["ready_for_reporting_layer"] is True
