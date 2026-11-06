"""
Comprehensive Test Runner using Python's standard unittest library.
Tests all 5 layers:
  - Layer 1: Validation
  - Layer 2: Analysis
  - Layer 3: Modelling
  - Layer 4: Reporting
  - Layer 5: Versioning and Concerns
"""
import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

# Set paths
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from layer1_validation.schema_validator import SchemaValidator
from layer1_validation.null_validator import NullValidator
from layer1_validation.domain_validator import DomainValidator
from layer1_validation.validation_engine import ValidationEngine

from layer2_analysis.eda_agent import EDAAgent
from layer2_analysis.dsl_executor import DSLExecutor
from layer2_analysis.feature_engineering_agent import FeatureEngineeringAgent
from layer2_analysis.verification_gate import FeatureVerificationGate
from layer2_analysis.analysis_orchestrator import AnalysisOrchestrator

from layer3_modelling.cv_strategy_selector import CVStrategySelector
from layer3_modelling.tabpfn_predictor import TabPFNZeroShotPredictor
from layer3_modelling.decision_engine import DecisionEngine
from layer3_modelling.adaptive_baseline import AdaptiveBaselinePredictor
from layer3_modelling.ensemble_predictor import EnsembleTabularPredictor
from layer3_modelling.modelling_orchestrator import ModellingOrchestrator

from layer4_reporting.results_adapter import ResultsAdapter
from layer4_reporting.storytelling_agent import StorytellingAgent
from layer4_reporting.html_report_builder import HTMLReportBuilder
from layer4_reporting.reporting_orchestrator import ReportingOrchestrator

from layer5_versioning.experiment_tracker import ExperimentTracker
from layer5_versioning.checkpoint_manager import CheckpointManager
from layer5_versioning.drift_detector import DriftDetector
from layer5_versioning.feature_registry import FeatureRegistryClient
from layer5_versioning.versioning_orchestrator import VersioningOrchestrator


class TestLayer1Validation(unittest.TestCase):
    def test_schema_validator_clean(self):
        df = pd.DataFrame({"timestamp": ["2026-01-01"], "machine_id": ["M1"], "voltage": [220.0], "anomaly": [0]})
        config = {"required_columns": ["machine_id", "voltage", "anomaly"]}
        res = SchemaValidator().validate(df, config)
        self.assertEqual(res["status"], "PASS")

    def test_schema_validator_missing_column(self):
        df = pd.DataFrame({"machine_id": ["M1"]})
        config = {"required_columns": ["machine_id", "voltage"]}
        res = SchemaValidator().validate(df, config)
        self.assertEqual(res["status"], "FAIL")

    def test_null_validator_threshold(self):
        data = [1.0] * 8 + [None, None]
        df = pd.DataFrame({"val": data})
        res = NullValidator().validate(df, {"missing_threshold": 0.05})
        self.assertEqual(res["status"], "WARNING")

    def test_domain_validator_physical_bounds(self):
        df = pd.DataFrame({"voltage": [-15.0, 220.0]})
        config = {"physical_constraints": [{"column": "voltage", "rule": "non_negative"}]}
        res = DomainValidator().validate(df, config)
        self.assertEqual(res["status"], "FAIL")


class TestLayer2Analysis(unittest.TestCase):
    def test_eda_agent_overview_and_insights(self):
        df = pd.read_csv(root_dir / "data" / "sample_high_voltage_power.csv")
        agent = EDAAgent(target_column="anomaly")
        profile = agent.run_parallel(df)
        self.assertIn("insights", profile)
        self.assertTrue(profile["target_analysis"]["detected"])

    def test_dsl_executor_whitelisted_operations(self):
        executor = DSLExecutor()
        df = pd.DataFrame({"v": [10.0, 20.0], "i": [2.0, 5.0]})
        res_df, meta = executor.execute_step(df, {
            "operation": "multiply",
            "inputs": ["v", "i"],
            "output_name": "p",
            "reason": "power"
        })
        self.assertIn("p", res_df.columns)
        self.assertEqual(list(res_df["p"]), [20.0, 100.0])

    def test_verification_gate_leakage(self):
        gate = FeatureVerificationGate(target_column="anomaly")
        df = pd.DataFrame({"v": [1.0, 2.0], "anomaly": [0, 1]})
        eng_df = df.copy()
        eng_df["leaked_feature"] = eng_df["anomaly"]
        plan = [{"output_name": "leaked_feature", "operation": "copy", "inputs": ["anomaly"]}]
        metadata = [{"name": "leaked_feature", "data_type": "int", "origin": "test", "source_columns": ["anomaly"], "operation": "copy", "reason": "leak", "version": 1}]
        res = gate.verify(df, eng_df, plan, metadata)
        self.assertEqual(res["status"], "FAILED")


class TestLayer3Modelling(unittest.TestCase):
    def test_cv_strategy_time_series(self):
        selector = CVStrategySelector(n_splits=5)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=100, freq="1h"),
            "val": range(100),
            "anomaly": [0] * 95 + [1] * 5
        })
        res = selector.select_strategy(df, target_column="anomaly", time_column="timestamp")
        self.assertEqual(res["strategy_name"], "TimeSeriesSplit")

    def test_decision_engine_routing(self):
        engine = DecisionEngine()
        df_large = pd.DataFrame({"x": range(600), "y": range(600)})
        eda = {"correlations": {"high_correlations": []}, "target_analysis": {"is_imbalanced": False}}
        route = engine.evaluate_routing(df_large, eda)
        self.assertEqual(route["pathway"], "AutoGluon_Ensemble")


class TestLayer4Reporting(unittest.TestCase):
    def test_storytelling_agent_no_forbidden_claims(self):
        agent = StorytellingAgent()
        adapted = {
            "run_id": "test-run",
            "domain": "power",
            "validation": {"overall_status": "PASS", "total_issues": 0, "checks_run": ["schema"]},
            "analysis": {"problem_type": "binary_classification", "rows": 100, "columns": 5, "planned_features_count": 2, "gate_status": "PASSED"},
            "modelling": {
                "champion_model": "Ensemble",
                "cv_strategy": "KFold",
                "cv_reason": "Balanced split",
                "metrics": {"ROC-AUC": 0.95, "PR-AUC": 0.92},
                "anomaly_statistics": {"decision_threshold": 0.5, "anomalies_flagged": 5, "anomaly_rate_percent": 5.0}
            }
        }
        narrative = agent.draft_narrative(adapted)
        self.assertTrue(narrative["claims_reviewed"])
        self.assertIn("test-run", narrative["executive_summary"])

    def test_html_report_builder_generation(self):
        builder = HTMLReportBuilder()
        adapted = {
            "run_id": "unit_test_run",
            "domain": "high_voltage_power",
            "pipeline_version": "2.0.0",
            "started_at": "2026-11-06T12:00:00Z",
            "validation": {"overall_status": "PASS", "checks_run": ["schema"], "warnings": []},
            "analysis": {"gate_status": "PASSED"},
            "modelling": {
                "champion_model": "AutoGluon_Ensemble",
                "cv_strategy": "TimeSeriesSplit",
                "cv_reason": "Temporal",
                "decision_route": "Ensemble",
                "metrics": {"ROC-AUC": 1.0, "PR-AUC": 1.0, "F1-Score": 0.95},
                "anomaly_statistics": {"decision_threshold": 0.5},
                "candidate_benchmarks": {},
                "feature_importance": {"voltage": 0.4}
            }
        }
        narrative = {"executive_summary": "Test summary", "operational_findings": [], "trust_and_limitations": []}
        explain = {"disclaimer": "Test disclaimer"}
        path = builder.build_report(adapted, narrative, explain)
        self.assertTrue(path.exists())


class TestLayer5VersioningAndConcerns(unittest.TestCase):
    def test_dataset_hashing_and_tracking(self):
        tracker = ExperimentTracker()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        h1 = tracker.calculate_dataset_hash(df)
        self.assertEqual(len(h1), 32)  # MD5 hex length

        # Track run
        tracker.start_run(run_id="test_exp_run", experiment_name="test_suite", df=df)
        tracker.log_param("param_test", "123", run_id="test_exp_run")
        tracker.log_metric("metric_test", 0.99, run_id="test_exp_run")
        tracker.end_run("FINISHED", run_id="test_exp_run")
        history = tracker.get_runs_history()
        self.assertTrue(any(r["run_id"] == "test_exp_run" for r in history))

    def test_checkpoint_save_and_load(self):
        mgr = CheckpointManager()
        df = pd.DataFrame({"sensor": [10.5, 20.1]})
        state = {"weights": [0.1, 0.9]}
        manifest_p = mgr.save_checkpoint(run_id="chk_test_run", stage_name="STAGE_1", df=df, state_dict=state)
        self.assertTrue(manifest_p.exists())

        loaded_df, loaded_state = mgr.load_checkpoint(run_id="chk_test_run", stage_name="STAGE_1")
        self.assertEqual(len(loaded_df), 2)
        self.assertEqual(loaded_state["weights"], [0.1, 0.9])

    def test_drift_detection_psi_and_ks(self):
        detector = DriftDetector()
        np.random.seed(42)
        ref_df = pd.DataFrame({"v": np.random.normal(220, 5, 200)})
        # Drastic shift: mean shifts from 220 to 280
        cur_df = pd.DataFrame({"v": np.random.normal(280, 5, 200)})
        drift_res = detector.detect_drift(ref_df, cur_df)
        self.assertTrue(drift_res["per_feature_details"]["v"]["drift_detected"])
        self.assertGreater(drift_res["per_feature_details"]["v"]["psi"], 0.20)

    def test_feature_registry_register_and_retrieve(self):
        reg = FeatureRegistryClient()
        s = pd.Series([100, 200, 300], name="test_feat")
        reg.register_feature(
            name="test_feat",
            series=s,
            operation="multiply",
            source_columns=["a", "b"],
            reason="test registration"
        )
        cat = reg.list_registered_features()
        self.assertIn("test_feat", cat)

        retrieved_df = reg.get_features(["test_feat"])
        self.assertEqual(list(retrieved_df["test_feat"]), [100, 200, 300])


if __name__ == "__main__":
    unittest.main(verbosity=2)
