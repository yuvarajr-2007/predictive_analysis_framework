"""
End-to-End Pipeline Runner for All 5 Layers:
1. Validation Layer
2. Analysis Layer
3. Modelling Layer
4. Reporting Layer
5. Versioning & Concerns Layer
"""
import sys
import os
import json
import pandas as pd

# Fix standard output encoding for Windows shells
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
os.environ["MPLCONFIGDIR"] = "d:/predictive_framework/.cache"

from layer1_validation import ValidationEngine
from layer2_analysis import AnalysisOrchestrator
from layer3_modelling import ModellingOrchestrator
from layer4_reporting import ReportingOrchestrator
from layer5_versioning import VersioningOrchestrator


def run_pipeline(csv_path: str, domain: str = "high_voltage_power", target_column: str = None, time_column: str = None):
    print("=" * 80)
    print("[RUN] EXECUTING FULL ANOMALY AGENTIC AI FRAMEWORK (LAYERS 1 -> 5)")
    print("=" * 80)

    # Ingest dataset
    print(f"\n[1] Ingesting dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"    Loaded {len(df)} rows and {len(df.columns)} columns.")

    # Auto-detect target and time columns if not explicitly provided
    if target_column is None:
        for c in ["Prototype_Condition_Label", "anomaly", "risk_score", "target", "label"]:
            if c in df.columns:
                target_column = c
                break
        if target_column is None:
            target_column = df.columns[-1]

    if time_column is None:
        for c in ["timestamp", "time", "date", "Cycle"]:
            if c in df.columns:
                time_column = c
                break

    print(f"    Target Column detected: '{target_column}'")
    print(f"    Temporal/Grouping Column: '{time_column}'")

    # --------------------------------------------------------------------------
    # LAYER 1: VALIDATION
    # --------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("[LAYER 1] VALIDATION LAYER")
    print("-" * 40)
    val_engine = ValidationEngine()
    val_res = val_engine.run(df, domain_name=domain)
    print(f"    Verdict: {val_res['overall_status']}")
    print(f"    Checks run: {val_res['checks_run']}")
    print(f"    Total issues flagged: {val_res['total_issues']}")

    if val_res["overall_status"] == "FAIL":
        print("[FAIL] Layer 1 FAILED! Pipeline stopped.")
        return

    validated_df = val_res["validated_dataframe"]
    run_id = val_res["run_id"]

    # --------------------------------------------------------------------------
    # LAYER 2: ANALYSIS
    # --------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("[LAYER 2] ANALYSIS LAYER")
    print("-" * 40)
    analysis_orch = AnalysisOrchestrator(target_column=target_column, time_column=time_column or "timestamp")
    analysis_res = analysis_orch.run(
        validated_df=validated_df,
        domain_name=domain,
        validation_package=val_res
    )
    print(f"    Analysis Status: {analysis_res['status']}")
    print(f"    Engineered Columns Planned: {len(analysis_res['feature_plan'])}")
    print(f"    Feature Verification Gate: {analysis_res['gate_result']['status']}")

    if analysis_res["status"] == "FAILED":
        print("[FAIL] Layer 2 FAILED! Pipeline stopped.")
        return

    engineered_df = analysis_res["engineered_dataframe"]
    eda_report = analysis_res["eda_report"]

    # --------------------------------------------------------------------------
    # LAYER 3: MODELLING
    # --------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("[LAYER 3] MODELLING LAYER")
    print("-" * 40)
    modelling_orch = ModellingOrchestrator(target_column=target_column, time_column=time_column or "timestamp")
    model_res = modelling_orch.run(
        engineered_df=engineered_df,
        eda_report=eda_report,
        run_id=run_id
    )
    print(f"    Modelling Status: {model_res['status']}")
    print(f"    Champion Model Selected: {model_res['champion_model']}")
    print(f"    CV Strategy Selected: {model_res['cv_strategy']['strategy_name']}")
    summary = model_res["unified_model_summary"]
    print(f"    ROC-AUC: {summary['evaluation_metrics'].get('ROC-AUC')}")
    print(f"    PR-AUC:  {summary['evaluation_metrics'].get('PR-AUC')}")
    print(f"    Anomalies Flagged: {summary['anomaly_statistics']['anomalies_flagged']} / {summary['anomaly_statistics']['total_samples']}")

    # --------------------------------------------------------------------------
    # LAYER 4: REPORTING
    # --------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("[LAYER 4] REPORTING LAYER")
    print("-" * 40)
    reporting_orch = ReportingOrchestrator()
    report_res = reporting_orch.run(
        val_result=val_res,
        analysis_result=analysis_res,
        model_result=model_res
    )
    print(f"    Reporting Status: {report_res['status']}")
    print(f"    Generated HTML Report: {report_res['report_path']}")
    print(f"    Executive Narrative Grounded: {report_res['narrative']['numbers_grounded']}")

    # --------------------------------------------------------------------------
    # LAYER 5: VERSIONING AND CONCERNS
    # --------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("[LAYER 5] VERSIONING & CONCERNS LAYER")
    print("-" * 40)
    versioning_orch = VersioningOrchestrator()
    vers_res = versioning_orch.record_run(
        run_id=run_id,
        experiment_name=f"{domain}_experiment",
        raw_df=df,
        val_result=val_res,
        analysis_result=analysis_res,
        model_result=model_res,
        report_path=report_res["report_path"]
    )
    print(f"    Versioning Status: {vers_res['status']}")
    print(f"    Dataset Hash (MD5): {vers_res['dataset_hash']}")
    print(f"    Checkpoint Manifest: {vers_res['checkpoint_manifest']}")
    print(f"    Features Registered in Feature Store: {vers_res['features_registered_count']}")

    # Quick Drift Detection Demo
    drift_res = versioning_orch.drift_detector.detect_drift(
        reference_df=df,
        current_df=df.sample(frac=0.8, random_state=99)
    )
    print(f"    Drift Detection Baseline Check: Drift Alert = {drift_res['dataset_drift_alert']} ({drift_res['drifted_features_count']} drifted features)")

    print("\n" + "=" * 80)
    print("[SUCCESS] ALL 5 LAYERS EXECUTED SUCCESSFULLY WITH FULL TRACEABILITY!")
    print("=" * 80)

    return val_res, analysis_res, model_res, report_res, vers_res


if __name__ == "__main__":
    run_pipeline("d:/predictive_framework/data/sample_high_voltage_power.csv", domain="high_voltage_power")
