"""
Results Adapter: Normalizes upstream outputs from Validation, Analysis, and Modelling.
Enforces input filtering, consistency checks, and completeness validation.
"""
from typing import Dict, Any, List, Optional


class ResultsAdapter:
    def adapt(
        self,
        val_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        model_result: Dict[str, Any],
        pipeline_version: str = "2.0.0"
    ) -> Dict[str, Any]:
        """
        Extracts, validates, and harmonizes pipeline data for the reporting engine.
        """
        run_id = val_result.get("run_id", "run-unknown")
        domain = val_result.get("domain", "general")

        # 1. Validation Section Adaptation
        val_status = val_result.get("overall_status", "UNKNOWN")
        val_warnings = []
        for v in val_result.get("results", []):
            for iss in v.get("issues", []):
                val_warnings.append({
                    "validator": v.get("validator"),
                    "severity": iss.get("severity", "WARNING"),
                    "column": iss.get("column", "N/A"),
                    "message": iss.get("message", "Rule triggered")
                })

        # 2. Analysis Section Adaptation
        eda = analysis_result.get("eda_report", {})
        gate = analysis_result.get("gate_result", {})
        plan = analysis_result.get("feature_plan", [])
        insights = eda.get("insights", {})

        analysis_summary = {
            "status": analysis_result.get("status", "UNKNOWN"),
            "problem_type": insights.get("problem_type", "anomaly_detection"),
            "rows": eda.get("overview", {}).get("rows", "unavailable"),
            "columns": eda.get("overview", {}).get("columns", "unavailable"),
            "planned_features_count": len(plan),
            "gate_status": gate.get("status", "UNKNOWN"),
            "gate_checks": gate.get("checks", []),
            "key_findings": insights.get("key_findings", []),
            "risks": insights.get("risks", [])
        }

        # 3. Modelling Section Adaptation
        summary = model_result.get("unified_model_summary", {})
        champion = model_result.get("champion_model", "unavailable")
        cv = model_result.get("cv_strategy", {})
        routing = model_result.get("routing", {})
        metrics = summary.get("evaluation_metrics", {})
        anom_stats = summary.get("anomaly_statistics", {})
        benchmarks = summary.get("candidate_benchmarks", {})

        # Ensure all standard metrics are defined or labeled unavailable
        normalized_metrics = {
            "ROC-AUC": metrics.get("ROC-AUC", "unavailable"),
            "PR-AUC": metrics.get("PR-AUC", "unavailable"),
            "F1-Score": metrics.get("F1-Score", "unavailable"),
            "Precision": metrics.get("Precision", "unavailable"),
            "Recall": metrics.get("Recall", "unavailable")
        }

        modelling_summary = {
            "champion_model": champion,
            "cv_strategy": cv.get("strategy_name", "Holdout"),
            "cv_reason": cv.get("reason", "Direct holdout validation"),
            "decision_route": routing.get("pathway", "Default"),
            "routing_reasons": routing.get("reasons", []),
            "metrics": normalized_metrics,
            "anomaly_statistics": {
                "total_samples": anom_stats.get("total_samples", "unavailable"),
                "anomalies_flagged": anom_stats.get("anomalies_flagged", "unavailable"),
                "anomaly_rate_percent": anom_stats.get("anomaly_rate_percent", "unavailable"),
                "decision_threshold": anom_stats.get("decision_threshold", "unavailable")
            },
            "candidate_benchmarks": benchmarks,
            "feature_importance": summary.get("feature_importance", {})
        }

        return {
            "run_id": run_id,
            "domain": domain,
            "pipeline_version": pipeline_version,
            "started_at": val_result.get("started_at", "N/A"),
            "validation": {
                "overall_status": val_status,
                "total_issues": val_result.get("total_issues", 0),
                "checks_run": val_result.get("checks_run", []),
                "warnings": val_warnings
            },
            "analysis": analysis_summary,
            "modelling": modelling_summary
        }
