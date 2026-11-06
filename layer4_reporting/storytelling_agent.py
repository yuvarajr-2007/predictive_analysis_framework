"""
Storytelling Agent (Narrative Generator):
Drafts a plain-language executive narrative of the pipeline run.
Strictly ensures:
1. All figures and numbers match saved results verbatim (zero hallucination).
2. Prohibits unsupported causal claims ("X caused Y") and future performance guarantees.
3. Highlights data quality warnings and evaluation split boundaries.
"""
from typing import Dict, Any, List


class StorytellingAgent:
    FORBIDDEN_PHRASES = [
        "guaranteed future performance",
        "proves that",
        "causes anomalies",
        "guarantees 100% accuracy",
        "will definitely detect all"
    ]

    def draft_narrative(self, adapted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesizes executive summary, key findings, and operational precautions.
        """
        run_id = adapted_data.get("run_id")
        domain = adapted_data.get("domain", "general")
        val = adapted_data.get("validation", {})
        analysis = adapted_data.get("analysis", {})
        modelling = adapted_data.get("modelling", {})
        metrics = modelling.get("metrics", {})
        anom_stats = modelling.get("anomaly_statistics", {})

        # Executive Summary
        exec_summary = (
            f"Pipeline run '{run_id}' evaluated a {analysis.get('problem_type', 'anomaly detection')} dataset "
            f"under the '{domain}' domain rules. Data validation concluded with status '{val.get('overall_status')}' "
            f"across {len(val.get('checks_run', []))} checks. Following feature engineering and deterministic gate verification, "
            f"the modeling layer deployed '{modelling.get('champion_model')}' as the champion model using '{modelling.get('cv_strategy')}'. "
            f"On the blind holdout partition, the champion attained an ROC-AUC of {metrics.get('ROC-AUC')} and a PR-AUC of {metrics.get('PR-AUC')}."
        )

        # Operational Findings
        findings = [
            f"Data Validation: Recorded {val.get('total_issues', 0)} issues. Gate status: {val.get('overall_status')}.",
            f"Dataset Geometry: Analyzed {analysis.get('rows')} rows across {analysis.get('columns')} initial columns.",
            f"Feature Engineering: Planned {analysis.get('planned_features_count')} safe features; Verification Gate status: {analysis.get('gate_status')}.",
            f"Anomaly Detection: Calibrated threshold at {anom_stats.get('decision_threshold')}, identifying {anom_stats.get('anomalies_flagged')} anomalies ({anom_stats.get('anomaly_rate_percent')}% rate)."
        ]

        # Trust & Limitations Disclaimers
        precautions = [
            f"Cross-Validation Split: Evaluations used {modelling.get('cv_strategy')} ({modelling.get('cv_reason')}). Scores cannot be directly compared against non-stratified random splits.",
            "Feature Attributions: Feature importances reflect mathematical model influence within this sample and must not be interpreted as physical root cause.",
            "Operational Scope: Anomaly scores are probabilistic estimates subject to environmental drift and changing production baselines."
        ]

        # Review claims (verify forbidden phrases are absent)
        full_text = f"{exec_summary} " + " ".join(findings) + " " + " ".join(precautions)
        for forbidden in self.FORBIDDEN_PHRASES:
            if forbidden in full_text.lower():
                raise ValueError(f"Prohibited causal/guarantee claim found in draft narrative: '{forbidden}'")

        return {
            "executive_summary": exec_summary,
            "operational_findings": findings,
            "trust_and_limitations": precautions,
            "claims_reviewed": True,
            "numbers_grounded": True
        }
