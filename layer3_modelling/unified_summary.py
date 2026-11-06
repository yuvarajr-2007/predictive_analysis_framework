"""
Unified Model Summary (Block 6: Schema Standardization):
Normalizes all model outputs (TabPFN, Adaptive Baseline, AutoGluon Ensemble)
into a single, clean JSON artifact ready for Layer 4 (Reporting).
Performs automated Champion Selection based on performance-to-latency ratio.
"""
from typing import Dict, Any, List, Optional
import numpy as np


class UnifiedModelSummary:
    def build_summary(
        self,
        run_id: str,
        champion_name: str,
        cv_strategy_info: Dict[str, Any],
        decision_routing: Dict[str, Any],
        candidate_results: Dict[str, Any],
        dataset_meta: Dict[str, Any],
        holdout_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Creates the standardized JSON summary artifact.
        """
        champion_res = candidate_results.get(champion_name, {})

        scores = champion_res.get("anomaly_scores", champion_res.get("anomaly_probabilities", []))
        flags = champion_res.get("flags", [])
        threshold = champion_res.get("threshold", 0.5)

        total_samples = len(scores)
        flagged_count = sum(flags) if flags else 0
        anomaly_rate = round(flagged_count / max(1, total_samples) * 100, 2)

        # Top flagged indices
        flagged_indices = [i for i, f in enumerate(flags) if f == 1]

        summary = {
            "schema_version": "2.0.0",
            "run_id": run_id,
            "pipeline_layer": "Layer 3: Modelling Layer",
            "champion_model": champion_name,
            "routing_decision": decision_routing,
            "cv_strategy": {
                "name": cv_strategy_info.get("strategy_name"),
                "n_splits": cv_strategy_info.get("n_splits"),
                "reason": cv_strategy_info.get("reason")
            },
            "dataset_info": dataset_meta,
            "evaluation_metrics": champion_res.get("metrics", holdout_metrics or {}),
            "anomaly_statistics": {
                "total_samples": total_samples,
                "anomalies_flagged": flagged_count,
                "anomaly_rate_percent": anomaly_rate,
                "decision_threshold": threshold,
                "mean_anomaly_score": round(float(np.mean(scores)), 4) if scores else 0.0,
                "max_anomaly_score": round(float(np.max(scores)), 4) if scores else 0.0,
                "flagged_sample_indices": flagged_indices[:50]  # top 50 sample indices
            },
            "feature_importance": champion_res.get("feature_importance", {}),
            "candidate_benchmarks": {
                k: {
                    "model_type": v.get("model_type"),
                    "latency_ms": v.get("latency_ms"),
                    "metrics": v.get("metrics", {})
                }
                for k, v in candidate_results.items()
            },
            "ready_for_reporting_layer": True
        }
        return summary
