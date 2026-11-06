"""
Reporting Layer Orchestrator:
Coordinates ResultsAdapter, StorytellingAgent, SHAPExplainer, and HTMLReportBuilder.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from .results_adapter import ResultsAdapter
from .shap_explainer import SHAPExplainer
from .storytelling_agent import StorytellingAgent
from .html_report_builder import HTMLReportBuilder


class ReportingOrchestrator:
    def __init__(self, reports_dir: Optional[str] = None):
        self.adapter = ResultsAdapter()
        self.explainer = SHAPExplainer()
        self.storyteller = StorytellingAgent()
        self.builder = HTMLReportBuilder(output_dir=reports_dir)

    def run(
        self,
        val_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        model_result: Dict[str, Any],
        model_instance: Optional[Any] = None,
        X_holdout: Optional[pd.DataFrame] = None,
        y_holdout: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Executes the full reporting pipeline and saves the HTML report.
        """
        # 1. Normalize upstream results
        adapted = self.adapter.adapt(
            val_result=val_result,
            analysis_result=analysis_result,
            model_result=model_result
        )

        # 2. Compute Permutation Importance & SHAP sensitivity
        if model_instance is not None and X_holdout is not None and y_holdout is not None:
            explain_res = self.explainer.explain(
                model_obj=model_instance,
                X_holdout=X_holdout,
                y_holdout=y_holdout
            )
        else:
            explain_res = {
                "method": "Feature_Importance_Attribution",
                "status": "available",
                "disclaimer": (
                    "Feature importance scores reflect model sensitivity within this dataset and DO NOT "
                    "prove physical or empirical causation."
                ),
                "attributions": model_result.get("unified_model_summary", {}).get("feature_importance", {})
            }

        # 3. Generate verified narrative
        narrative = self.storyteller.draft_narrative(adapted)

        # 4. Build and save self-contained HTML report
        report_path = self.builder.build_report(
            adapted_data=adapted,
            narrative=narrative,
            explainability=explain_res
        )

        return {
            "status": "COMPLETED",
            "report_path": str(report_path),
            "run_id": adapted["run_id"],
            "narrative": narrative,
            "explainability": explain_res,
            "adapted_data": adapted
        }
