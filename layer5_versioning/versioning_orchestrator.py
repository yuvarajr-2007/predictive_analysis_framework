"""
Versioning and Concerns Orchestrator (Layer 5 Master Coordinator):
Synchronizes Experiment Tracking (MLflow), Checkpointing, Drift Monitoring,
Structured Logging, and Feature Registry operations.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from .structured_logger import StructuredLogger, timer
from .experiment_tracker import ExperimentTracker
from .checkpoint_manager import CheckpointManager
from .drift_detector import DriftDetector
from .feature_registry import FeatureRegistryClient


class VersioningOrchestrator:
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent
        self.logger = StructuredLogger(log_dir=str(self.base_dir / "experiments"))
        self.tracker = ExperimentTracker(exp_dir=str(self.base_dir / "experiments"))
        self.checkpointer = CheckpointManager(checkpoints_dir=str(self.base_dir / "checkpoints"))
        self.drift_detector = DriftDetector()
        self.feature_registry = FeatureRegistryClient(store_dir=str(self.base_dir / "feature_store"))

    def record_run(
        self,
        run_id: str,
        experiment_name: str,
        raw_df: pd.DataFrame,
        val_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        model_result: Dict[str, Any],
        report_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Records an end-to-end experiment run across tracking, checkpoints, and registry.
        """
        self.logger.info("versioning_orchestrator", f"Commencing versioning capture for run {run_id}", agent_id="layer5")

        # 1. Start experiment tracking run
        self.tracker.start_run(run_id=run_id, experiment_name=experiment_name, df=raw_df)

        # 2. Log configuration and domain parameters
        self.tracker.log_param("domain", val_result.get("domain", "general"), run_id=run_id)
        self.tracker.log_param("validation_status", val_result.get("overall_status"), run_id=run_id)
        self.tracker.log_param("champion_model", model_result.get("champion_model"), run_id=run_id)
        self.tracker.log_param("cv_strategy", model_result.get("cv_strategy", {}).get("strategy_name"), run_id=run_id)

        # 3. Log metrics
        summary = model_result.get("unified_model_summary", {})
        metrics = summary.get("evaluation_metrics", {})
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.tracker.log_metric(k, float(v), run_id=run_id)

        # 4. Save final checkpoint manifest
        eng_df = analysis_result.get("engineered_dataframe")
        chk_path = self.checkpointer.save_checkpoint(
            run_id=run_id,
            stage_name="POST_MODELLING",
            df=eng_df,
            state_dict={"model_summary": summary}
        )

        # 5. Register newly engineered features
        registered_features = []
        if eng_df is not None:
            metadata_list = analysis_result.get("feature_metadata", [])
            for m in metadata_list:
                fname = m.get("name")
                if fname in eng_df.columns:
                    reg_meta = self.feature_registry.register_feature(
                        name=fname,
                        series=eng_df[fname],
                        operation=m.get("operation", "custom"),
                        source_columns=m.get("source_columns", []),
                        reason=m.get("reason", "Engineered feature"),
                        origin=m.get("origin", "feature_engineering_agent")
                    )
                    registered_features.append(reg_meta)

        # 6. Log report artifact if present
        if report_path and Path(report_path).exists():
            self.tracker.log_artifact(report_path, run_id=run_id)

        # 7. End run
        self.tracker.end_run(status="FINISHED", run_id=run_id)
        self.logger.info("versioning_orchestrator", f"Run {run_id} successfully registered in MLflow tracking.", agent_id="layer5")

        return {
            "status": "RECORDED",
            "run_id": run_id,
            "dataset_hash": self.tracker.calculate_dataset_hash(raw_df),
            "checkpoint_manifest": str(chk_path),
            "features_registered_count": len(registered_features),
            "mlflow_db_path": str(self.tracker.db_path)
        }
