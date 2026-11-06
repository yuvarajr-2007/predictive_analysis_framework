"""
Checkpoint Manager:
Enables atomic serialization of pipeline state (Parquet for DataFrames, Joblib for Python objects),
maintains an auditable JSON checkpoint manifest, and powers seamless recovery from failure.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import joblib


class CheckpointManager:
    CHECKPOINT_NODES = [
        "POST_VALIDATION",
        "POST_EDA",
        "POST_FEATURE_ENGINEERING",
        "POST_MODELLING"
    ]

    def __init__(self, checkpoints_dir: Optional[str] = None):
        if checkpoints_dir:
            self.checkpoints_dir = Path(checkpoints_dir)
        else:
            self.checkpoints_dir = Path(__file__).resolve().parent.parent / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def _manifest_path(self, run_id: str) -> Path:
        return self.checkpoints_dir / f"{run_id}_manifest.json"

    def save_checkpoint(
        self,
        run_id: str,
        stage_name: str,
        df: Optional[pd.DataFrame] = None,
        state_dict: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Saves an atomic checkpoint of data (Parquet) and state (Joblib), updating manifest.
        """
        stage_dir = self.checkpoints_dir / run_id / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        parquet_file = None
        if df is not None:
            parquet_file = stage_dir / "data.parquet"
            # Fast Parquet write
            df.to_parquet(parquet_file, index=False)

        joblib_file = None
        if state_dict is not None:
            joblib_file = stage_dir / "state.joblib"
            joblib.dump(state_dict, joblib_file)

        # Update manifest atomically
        manifest_p = self._manifest_path(run_id)
        manifest = {"run_id": run_id, "stages": {}}
        if manifest_p.exists():
            try:
                with open(manifest_p, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                pass

        manifest["latest_completed_stage"] = stage_name
        manifest["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest["stages"][stage_name] = {
            "parquet_path": str(parquet_file) if parquet_file else None,
            "joblib_path": str(joblib_file) if joblib_file else None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        with open(manifest_p, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest_p

    def load_checkpoint(
        self,
        run_id: str,
        stage_name: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """
        Loads saved DataFrame and state dictionary for a specified stage.
        """
        manifest_p = self._manifest_path(run_id)
        if not manifest_p.exists():
            return None, None

        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        stage_info = manifest.get("stages", {}).get(stage_name)
        if not stage_info:
            return None, None

        loaded_df = None
        if stage_info.get("parquet_path") and Path(stage_info["parquet_path"]).exists():
            loaded_df = pd.read_parquet(stage_info["parquet_path"])

        loaded_state = None
        if stage_info.get("joblib_path") and Path(stage_info["joblib_path"]).exists():
            loaded_state = joblib.load(stage_info["joblib_path"])

        return loaded_df, loaded_state

    def get_latest_checkpoint(self, run_id: str) -> Optional[str]:
        manifest_p = self._manifest_path(run_id)
        if not manifest_p.exists():
            return None
        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            return manifest.get("latest_completed_stage")
