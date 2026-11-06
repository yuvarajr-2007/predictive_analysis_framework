"""
Experiment Tracker (AI Flow Tracking / MLflow Architecture):
Maintains immutable records of all pipeline runs, hyperparameters, metrics, and artifacts
backed by a persistent SQLite store and local artifact repository.
Calculates MD5 dataset hashes for data integrity.
"""
import os
import shutil
import sqlite3
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd


class ExperimentTracker:
    def __init__(self, exp_dir: Optional[str] = None):
        if exp_dir:
            self.exp_dir = Path(exp_dir)
        else:
            self.exp_dir = Path(__file__).resolve().parent.parent / "experiments"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.exp_dir / "mlflow_tracking.db"
        self.artifacts_dir = self.exp_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.current_run_id = None
        self._init_db()

    def _init_db(self):
        """Initializes SQLite schema for runs, parameters, metrics, and artifacts."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_name TEXT,
                    dataset_hash TEXT,
                    status TEXT,
                    started_at TEXT,
                    ended_at TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS params (
                    run_id TEXT,
                    param_key TEXT,
                    param_val TEXT,
                    PRIMARY KEY(run_id, param_key)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    run_id TEXT,
                    metric_key TEXT,
                    metric_val REAL,
                    step INTEGER,
                    timestamp TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    run_id TEXT,
                    artifact_name TEXT,
                    artifact_path TEXT
                )
            """)
            conn.commit()

    @staticmethod
    def calculate_dataset_hash(df: pd.DataFrame) -> str:
        """Computes deterministic MD5 hash of dataframe content for integrity validation."""
        sample_bytes = df.to_csv(index=False).encode("utf-8")
        return hashlib.md5(sample_bytes).hexdigest()

    def start_run(self, run_id: str, experiment_name: str = "default_anomaly_experiment", df: Optional[pd.DataFrame] = None):
        """Instantiates an experiment run."""
        self.current_run_id = run_id
        d_hash = self.calculate_dataset_hash(df) if df is not None else "no_df_supplied"
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO runs (run_id, experiment_name, dataset_hash, status, started_at, ended_at)
                VALUES (?, ?, ?, 'RUNNING', ?, NULL)
            """, (run_id, experiment_name, d_hash, now))
            conn.commit()

    def log_param(self, key: str, value: Any, run_id: Optional[str] = None):
        rid = run_id or self.current_run_id
        if not rid: return
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO params (run_id, param_key, param_val)
                VALUES (?, ?, ?)
            """, (rid, key, str(value)))
            conn.commit()

    def log_metric(self, key: str, value: float, step: int = 0, run_id: Optional[str] = None):
        rid = run_id or self.current_run_id
        if not rid: return
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO metrics (run_id, metric_key, metric_val, step, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (rid, key, float(value), step, now))
            conn.commit()

    def log_artifact(self, local_file_path: str, run_id: Optional[str] = None) -> str:
        rid = run_id or self.current_run_id
        if not rid: return ""
        src = Path(local_file_path)
        if not src.exists(): return ""

        run_art_dir = self.artifacts_dir / rid
        run_art_dir.mkdir(parents=True, exist_ok=True)
        dest = run_art_dir / src.name
        shutil.copy2(src, dest)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO artifacts (run_id, artifact_name, artifact_path)
                VALUES (?, ?, ?)
            """, (rid, src.name, str(dest)))
            conn.commit()
        return str(dest)

    def end_run(self, status: str = "FINISHED", run_id: Optional[str] = None):
        rid = run_id or self.current_run_id
        if not rid: return
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE runs SET status = ?, ended_at = ? WHERE run_id = ?
            """, (status, now, rid))
            conn.commit()

    def get_runs_history(self) -> List[Dict[str, Any]]:
        """Returns all recorded experiment runs with their params and latest metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT run_id, experiment_name, dataset_hash, status, started_at, ended_at FROM runs ORDER BY started_at DESC")
            run_rows = cursor.fetchall()
            history = []

            for r in run_rows:
                rid = r[0]
                cursor.execute("SELECT param_key, param_val FROM params WHERE run_id = ?", (rid,))
                params = dict(cursor.fetchall())
                cursor.execute("SELECT metric_key, metric_val FROM metrics WHERE run_id = ?", (rid,))
                metrics = dict(cursor.fetchall())

                history.append({
                    "run_id": rid,
                    "experiment_name": r[1],
                    "dataset_hash": r[2],
                    "status": r[3],
                    "started_at": r[4],
                    "ended_at": r[5],
                    "params": params,
                    "metrics": metrics
                })
            return history
