"""
Modelling Layer Orchestrator:
Coordinates CV selection, fast TabPFN zero-shot inference, Decision Engine routing,
Adaptive Baseline vs Ensemble execution, Champion Selection, and Unified Model Summary packaging.
"""
import time
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .cv_strategy_selector import CVStrategySelector
from .tabpfn_predictor import TabPFNZeroShotPredictor
from .decision_engine import DecisionEngine
from .adaptive_baseline import AdaptiveBaselinePredictor
from .ensemble_predictor import EnsembleTabularPredictor
from .unified_summary import UnifiedModelSummary


class ModellingOrchestrator:
    def __init__(self, target_column: str = "anomaly", time_column: str = "timestamp"):
        self.target_column = target_column
        self.time_column = time_column
        self.cv_selector = CVStrategySelector(n_splits=5)
        self.tabpfn = TabPFNZeroShotPredictor(max_rows=10000)
        self.decision_engine = DecisionEngine()
        self.adaptive_baseline = AdaptiveBaselinePredictor()
        self.ensemble = EnsembleTabularPredictor()
        self.summarizer = UnifiedModelSummary()

    def run(
        self,
        engineered_df: pd.DataFrame,
        eda_report: Dict[str, Any],
        run_id: str = "run-001",
        stream_mode: bool = False
    ) -> Dict[str, Any]:
        start_time = time.time()

        if engineered_df is None or len(engineered_df) == 0:
            return {"status": "FAILED", "error": "Engineered dataframe is empty."}

        # 1. CV Strategy Selection
        cv_info = self.cv_selector.select_strategy(
            df=engineered_df,
            target_column=self.target_column,
            time_column=self.time_column
        )

        # 2. Dynamic Holdout Isolation (20% blind holdout)
        has_target = self.target_column in engineered_df.columns
        stratify = None
        if has_target and eda_report.get("target_analysis", {}).get("is_imbalanced"):
            stratify = engineered_df[self.target_column]

        # If time series, split chronologically without shuffle
        if cv_info["strategy_name"] == "TimeSeriesSplit":
            split_idx = int(len(engineered_df) * 0.8)
            train_df = engineered_df.iloc[:split_idx].copy()
            holdout_df = engineered_df.iloc[split_idx:].copy()
        else:
            train_df, holdout_df = train_test_split(
                engineered_df,
                test_size=0.20,
                random_state=42,
                stratify=stratify
            )

        # Prepare X and y for train and holdout
        X_train, y_train = self.ensemble._prepare_features(train_df, self.target_column)
        X_holdout, y_holdout = self.ensemble._prepare_features(holdout_df, self.target_column)

        candidate_results = {}

        # 3. Fast Zero-Shot TabPFN Inference (Runs if rows <= 10,000)
        if len(engineered_df) <= 10000:
            try:
                y_tr_series = train_df[self.target_column] if has_target else None
                tabpfn_res = self.tabpfn.fit_predict(X_train, y_tr_series, X_holdout)
                # Calculate metrics on holdout
                if y_holdout is not None:
                    from sklearn.metrics import roc_auc_score, f1_score
                    p = np.array(tabpfn_res["anomaly_scores"])
                    fl = np.array(tabpfn_res["flags"])
                    try:
                        tabpfn_res["metrics"] = {
                            "ROC-AUC": round(float(roc_auc_score(y_holdout, p)), 4),
                            "F1-Score": round(float(f1_score(y_holdout, fl, zero_division=0)), 4)
                        }
                    except Exception:
                        pass
                candidate_results["TabPFN_ZeroShot"] = tabpfn_res
            except Exception as e:
                candidate_results["TabPFN_ZeroShot"] = {"error": str(e), "latency_ms": 0}

        # 4. Decision Engine Routing
        routing = self.decision_engine.evaluate_routing(
            df=engineered_df,
            eda_report=eda_report,
            is_streaming_stream=stream_mode
        )

        # 5. Core Execution (Pathway A or Pathway B)
        if routing["pathway"] == "Adaptive_Baseline":
            baseline_res = self.adaptive_baseline.fit_predict(
                df=holdout_df,
                target_col=self.target_column
            )
            candidate_results["Adaptive_Baseline"] = baseline_res
        else:
            ensemble_res = self.ensemble.fit_predict(
                X_train_df=X_train,
                y_train=y_train if y_train is not None else np.zeros(len(X_train)),
                X_test_df=X_holdout,
                y_test=y_holdout
            )
            candidate_results["AutoGluon_Ensemble"] = ensemble_res

        # 6. Automated A/B Pipeline Champion Selection
        champion_name = "AutoGluon_Ensemble"
        if routing["pathway"] == "Adaptive_Baseline":
            champion_name = "Adaptive_Baseline"
        elif "AutoGluon_Ensemble" in candidate_results:
            champion_name = "AutoGluon_Ensemble"
        elif "TabPFN_ZeroShot" in candidate_results and "error" not in candidate_results["TabPFN_ZeroShot"]:
            champion_name = "TabPFN_ZeroShot"

        # 7. Unified Model Summary Packaging
        dataset_meta = {
            "total_rows": int(len(engineered_df)),
            "train_rows": int(len(train_df)),
            "holdout_rows": int(len(holdout_df)),
            "features_count": int(X_train.shape[1]),
            "feature_names": list(X_train.columns),
            "target_column": self.target_column if has_target else None
        }

        unified_summary = self.summarizer.build_summary(
            run_id=run_id,
            champion_name=champion_name,
            cv_strategy_info=cv_info,
            decision_routing=routing,
            candidate_results=candidate_results,
            dataset_meta=dataset_meta
        )

        duration = round(time.time() - start_time, 4)

        return {
            "status": "COMPLETED",
            "duration_seconds": duration,
            "champion_model": champion_name,
            "cv_strategy": cv_info,
            "routing": routing,
            "unified_model_summary": unified_summary,
            "candidate_results": candidate_results,
            "holdout_predictions": {
                "indices": holdout_df.index.tolist(),
                "actual": y_holdout.tolist() if y_holdout is not None else [],
                "scores": candidate_results[champion_name].get("anomaly_scores", []),
                "flags": candidate_results[champion_name].get("flags", [])
            }
        }
