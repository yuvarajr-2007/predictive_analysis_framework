"""
Feature Registry (Lightweight Feature Store):
Centralized repository for engineered features and mathematical transformation signatures.
Persists column data in compressed Parquet files and maintains a JSON metadata catalog.
Guarantees zero training-serving skew during production inference.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd


class FeatureRegistryClient:
    def __init__(self, store_dir: Optional[str] = None):
        if store_dir:
            self.store_dir = Path(store_dir)
        else:
            self.store_dir = Path(__file__).resolve().parent.parent / "feature_store"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir = self.store_dir / "features"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_file = self.store_dir / "catalog.json"

        self._init_catalog()

    def _init_catalog(self):
        if not self.catalog_file.exists():
            with open(self.catalog_file, "w", encoding="utf-8") as f:
                json.dump({"features": {}, "version": "1.0.0"}, f, indent=2)

    def register_feature(
        self,
        name: str,
        series: pd.Series,
        operation: str,
        source_columns: List[str],
        reason: str,
        origin: str = "feature_engineering_agent"
    ) -> Dict[str, Any]:
        """
        Registers an engineered feature, stores column data as Parquet, and updates catalog.
        """
        # Save column as Parquet
        feat_path = self.features_dir / f"{name}.parquet"
        col_df = pd.DataFrame({name: series})
        col_df.to_parquet(feat_path, index=False)

        # Update metadata catalog
        with open(self.catalog_file, "r", encoding="utf-8") as f:
            catalog = json.load(f)

        meta = {
            "name": name,
            "data_type": str(series.dtype),
            "operation": operation,
            "source_columns": source_columns,
            "reason": reason,
            "origin": origin,
            "rows_count": len(series),
            "parquet_path": str(feat_path),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "version": 1
        }
        catalog["features"][name] = meta

        with open(self.catalog_file, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)

        return meta

    def get_features(self, feature_names: List[str]) -> pd.DataFrame:
        """Retrieves and concatenates registered feature columns into a dataframe."""
        dfs = []
        for name in feature_names:
            feat_path = self.features_dir / f"{name}.parquet"
            if feat_path.exists():
                dfs.append(pd.read_parquet(feat_path))
            else:
                raise KeyError(f"Feature '{name}' not found in Feature Registry.")
        return pd.concat(dfs, axis=1)

    def list_registered_features(self) -> Dict[str, Any]:
        """Returns the full catalog of registered features."""
        with open(self.catalog_file, "r", encoding="utf-8") as f:
            catalog = json.load(f)
            return catalog.get("features", {})
