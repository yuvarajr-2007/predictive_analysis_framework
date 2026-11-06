"""
Validation Engine: The central coordinator for Layer 1.
Runs all validators, enforces PASS/WARN/FAIL policy, and produces the handover package.
"""
import time
import uuid
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

from .schema_validator import SchemaValidator
from .null_validator import NullValidator
from .domain_validator import DomainValidator


class ValidationEngine:
    def __init__(self, domain_dir: Optional[str] = None):
        if domain_dir:
            self.domain_dir = Path(domain_dir)
        else:
            self.domain_dir = Path(__file__).resolve().parent.parent / "domain_knowledge"
        
        self.validators = [
            SchemaValidator(),
            NullValidator(),
            DomainValidator()
        ]

    def load_config(self, domain_name: str) -> Dict[str, Any]:
        """Loads validation rules and physical constraints for a given domain."""
        merged_config: Dict[str, Any] = {"domain": domain_name, "version": "1.0.0"}

        # 1. Validation rules
        vrule_path = self.domain_dir / "validation_rules" / f"{domain_name}.yaml"
        if vrule_path.exists():
            with open(vrule_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f) or {}
                merged_config.update(content)

        # 2. Physical constraints
        pconst_path = self.domain_dir / "physical_constraints" / f"{domain_name}.yaml"
        if pconst_path.exists():
            with open(pconst_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f) or {}
                merged_config["physical_constraints"] = content.get("constraints", [])

        return merged_config

    def run(self, df: pd.DataFrame, domain_name: str = "high_voltage_power", user_config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        start_time = time.time()
        run_id = f"run-{uuid.uuid4().hex[:8]}"

        config = self.load_config(domain_name)
        if user_config_override:
            config.update(user_config_override)

        results: List[Dict[str, Any]] = []
        overall_status = "PASS"
        checks_run = []

        for validator in self.validators:
            v_name = validator.__class__.__name__
            checks_run.append(v_name)
            try:
                res = validator.validate(df, config)
                results.append(res)
                if res.get("status") == "FAIL":
                    overall_status = "FAIL"
                elif res.get("status") == "WARNING" and overall_status != "FAIL":
                    overall_status = "WARNING"
            except Exception as e:
                results.append({
                    "validator": v_name,
                    "status": "FAIL",
                    "total_issues": 1,
                    "issues": [{"type": "runtime_exception", "severity": "FAIL", "message": str(e)}]
                })
                overall_status = "FAIL"

        duration = round(time.time() - start_time, 4)

        total_issues_count = sum(r.get("total_issues", 0) for r in results)

        handover_package = {
            "run_id": run_id,
            "domain": domain_name,
            "dataset": {
                "rows": int(len(df)) if df is not None else 0,
                "columns": int(len(df.columns)) if df is not None else 0,
                "column_names": list(df.columns) if df is not None else []
            },
            "overall_status": overall_status,
            "total_issues": total_issues_count,
            "checks_run": checks_run,
            "results": results,
            "config_version": config.get("version", "1.0.0"),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
            "duration_seconds": duration,
            "validated_dataframe": df
        }
        return handover_package
