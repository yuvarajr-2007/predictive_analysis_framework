import pytest
import pandas as pd
from layer1_validation.schema_validator import SchemaValidator
from layer1_validation.null_validator import NullValidator
from layer1_validation.domain_validator import DomainValidator
from layer1_validation.validation_engine import ValidationEngine


def test_schema_validator_clean():
    df = pd.DataFrame({"timestamp": ["2026-01-01"], "machine_id": ["M1"], "voltage": [220.0], "anomaly": [0]})
    config = {"required_columns": ["machine_id", "voltage", "anomaly"]}
    res = SchemaValidator().validate(df, config)
    assert res["status"] == "PASS"
    assert res["total_issues"] == 0


def test_schema_validator_missing_column():
    df = pd.DataFrame({"machine_id": ["M1"]})
    config = {"required_columns": ["machine_id", "voltage"]}
    res = SchemaValidator().validate(df, config)
    assert res["status"] == "FAIL"
    assert res["total_issues"] == 1
    assert res["issues"][0]["type"] == "missing_required_column"


def test_null_validator_threshold():
    # 2 out of 10 rows null = 20% > 5% threshold
    data = [1.0] * 8 + [None, None]
    df = pd.DataFrame({"val": data})
    res = NullValidator().validate(df, {"missing_threshold": 0.05})
    assert res["status"] == "WARNING"
    assert res["total_issues"] == 1


def test_domain_validator_physical_negative():
    df = pd.DataFrame({"voltage": [-15.0, 220.0]})
    config = {
        "physical_constraints": [{"column": "voltage", "rule": "non_negative"}]
    }
    res = DomainValidator().validate(df, config)
    assert res["status"] == "FAIL"
    assert res["issues"][0]["type"] == "physical_constraint_violation"


def test_validation_engine_end_to_end():
    engine = ValidationEngine()
    df = pd.read_csv("d:/predictive_framework/data/sample_high_voltage_power.csv")
    res = engine.run(df, domain_name="high_voltage_power")
    assert res["overall_status"] == "PASS"
    assert res["dataset"]["rows"] == len(df)
