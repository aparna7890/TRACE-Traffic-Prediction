"""Small offline tests for the audit functions."""
import pandas as pd
from src.audit_data import normalize_columns, parse_timestamp

def test_normalize_expected_upstream_columns():
    df = pd.DataFrame(columns=["date", "time", "route_code", "duration", "distance"])
    normalized = normalize_columns(df)
    assert {"date", "time", "route_code", "duration", "distance"}.issubset(normalized.columns)

def test_parse_timestamp_date_and_time():
    df = pd.DataFrame({"date": ["2025-01-02"], "time": ["14:30"]})
    result = parse_timestamp(df)
    assert str(result.iloc[0]) == "2025-01-02 14:30:00"

def test_parse_timestamp_date_only():
    df = pd.DataFrame({"date": ["2025-01-02"]})
    result = parse_timestamp(df)
    assert str(result.iloc[0]) == "2025-01-02 00:00:00"
