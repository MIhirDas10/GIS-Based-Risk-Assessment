"""
tests/test_dashboard.py

Unit tests for the Streamlit dashboard's pure helpers. We don't try to
render the page — that needs a running Streamlit server. Instead we
exercise the data-transformation functions, which is where bugs usually
live.

Streamlit is imported as a side effect (it logs a "No runtime found"
warning at import time without a running app — harmless).
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

# Make `dashboard.app` importable
sys.path.insert(0, "/opt/airflow")
os.environ.setdefault("API_URL", "http://api:8000")

# Import lazily so test collection doesn't fail when streamlit is missing
dashboard_app = pytest.importorskip("dashboard.app", reason="streamlit not installed")


# ---------------------------------------------------------------------------
# format_number
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value, digits, expected", [
    (None, 1, "-"),
    (0, 0, "0"),
    (123.456, 1, "123.5"),
    (1234567, 0, "1,234,567"),
    (1234.5, 2, "1,234.50"),
    ("not a number", 1, "not a number"),
])
def test_format_number(value, digits, expected):
    assert dashboard_app.format_number(value, digits) == expected


# ---------------------------------------------------------------------------
# properties_frame
# ---------------------------------------------------------------------------

def test_properties_frame_empty():
    df = dashboard_app.properties_frame({"features": []})
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_properties_frame_missing_features_key():
    df = dashboard_app.properties_frame({})
    assert df.empty


def test_properties_frame_categorical_risk_tier():
    geojson = {
        "features": [
            {"properties": {"district_name": "Dhaka", "risk_tier": "Critical"}},
            {"properties": {"district_name": "Sylhet", "risk_tier": "Low"}},
            {"properties": {"district_name": "Khulna", "risk_tier": "High"}},
        ]
    }
    df = dashboard_app.properties_frame(geojson)
    assert len(df) == 3
    # risk_tier should be a Categorical in the canonical order
    assert df["risk_tier"].cat.categories.tolist() == dashboard_app.RISK_ORDER


def test_properties_frame_skips_features_without_properties():
    geojson = {
        "features": [
            {"properties": {"district_name": "A", "risk_tier": "Low"}},
            {"geometry": {"type": "Point", "coordinates": [0, 0]}},  # no properties
            {"properties": None},
            {"properties": {"district_name": "B", "risk_tier": "High"}},
        ]
    }
    df = dashboard_app.properties_frame(geojson)
    assert len(df) == 2
    assert set(df["district_name"]) == {"A", "B"}


# ---------------------------------------------------------------------------
# latest_week_label
# ---------------------------------------------------------------------------

def test_latest_week_label_returns_dash_when_empty():
    assert dashboard_app.latest_week_label({"features": []}) == "-"


def test_latest_week_label_picks_max_year_week():
    geojson = {
        "features": [
            {"properties": {"year": 2023, "week": 5, "risk_tier": "Low"}},
            {"properties": {"year": 2023, "week": 52, "risk_tier": "Critical"}},
            {"properties": {"year": 2022, "week": 50, "risk_tier": "High"}},
        ]
    }
    assert dashboard_app.latest_week_label(geojson) == "2023-W52"


def test_latest_week_label_pads_single_digit_week():
    geojson = {"features": [{"properties": {"year": 2024, "week": 3, "risk_tier": "Low"}}]}
    assert dashboard_app.latest_week_label(geojson) == "2024-W03"


# ---------------------------------------------------------------------------
# top_risk_table
# ---------------------------------------------------------------------------

def test_top_risk_table_empty_passthrough():
    df = pd.DataFrame()
    assert dashboard_app.top_risk_table(df).empty


def test_top_risk_table_sorts_and_renames():
    df = pd.DataFrame([
        {"district_name": "A", "risk_tier": "Low", "predicted_cases": 10, "risk_score": 1.0},
        {"district_name": "B", "risk_tier": "Critical", "predicted_cases": 500, "risk_score": 50.0},
        {"district_name": "C", "risk_tier": "High", "predicted_cases": 200, "risk_score": 20.0},
    ])
    top = dashboard_app.top_risk_table(df, limit=2)
    assert list(top.columns) == ["District", "Risk", "Predicted", "Score"]
    # Sorted by risk_score descending
    assert top.iloc[0]["District"] == "B"
    assert top.iloc[1]["District"] == "C"
    assert len(top) == 2


def test_top_risk_table_includes_ci_when_present():
    df = pd.DataFrame([
        {
            "district_name": "Dhaka", "risk_tier": "Critical",
            "predicted_cases": 500, "risk_score": 50.0,
            "confidence_lower": 200, "confidence_upper": 700,
        }
    ])
    top = dashboard_app.top_risk_table(df)
    assert "CI low" in top.columns
    assert "CI high" in top.columns


# ---------------------------------------------------------------------------
# RISK_COLORS / RISK_ORDER consistency
# ---------------------------------------------------------------------------

def test_risk_colors_cover_all_risk_tiers():
    assert set(dashboard_app.RISK_COLORS.keys()) == set(dashboard_app.RISK_ORDER)


def test_risk_colors_are_hex():
    for color in dashboard_app.RISK_COLORS.values():
        assert color.startswith("#") and len(color) == 7


# ---------------------------------------------------------------------------
# safe_api_get — wraps errors
# ---------------------------------------------------------------------------

def test_safe_api_get_returns_data_on_success(monkeypatch):
    def fake_api_get(_path):
        return {"status": "ok"}
    monkeypatch.setattr(dashboard_app, "api_get", fake_api_get)
    data, err = dashboard_app.safe_api_get("/health")
    assert data == {"status": "ok"}
    assert err is None


def test_safe_api_get_returns_error_on_request_exception(monkeypatch):
    import requests

    def raises(_path):
        raise requests.ConnectionError("boom")
    monkeypatch.setattr(dashboard_app, "api_get", raises)
    data, err = dashboard_app.safe_api_get("/health")
    assert data is None
    assert "boom" in err


# ---------------------------------------------------------------------------
# status_pill — small HTML formatter
# ---------------------------------------------------------------------------

def test_status_pill_ok_class():
    html = dashboard_app.status_pill("db", True)
    assert 'class="status-dot active"' in html
    assert ">db: Active<" in html


def test_status_pill_warn_class():
    html = dashboard_app.status_pill("redis", False)
    assert 'class="status-dot inactive"' in html
    assert ">redis: Offline<" in html
