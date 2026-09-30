"""
Quick sanity-test of dashboard pure helpers — no Streamlit context needed.
Run inside the airflow container which has streamlit/folium/plotly installed:

    docker exec -w /opt/airflow -e API_URL=http://api:8000 \\
      dengue_airflow_scheduler python scripts/smoke_dashboard.py
"""
import os
import sys

# Make `dashboard.app` importable
sys.path.insert(0, "/opt/airflow")
os.environ.setdefault("API_URL", "http://api:8000")

# Import without triggering Streamlit page render
import streamlit as _streamlit  # noqa: F401

from dashboard.app import (
    RISK_COLORS,
    RISK_ORDER,
    format_number,
    latest_week_label,
    properties_frame,
    top_risk_table,
)

# 1. format_number
assert format_number(None) == "-"
assert format_number(123.456) == "123.5"
assert format_number(1234567, 0) == "1,234,567"
assert format_number("not a number") == "not a number"
print("[ok] format_number")

# 2. properties_frame — empty in, empty out
import pandas as pd

empty = properties_frame({"features": []})
assert isinstance(empty, pd.DataFrame) and empty.empty
print("[ok] properties_frame (empty)")

# 3. properties_frame with realistic features
gjson = {
    "features": [
        {"properties": {"district_name": "Dhaka", "risk_tier": "Critical",
                        "predicted_cases": 386.5, "risk_score": 12.3,
                        "year": 2023, "week": 52}},
        {"properties": {"district_name": "Sylhet", "risk_tier": "Low",
                        "predicted_cases": 91.0, "risk_score": 1.2,
                        "year": 2023, "week": 52}},
    ]
}
df = properties_frame(gjson)
assert len(df) == 2
assert df["risk_tier"].cat.categories.tolist() == RISK_ORDER
print("[ok] properties_frame (real)")

# 4. latest_week_label
assert latest_week_label(gjson) == "2023-W52"
assert latest_week_label({"features": []}) == "-"
print("[ok] latest_week_label")

# 5. top_risk_table — sorts by risk_score desc and renames columns
top = top_risk_table(df)
assert list(top.columns) == ["District", "Risk", "Predicted", "Score"]
assert top.iloc[0]["District"] == "Dhaka"
print("[ok] top_risk_table")

# 6. RISK_COLORS shape
assert set(RISK_COLORS.keys()) == set(RISK_ORDER)
print("[ok] RISK_COLORS")

print("\nAll dashboard helper smoke tests pass.")
