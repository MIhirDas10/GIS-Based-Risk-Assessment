"""
tests/test_predict_outputs.py

Verify predict.py's outputs (GeoJSON files, Redis cache, prediction_log)
are present and well-formed after the most recent run.
"""
import json
from pathlib import Path

import pytest

# Every test here reads predict.py's on-disk output, Redis, or the database.
pytestmark = pytest.mark.integration

GEOJSON_DIR = Path("/opt/airflow/data/processed")
EXPECTED_FILES = [
    "risk_current.geojson",
    "forecast_1w.geojson",
    "forecast_2w.geojson",
    "forecast_3w.geojson",
    "forecast_4w.geojson",
]


@pytest.mark.parametrize("filename", EXPECTED_FILES)
def test_geojson_file_exists(filename):
    path = GEOJSON_DIR / filename
    assert path.exists(), f"Missing {path}"
    assert path.stat().st_size > 1000, f"{filename} suspiciously small"


@pytest.mark.parametrize("filename", EXPECTED_FILES)
def test_geojson_is_valid_featurecollection(filename):
    with open(GEOJSON_DIR / filename) as f:
        data = json.load(f)
    assert data["type"] == "FeatureCollection"
    assert "features" in data
    assert len(data["features"]) >= 1


@pytest.mark.parametrize("filename", EXPECTED_FILES)
def test_geojson_features_have_required_properties(filename):
    """Every feature must carry the props the dashboard expects."""
    required = {
        "district_id", "district_name", "year", "week",
        "predicted_cases", "risk_score", "risk_tier", "risk_color",
        "population_density", "hotspot_rank",
    }
    with open(GEOJSON_DIR / filename) as f:
        data = json.load(f)
    for feat in data["features"]:
        missing = required - set(feat["properties"].keys())
        assert not missing, f"{filename} feature missing: {missing}"
        # Sanity: risk_tier must be one of the 4 valid values
        assert feat["properties"]["risk_tier"] in {"Low", "Moderate", "High", "Critical"}


@pytest.mark.parametrize("filename", EXPECTED_FILES)
def test_geojson_geometry_is_multipolygon(filename):
    with open(GEOJSON_DIR / filename) as f:
        data = json.load(f)
    for feat in data["features"]:
        assert feat["geometry"]["type"] in {"MultiPolygon", "Polygon"}


# ---------------------------------------------------------------------------
# Redis cache
# ---------------------------------------------------------------------------

def test_redis_current_risk_cached(redis_client):
    val = redis_client.get("dengue:risk:current")
    assert val is not None, "predict.py should cache current risk in Redis"
    parsed = json.loads(val)
    assert parsed["type"] == "FeatureCollection"


def test_redis_forecast_keys_present(redis_client):
    keys = list(redis_client.scan_iter(match="dengue:forecast:*"))
    assert len(keys) >= 4, f"Expected ≥4 forecast cache keys, found {keys}"


# ---------------------------------------------------------------------------
# prediction_log DB table
# ---------------------------------------------------------------------------

def test_prediction_log_populated(pg_conn):
    with pg_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM monitoring.prediction_log")
        n = cur.fetchone()[0]
    assert n >= 8, f"Expected ≥8 predictions logged, got {n}"


def test_prediction_log_model_version_set(pg_conn):
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT model_version FROM monitoring.prediction_log WHERE model_version IS NOT NULL"
        )
        versions = [r[0] for r in cur.fetchall()]
    assert len(versions) >= 1
    assert any("dengue_risk_model" in v for v in versions)
