"""
tests/test_api.py

Exercise the FastAPI service end-to-end with the in-process TestClient.
The client talks to the real DB / Redis / MLflow that the test fixtures
already provide; no need to spin up a separate uvicorn process.
"""
from __future__ import annotations

import pytest

# The TestClient boots the real app, which needs the database, Redis and MLflow.
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    """Bring up the FastAPI app in-process. Lifespan runs (loads model,
    caches districts) before the first request."""
    from fastapi.testclient import TestClient

    # Import here so test collection doesn't fail if the api package
    # is somehow missing; the import error surfaces as a clean test fail.
    from api.main import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    for key in ("status", "db_ok", "redis_ok", "mlflow_ok", "uptime_s"):
        assert key in body


def test_health_db_redis_mlflow_all_up(client):
    """In the test environment all 3 deps should be reachable."""
    body = client.get("/health").json()
    assert body["db_ok"] is True
    assert body["redis_ok"] is True
    assert body["mlflow_ok"] is True
    assert body["status"] == "ok"


def test_health_reports_model_version(client):
    body = client.get("/health").json()
    assert body["model_version"], "Production model should be registered"
    assert "dengue_risk_model" in body["model_version"]


def test_health_reports_data_freshness(client):
    body = client.get("/health").json()
    # Format is YYYY-WNN
    assert body["data_freshness"] is not None
    assert "W" in body["data_freshness"]


# ---------------------------------------------------------------------------
# /districts and /district/{id}
# ---------------------------------------------------------------------------

def test_list_districts_returns_64(client):
    r = client.get("/districts")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 64
    for d in body:
        assert "district_id" in d
        assert "district_name" in d
        # Centroid should be inside Bangladesh BBOX
        assert 20.0 < d["centroid_lat"] < 27.0
        assert 87.5 < d["centroid_lon"] < 93.0


def test_district_detail_dhaka_has_history(client):
    # Find Dhaka's id from the list endpoint
    districts = client.get("/districts").json()
    dhaka = next(d for d in districts if d["district_name"] == "Dhaka")
    r = client.get(f"/district/{dhaka['district_id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["district"]["district_name"] == "Dhaka"
    assert len(body["history"]) > 0, "Dhaka should have weekly history rows"


def test_district_detail_missing_returns_404(client):
    r = client.get("/district/9999")
    assert r.status_code == 404


def test_district_detail_invalid_id_returns_422(client):
    """Pydantic validation rejects ids outside 1-1000."""
    r = client.get("/district/0")
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /risk/current and /risk/forecast/{weeks}
# ---------------------------------------------------------------------------

def test_risk_current_returns_geojson(client):
    r = client.get("/risk/current")
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "FeatureCollection"
    assert "features" in body
    assert len(body["features"]) >= 1


def test_risk_current_features_have_required_properties(client):
    body = client.get("/risk/current").json()
    feat = body["features"][0]
    assert feat["geometry"]["type"] in {"MultiPolygon", "Polygon"}
    required = {
        "district_id", "district_name", "year", "week",
        "predicted_cases", "risk_score", "risk_tier", "risk_color",
    }
    assert required.issubset(feat["properties"].keys())


@pytest.mark.parametrize("weeks", [1, 2, 3, 4])
def test_risk_forecast_returns_geojson(client, weeks):
    r = client.get(f"/risk/forecast/{weeks}")
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "FeatureCollection"
    assert len(body["features"]) >= 1


@pytest.mark.parametrize("weeks", [0, 5, 99])
def test_risk_forecast_out_of_range_returns_422(client, weeks):
    """Pydantic constrains weeks to 1-4."""
    r = client.get(f"/risk/forecast/{weeks}")
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /metrics/model
# ---------------------------------------------------------------------------

def test_model_metrics_returns_production_info(client):
    r = client.get("/metrics/model")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "dengue_risk_model"
    assert body["stage"] == "Production"
    # The Production model's RMSE should beat the seasonal naive baseline (1349)
    assert body["rmse"] is not None
    assert body["rmse"] < 1349


# ---------------------------------------------------------------------------
# /
# ---------------------------------------------------------------------------

def test_root_redirects_to_docs(client):
    # follow_redirects=False so we can see the 307
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"].endswith("/docs")


def test_docs_endpoint_renders(client):
    r = client.get("/docs")
    assert r.status_code == 200
    assert "swagger" in r.text.lower() or "openapi" in r.text.lower()
