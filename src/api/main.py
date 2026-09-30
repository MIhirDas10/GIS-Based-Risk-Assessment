"""
src/api/main.py

FastAPI service that serves dengue risk predictions to the dashboard and
external clients. Read-only; mutations (ingest, retrain) live in Airflow.

Strategy
--------
1. Redis is the fast path. `predict.py` populates `dengue:risk:current` and
   `dengue:forecast:{year}:{week}` after each run; the API returns those
   verbatim when the cache is warm.
2. Postgres is the source of truth fallback (cold start, TTL expiry,
   ad-hoc per-district drill-down).
3. The MLflow Production model is loaded once at startup and cached in
   memory; `/metrics/model` reports the run metrics.

All endpoints are GET, idempotent, and safe to call repeatedly.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import redis
from fastapi import FastAPI, HTTPException
from fastapi import Path as PathParam
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text

import mlflow

# Make the shared db helpers importable regardless of which container we're
# running in. Both paths are no-ops if the dir doesn't exist.
sys.path.insert(0, "/opt/airflow/src")                              # airflow container
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # API container (= /app)

from db import get_sqlalchemy_engine  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("api")


# ---------------------------------------------------------------------------
# Config (env-driven, with sensible defaults)
# ---------------------------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_REGISTRY_NAME = "dengue_risk_model"
MODEL_STAGE = "Production"

# Cache keys produced by predict.py
REDIS_KEY_CURRENT = "dengue:risk:current"
PREDICTION_OUTPUT_DIR = Path(os.getenv("PREDICTION_OUTPUT_DIR", "/opt/airflow/data/processed"))


# ---------------------------------------------------------------------------
# In-memory state, populated by the lifespan handler
# ---------------------------------------------------------------------------
class AppState:
    started_at: float = 0.0
    model: object | None = None
    model_version: str | None = None
    model_metrics: dict = {}
    districts: list[dict] = []   # cached district list (rare query, small)


state = AppState()


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    db_ok: bool
    redis_ok: bool
    mlflow_ok: bool
    model_version: str | None = None
    data_freshness: str | None = Field(
        None, description="ISO 8601 year-week of the most recent feature mart row"
    )
    uptime_s: float


class DistrictRef(BaseModel):
    district_id: int
    district_name: str
    division_name: str | None = None
    area_km2: float | None = None
    centroid_lat: float | None = None
    centroid_lon: float | None = None


class HistoryPoint(BaseModel):
    year: int
    week: int
    dengue_cases: int
    temp_mean_c: float | None = None
    rainfall_mm: float | None = None


class DistrictDetail(BaseModel):
    district: DistrictRef
    latest_prediction: dict | None = None
    history: list[HistoryPoint]


class ModelMetrics(BaseModel):
    name: str
    version: str
    stage: str
    rmse: float | None = None
    mae: float | None = None
    r2: float | None = None
    outbreak_f1: float | None = None


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

def _redis():
    """Cached Redis client. Returns None if Redis is unreachable so callers
    can fall back to the DB without crashing the request."""
    if not hasattr(_redis, "_client"):
        try:
            _redis._client = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, decode_responses=True,
                socket_connect_timeout=2,
            )
        except Exception as e:
            log.warning("Redis unreachable: %s", e)
            _redis._client = None
    return _redis._client


def _engine():
    """Shared SQLAlchemy engine."""
    return get_sqlalchemy_engine()


def _check_redis() -> bool:
    try:
        c = _redis()
        return c is not None and c.ping()
    except Exception:
        return False


def _check_db() -> bool:
    try:
        with _engine().connect() as conn:
            return conn.execute(text("SELECT 1")).scalar() == 1
    except Exception:
        return False


def _check_mlflow() -> bool:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        client.get_registered_model(MODEL_REGISTRY_NAME)
        return True
    except Exception:
        return False


def _load_model_metadata():
    """Populate state.model / state.model_version / state.model_metrics
    from MLflow. Safe to call multiple times."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=[MODEL_STAGE])
    except Exception as e:
        log.warning("Could not list MLflow versions: %s", e)
        return

    if not versions:
        log.warning("No %s version of %s registered yet", MODEL_STAGE, MODEL_REGISTRY_NAME)
        return

    v = versions[0]
    state.model_version = f"{MODEL_REGISTRY_NAME}/v{v.version}/{MODEL_STAGE}"

    # Pull metrics from the run that produced this model version
    try:
        run = client.get_run(v.run_id)
        m = run.data.metrics
        state.model_metrics = {
            "name": MODEL_REGISTRY_NAME,
            "version": str(v.version),
            "stage": MODEL_STAGE,
            "rmse": m.get("rmse"),
            "mae": m.get("mae"),
            "r2": m.get("r2"),
            "outbreak_f1": m.get("outbreak_f1"),
        }
    except Exception as e:
        log.warning("Could not fetch metrics for run %s: %s", v.run_id, e)

    # Eager-load the actual model object so first prediction request is fast
    try:
        state.model = mlflow.pyfunc.load_model(
            f"models:/{MODEL_REGISTRY_NAME}/{MODEL_STAGE}"
        )
        log.info("Loaded %s into memory", state.model_version)
    except Exception as e:
        log.warning("Could not load model into memory: %s", e)


def _load_districts():
    """Pull the (small, static) district list into memory once at startup."""
    try:
        df = pd.read_sql(
            text("""
                SELECT
                    district_id, district_name, division_name, area_km2,
                    ST_Y(ST_Centroid(geometry)) AS centroid_lat,
                    ST_X(ST_Centroid(geometry)) AS centroid_lon
                FROM geo.districts
                ORDER BY district_name
            """),
            _engine(),
        )
        state.districts = df.to_dict(orient="records")
        log.info("Cached %d districts", len(state.districts))
    except Exception as e:
        log.warning("Could not preload districts: %s", e)


# ---------------------------------------------------------------------------
# Lifespan: connect to deps, load model, cache districts
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    state.started_at = time.time()
    log.info("Starting Dengue API...")
    log.info("  REDIS_HOST=%s MLFLOW_TRACKING_URI=%s", REDIS_HOST, MLFLOW_TRACKING_URI)

    _load_districts()
    _load_model_metadata()

    log.info("Ready.")
    yield
    log.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Dengue Risk Prediction API",
    description=(
        "Read-only API serving weekly dengue case predictions and risk-tier "
        "maps for Bangladesh. Data refreshes weekly via Airflow; this service "
        "reads from Redis (warm cache) and Postgres (fallback)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    """Liveness + dependency check. Returns 200 even when deps are degraded,
    so a load balancer can distinguish 'API up' from 'API healthy'."""
    db_ok = _check_db()
    redis_ok = _check_redis()
    mlflow_ok = _check_mlflow()

    # Data freshness = ISO year/week of the latest mart row
    freshness = None
    if db_ok:
        try:
            with _engine().connect() as conn:
                row = conn.execute(text(
                    "SELECT year, week FROM features.mart ORDER BY year DESC, week DESC LIMIT 1"
                )).fetchone()
                if row:
                    freshness = f"{row[0]}-W{row[1]:02d}"
        except Exception:
            pass

    overall = "ok" if (db_ok and redis_ok and mlflow_ok) else "degraded"
    return HealthResponse(
        status=overall,
        db_ok=db_ok,
        redis_ok=redis_ok,
        mlflow_ok=mlflow_ok,
        model_version=state.model_version,
        data_freshness=freshness,
        uptime_s=round(time.time() - state.started_at, 1),
    )


# ---------------------------------------------------------------------------
# /districts and /district/{id}
# ---------------------------------------------------------------------------
@app.get("/districts", response_model=list[DistrictRef], tags=["districts"])
def list_districts():
    """List all 64 districts with centroid coordinates."""
    if not state.districts:
        _load_districts()
    return state.districts


@app.get("/district/{district_id}", response_model=DistrictDetail, tags=["districts"])
def district_detail(
    district_id: int = PathParam(..., ge=1, description="Internal district id"),
    weeks_history: int = 12,
):
    """Per-district drill-down: latest prediction, last N weeks of actuals + weather."""
    if not state.districts:
        _load_districts()

    matching = [d for d in state.districts if d["district_id"] == district_id]
    if not matching:
        raise HTTPException(status_code=404, detail=f"district_id {district_id} not found")
    district = DistrictRef(**matching[0])

    # Latest prediction (if any)
    latest_pred = None
    try:
        with _engine().connect() as conn:
            row = conn.execute(text("""
                SELECT year, week, predicted_cases, risk_tier, model_version, predicted_at
                FROM monitoring.prediction_log
                WHERE district_id = :did
                ORDER BY year DESC, week DESC, predicted_at DESC
                LIMIT 1
            """), {"did": district_id}).mappings().first()
            if row:
                latest_pred = {
                    "year": int(row["year"]),
                    "week": int(row["week"]),
                    "predicted_cases": float(row["predicted_cases"]),
                    "risk_tier": row["risk_tier"],
                    "model_version": row["model_version"],
                    "predicted_at": row["predicted_at"].isoformat() if row["predicted_at"] else None,
                }
    except Exception as e:
        log.warning("prediction_log lookup failed: %s", e)

    # History
    history: list[HistoryPoint] = []
    try:
        df = pd.read_sql(
            text("""
                SELECT year, week, dengue_cases, temp_mean_c, rainfall_mm
                FROM features.mart
                WHERE district_id = :did
                ORDER BY year DESC, week DESC
                LIMIT :n
            """),
            _engine(),
            params={"did": district_id, "n": weeks_history},
        )
        history = [HistoryPoint(**row) for row in df.to_dict(orient="records")]
        history.reverse()  # chronological
    except Exception as e:
        log.warning("history lookup failed: %s", e)

    return DistrictDetail(district=district, latest_prediction=latest_pred, history=history)


# ---------------------------------------------------------------------------
# /risk/current and /risk/forecast/{weeks}
# ---------------------------------------------------------------------------
def _geojson_from_redis(key: str) -> dict | None:
    c = _redis()
    if c is None:
        return None
    try:
        raw = c.get(key)
        return json.loads(raw) if raw else None
    except Exception as e:
        log.warning("Redis read failed (%s): %s", key, e)
        return None


def _geojson_from_disk(filename: str) -> dict | None:
    """Cold-start fallback: read the file predict.py wrote to disk."""
    path = PREDICTION_OUTPUT_DIR / filename
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Disk read failed (%s): %s", path, e)
        return None


def _current_geojson_from_db() -> dict | None:
    """Build a current-week GeoJSON response from prediction_log and PostGIS."""
    try:
        df = pd.read_sql(
            text("""
                WITH latest_week AS (
                    SELECT year, week
                    FROM monitoring.prediction_log
                    ORDER BY year DESC, week DESC
                    LIMIT 1
                )
                SELECT
                    p.district_id,
                    d.district_name,
                    p.year,
                    p.week,
                    p.predicted_cases,
                    p.risk_tier,
                    COALESCE(m.population_density, 0) AS population_density,
                    COALESCE(m.hotspot_rank, 0) AS hotspot_rank,
                    ST_AsGeoJSON(d.geometry)::json AS geometry
                FROM monitoring.prediction_log p
                JOIN latest_week lw
                  ON lw.year = p.year AND lw.week = p.week
                JOIN geo.districts d
                  ON d.district_id = p.district_id
                LEFT JOIN features.mart m
                  ON m.district_id = p.district_id
                 AND m.year = p.year
                 AND m.week = p.week
                WHERE d.geometry IS NOT NULL
                ORDER BY p.district_id
            """),
            _engine(),
        )
    except Exception as e:
        log.warning("DB GeoJSON fallback failed: %s", e)
        return None

    if df.empty:
        return None

    tier_colors = {
        "Low": "#22c55e",
        "Moderate": "#eab308",
        "High": "#f97316",
        "Critical": "#ef4444",
    }

    features = []
    for row in df.to_dict(orient="records"):
        population_density = float(row["population_density"] or 0)
        hotspot_weight = 1 + float(row["hotspot_rank"] or 0)
        predicted_cases = float(row["predicted_cases"] or 0)
        if population_density > 0:
            predicted_per_100k = predicted_cases / (population_density / 100000 + 1)
        else:
            predicted_per_100k = predicted_cases
        risk_score = predicted_per_100k * max(1.0, hotspot_weight)

        features.append({
            "type": "Feature",
            "geometry": row["geometry"],
            "properties": {
                "district_id": int(row["district_id"]),
                "district_name": row["district_name"],
                "year": int(row["year"]),
                "week": int(row["week"]),
                "predicted_cases": round(predicted_cases, 1),
                "risk_score": round(risk_score, 2),
                "risk_tier": row["risk_tier"],
                "risk_color": tier_colors.get(row["risk_tier"], "#eab308"),
                "population_density": round(population_density, 1),
                "hotspot_rank": round(float(row["hotspot_rank"] or 0), 3),
            },
        })

    year = int(df["year"].iloc[0])
    week = int(df["week"].iloc[0])
    return {
        "type": "FeatureCollection",
        "metadata": {
            "title": f"Dengue Risk Map - {year}-W{week:02d}",
            "source": "postgres",
            "n_districts": len(features),
        },
        "features": features,
    }


def _forecast_geojson_from_redis(weeks: int) -> dict | None:
    c = _redis()
    if c is None:
        return None
    try:
        for key in c.scan_iter(match="dengue:forecast:*"):
            candidate = _geojson_from_redis(key)
            if candidate and candidate.get("metadata", {}).get("title", "").endswith(
                f"(+{weeks}w)"
            ):
                return candidate
    except Exception as e:
        log.warning("Redis scan failed: %s", e)
    return None


@app.get("/risk/current", tags=["risk"])
def risk_current():
    """Current-week risk map as a GeoJSON FeatureCollection.

    Source order: Redis (`dengue:risk:current`) -> disk fallback
    (`risk_current.geojson`) -> Postgres fallback. 404 if none has data;
    that means predict.py hasn't run yet."""
    geo = (
        _geojson_from_redis(REDIS_KEY_CURRENT)
        or _geojson_from_disk("risk_current.geojson")
        or _current_geojson_from_db()
    )
    if geo is None:
        raise HTTPException(
            status_code=404,
            detail="No current-week predictions available; run predict.py first.",
        )
    return geo


@app.get("/risk/forecast/{weeks}", tags=["risk"])
def risk_forecast(
    weeks: int = PathParam(..., ge=1, le=4, description="Weeks ahead, 1-4"),
):
    """N-week-ahead forecast (1-4 weeks). Same source order as /risk/current."""
    # Forecast Redis keys use target year/week, so scan for the metadata title
    # suffix that predict.py writes, then fall back to the deterministic file.
    geo = _forecast_geojson_from_redis(weeks) or _geojson_from_disk(f"forecast_{weeks}w.geojson")
    if geo is None:
        raise HTTPException(
            status_code=404,
            detail=f"No {weeks}-week forecast available; run predict.py --weeks-ahead {weeks} first.",
        )
    return geo


# ---------------------------------------------------------------------------
# /metrics/model
# ---------------------------------------------------------------------------
@app.get("/metrics/model", response_model=ModelMetrics, tags=["meta"])
def model_metrics():
    """Current Production model's metrics from MLflow."""
    if not state.model_metrics:
        _load_model_metadata()
    if not state.model_metrics:
        raise HTTPException(
            status_code=503,
            detail="No Production model registered yet; run train.py.",
        )
    return ModelMetrics(**state.model_metrics)


# ---------------------------------------------------------------------------
# Convenience: redirect root to docs
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")
