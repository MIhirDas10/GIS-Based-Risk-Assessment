"""
src/models/predict.py

Loads the Production model from MLflow registry, generates dengue case
predictions for current/future weeks, computes risk scores and tiers,
outputs weekly GeoJSON for the dashboard map, and writes results to
monitoring.prediction_log for drift tracking.

Risk scoring formula
--------------------
    risk_score = predicted_cases_per_100k
                 × log(population_density + 1)
                 × hotspot_weight

    Where hotspot_weight = 1 + hotspot_rank  (range 1.0 – 2.0)

Risk tiers (percentile-based across all districts):
    Low:       0–25th percentile
    Moderate: 25–75th percentile
    High:     75–90th percentile
    Critical: >90th percentile

Outputs
-------
1. monitoring.prediction_log     — DB table for drift monitoring
2. data/processed/risk_current.geojson — current week risk map
3. data/processed/forecast_{n}w.geojson — 1–4 week forecast maps
4. Redis cache (via API)          — keyed by week for fast dashboard loads

Usage (standalone):
    python predict.py                       # predict current week
    python predict.py --weeks-ahead 4       # forecast 1–4 weeks ahead
    python predict.py --model-stage Staging  # use Staging model instead

Via Airflow:
    Called weekly by the orchestration DAG after feature mart rebuild.
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

import mlflow
from db import get_db_conn, get_sqlalchemy_engine  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("predict")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_REGISTRY_NAME = "dengue_risk_model"
DEFAULT_MODEL_STAGE = "Production"

OUTPUT_DIR = Path(os.getenv("PREDICTION_OUTPUT_DIR", "data/processed"))

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_TTL  = 60 * 60 * 24 * 7  # 7 days

# Features expected by the model (must match train.py FEATURE_COLUMNS)
FEATURE_COLUMNS = [
    "temp_mean_c", "temp_max_c", "rainfall_mm", "humidity_pct",
    "rainfall_lag_2w", "rainfall_lag_4w", "temp_lag_2w", "humidity_lag_2w",
    "temp_rolling_4w", "rainfall_rolling_4w",
    "humidity_x_temp", "rainfall_x_density",
    "cases_spatial_lag",
    "week_sin", "week_cos", "month",
    "population_density", "hotspot_rank",
]

# Risk tier boundaries (percentile-based)
RISK_TIERS = {
    "Low":      (0,   25),
    "Moderate": (25,  75),
    "High":     (75,  90),
    "Critical": (90, 100),
}


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

# Database helpers come from src/db.py (get_db_conn, get_sqlalchemy_engine)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(stage: str = DEFAULT_MODEL_STAGE):
    """
    Load the registered dengue model from MLflow.
    Falls back to Staging if Production is not available,
    then falls back to latest run if registry is empty.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Try registry first
    try:
        model_uri = f"models:/{MODEL_REGISTRY_NAME}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)
        log.info("Loaded model from registry: %s (stage=%s)", MODEL_REGISTRY_NAME, stage)
        return model, stage
    except Exception as e:
        log.warning("Could not load %s model: %s", stage, e)

    # Fallback: try Staging if Production failed
    if stage == "Production":
        try:
            model_uri = f"models:/{MODEL_REGISTRY_NAME}/Staging"
            model = mlflow.pyfunc.load_model(model_uri)
            log.info("Fell back to Staging model")
            return model, "Staging"
        except Exception as e:
            log.warning("Could not load Staging model either: %s", e)

    # Last resort: load the best run from the experiment
    log.warning("No registered model found — loading best run from experiment")
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("dengue_risk_prediction")
    if experiment is None:
        raise RuntimeError(
            "No MLflow experiment found. Run train.py first."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.model_type IN ('xgboost', 'lightgbm')",
        order_by=["metrics.rmse ASC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No model runs found in MLflow. Run train.py first.")

    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)
    log.info(
        "Loaded best run: %s (RMSE=%.2f)",
        best_run.info.run_id,
        best_run.data.metrics.get("rmse", -1),
    )
    return model, "latest_run"


# ---------------------------------------------------------------------------
# Feature loading — current and forecast weeks
# ---------------------------------------------------------------------------

def load_current_features(source_table: str = "features.forecast_mart") -> pd.DataFrame:
    """
    Load the most recent week's features.

    Defaults to ``features.forecast_mart`` (built from real current ERA5 +
    last-year spatial-lag stand-in) so dashboard predictions reflect the
    actual current week, not the last week we have disease ground truth for.
    Pass ``source_table='features.mart'`` to use the training mart instead,
    which is useful for back-testing against held-out actuals.

    Uses the SQLAlchemy engine (not a raw psycopg2 conn) so pandas
    doesn't emit its "only supports SQLAlchemy connectable" warning.
    """
    if source_table not in {"features.mart", "features.forecast_mart"}:
        raise ValueError(f"source_table must be one of mart/forecast_mart, got {source_table}")
    # Table name interpolated as a literal — both choices are hard-coded
    # whitelisted above, so this is safe from injection.
    sql = f"""
        SELECT
            m.*,
            d.district_name,
            d.geometry IS NOT NULL AS has_geometry
        FROM {source_table} m
        JOIN geo.districts d ON d.district_id = m.district_id
        WHERE (m.year, m.week) = (
            SELECT year, week
            FROM {source_table}
            ORDER BY year DESC, week DESC
            LIMIT 1
        )
        ORDER BY m.district_id;
    """
    df = pd.read_sql(sql, get_sqlalchemy_engine())
    log.info(
        "Loaded current features: %d districts, year=%d week=%d",
        len(df),
        df["year"].iloc[0] if len(df) > 0 else 0,
        df["week"].iloc[0] if len(df) > 0 else 0,
    )
    return df


def load_features_for_week(year: int, week: int) -> pd.DataFrame:
    """Load features for a specific (year, week)."""
    sql = """
        SELECT
            m.*,
            d.district_name
        FROM features.mart m
        JOIN geo.districts d ON d.district_id = m.district_id
        WHERE m.year = %(year)s AND m.week = %(week)s
        ORDER BY m.district_id;
    """
    return pd.read_sql(sql, get_sqlalchemy_engine(), params={"year": year, "week": week})


def load_historical_features(n_weeks: int = 52) -> pd.DataFrame:
    """Load the last N weeks of features (for forecast context)."""
    sql = """
        SELECT
            m.*,
            d.district_name
        FROM features.mart m
        JOIN geo.districts d ON d.district_id = m.district_id
        ORDER BY m.year DESC, m.week DESC
        LIMIT %(limit)s;
    """
    return pd.read_sql(
        sql, get_sqlalchemy_engine(), params={"limit": n_weeks * 64},  # 64 districts
    )


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------

def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute risk score and assign risk tiers.

    Risk score formula:
        predicted_cases_per_100k × log(pop_density + 1) × hotspot_weight

    This gives higher risk to:
      - Districts with more predicted cases (obviously)
      - Dense urban areas (more transmission potential)
      - Historically high-risk districts (structural vulnerability)
    """
    df = df.copy()

    # Predicted cases per 100k (normalize by population)
    # If population_density is 0, use raw predicted cases
    df["predicted_cases_per_100k"] = np.where(
        df["population_density"] > 0,
        df["predicted_cases"] / (df["population_density"] / 100000 + 1),
        df["predicted_cases"],
    )

    # Hotspot weight: 1.0 (lowest risk district) to 2.0 (highest)
    df["hotspot_weight"] = 1 + df["hotspot_rank"]

    # Composite risk score
    df["risk_score"] = (
        df["predicted_cases_per_100k"]
        * np.log1p(df["population_density"])
        * df["hotspot_weight"]
    )

    # Clip negative predictions (model can predict < 0)
    df["risk_score"] = df["risk_score"].clip(lower=0)

    # Assign risk tiers based on percentile of risk_score
    if len(df) > 1:
        percentiles = df["risk_score"].rank(pct=True) * 100
        conditions = [
            percentiles <= 25,
            percentiles <= 75,
            percentiles <= 90,
            percentiles > 90,
        ]
        tier_labels = ["Low", "Moderate", "High", "Critical"]
        df["risk_tier"] = np.select(conditions, tier_labels, default="Moderate")
    else:
        df["risk_tier"] = "Moderate"

    # Risk color for mapping (green → red)
    tier_colors = {
        "Low":      "#22c55e",  # green
        "Moderate": "#eab308",  # yellow
        "High":     "#f97316",  # orange
        "Critical": "#ef4444",  # red
    }
    df["risk_color"] = df["risk_tier"].map(tier_colors)

    log.info("Risk tier distribution:")
    for tier in tier_labels:
        count = (df["risk_tier"] == tier).sum()
        log.info("  %-10s: %d districts", tier, count)

    return df


# ---------------------------------------------------------------------------
# Confidence intervals (bootstrap-based)
# ---------------------------------------------------------------------------

def compute_confidence_intervals(
    model,
    X: np.ndarray,
    n_bootstrap: int = 100,
    ci_level: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate confidence intervals using feature noise injection.

    Since we don't have a proper ensemble or Bayesian model, we add
    small Gaussian noise to the features and observe prediction variance.
    This gives a rough uncertainty estimate.

    Returns (lower_bound, upper_bound) arrays.
    """
    predictions = []
    for _ in range(n_bootstrap):
        # Add 5% Gaussian noise to features
        noise = np.random.normal(1.0, 0.05, size=X.shape)
        X_noisy = X * noise
        pred = model.predict(pd.DataFrame(X_noisy, columns=FEATURE_COLUMNS))
        predictions.append(pred)

    predictions = np.array(predictions)
    alpha = (1 - ci_level) / 2

    lower = np.clip(np.percentile(predictions, alpha * 100, axis=0), 0, None)
    upper = np.clip(np.percentile(predictions, (1 - alpha) * 100, axis=0), 0, None)

    return lower, upper


# ---------------------------------------------------------------------------
# GeoJSON output
# ---------------------------------------------------------------------------

def load_district_geometries(conn) -> dict:
    """
    Load district geometries as GeoJSON from PostGIS.
    Returns {district_id: geojson_geometry_dict}.
    """
    sql = """
        SELECT
            district_id,
            district_name,
            ST_AsGeoJSON(geometry)::json AS geojson
        FROM geo.districts
        WHERE geometry IS NOT NULL;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    return {
        row[0]: {"name": row[1], "geometry": row[2]}
        for row in rows
    }


def build_geojson(
    df: pd.DataFrame,
    geometries: dict,
    title: str = "Dengue Risk Map",
) -> dict:
    """
    Build a GeoJSON FeatureCollection from predictions + district polygons.

    Each feature has properties:
        district_name, predicted_cases, risk_score, risk_tier, risk_color,
        population_density, hotspot_rank, year, week,
        confidence_lower, confidence_upper (if available)
    """
    features = []

    for _, row in df.iterrows():
        did = int(row["district_id"])
        geo_info = geometries.get(did)

        if geo_info is None:
            log.warning("No geometry for district_id=%d — skipping GeoJSON feature", did)
            continue

        properties = {
            "district_id":        did,
            "district_name":      row.get("district_name", geo_info["name"]),
            "year":               int(row["year"]),
            "week":               int(row["week"]),
            "predicted_cases":    round(float(row["predicted_cases"]), 1),
            "risk_score":         round(float(row["risk_score"]), 2),
            "risk_tier":          row["risk_tier"],
            "risk_color":         row["risk_color"],
            "population_density": round(float(row.get("population_density", 0)), 1),
            "hotspot_rank":       round(float(row.get("hotspot_rank", 0)), 3),
        }

        # Add confidence intervals if available
        if "ci_lower" in row:
            properties["confidence_lower"] = round(float(row["ci_lower"]), 1)
            properties["confidence_upper"] = round(float(row["ci_upper"]), 1)

        # Add actual cases if available (for monitoring)
        if "dengue_cases" in row and pd.notna(row["dengue_cases"]):
            properties["actual_cases"] = int(row["dengue_cases"])

        features.append({
            "type": "Feature",
            "geometry": geo_info["geometry"],
            "properties": properties,
        })

    geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "title": title,
            "generated_at": datetime.utcnow().isoformat(),
            "n_districts": len(features),
        },
        "features": features,
    }

    return geojson


def save_geojson(geojson: dict, filename: str) -> Path:
    """Save GeoJSON to the output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    log.info("Saved GeoJSON: %s (%d features)", path, len(geojson["features"]))
    return path


# ---------------------------------------------------------------------------
# Database logging (prediction_log)
# ---------------------------------------------------------------------------

def log_predictions_to_db(conn, df: pd.DataFrame, model_version: str):
    """
    Write predictions to monitoring.prediction_log for drift tracking.
    Actual cases will be backfilled later when they become available.
    """
    rows = []
    for _, row in df.iterrows():
        rows.append((
            int(row["district_id"]),
            int(row["year"]),
            int(row["week"]),
            round(float(row["predicted_cases"]), 2),
            None,   # actual_cases — filled later by monitor DAG
            row["risk_tier"],
            model_version,
        ))

    sql = """
        INSERT INTO monitoring.prediction_log
            (district_id, year, week, predicted_cases,
             actual_cases, risk_tier, model_version)
        VALUES %s
        ON CONFLICT (district_id, year, week)
        DO UPDATE SET
            predicted_cases = EXCLUDED.predicted_cases,
            risk_tier       = EXCLUDED.risk_tier,
            model_version   = EXCLUDED.model_version,
            predicted_at    = NOW();
    """

    # The UNIQUE(district_id, year, week) constraint lives in init_db.sql
    # (see uq_prediction_log_district_year_week). Don't redefine it here.
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=200)
    conn.commit()

    log.info("Logged %d predictions to monitoring.prediction_log", len(rows))


# ---------------------------------------------------------------------------
# Redis caching
# ---------------------------------------------------------------------------

def cache_to_redis(geojson: dict, cache_key: str):
    """Cache GeoJSON in Redis for fast dashboard loads."""
    try:
        import redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.setex(cache_key, REDIS_TTL, json.dumps(geojson))
        log.info("Cached to Redis: key=%s (TTL=%ds)", cache_key, REDIS_TTL)
    except ImportError:
        log.warning("redis package not installed — skipping cache")
    except Exception as e:
        log.warning("Redis cache failed: %s — dashboard will query DB directly", e)


# ---------------------------------------------------------------------------
# Forecast (multi-week ahead)
# ---------------------------------------------------------------------------

def generate_forecast(
    model,
    conn,
    df_current: pd.DataFrame,
    weeks_ahead: int,
    geometries: dict,
    model_version: str,
) -> list[dict]:
    """
    Generate forecasts for 1 to weeks_ahead weeks into the future.

    For week N+1, we use the most recent available features.
    For week N+2+, we shift cyclical features and keep weather features
    from the most recent week (persistence assumption for weather).

    Returns list of GeoJSON dicts, one per forecast week.
    """
    log.info("Generating %d-week forecast …", weeks_ahead)

    forecast_geojsons = []

    for w in range(1, weeks_ahead + 1):
        log.info("  Forecast week +%d …", w)

        df_forecast = df_current.copy()

        # Advance the week counter
        current_year = int(df_forecast["year"].iloc[0])
        current_week = int(df_forecast["week"].iloc[0])

        forecast_week = current_week + w
        forecast_year = current_year
        if forecast_week > 52:
            forecast_week -= 52
            forecast_year += 1

        df_forecast["year"] = forecast_year
        df_forecast["week"] = forecast_week
        df_forecast["month"] = min(12, max(1, (forecast_week * 12) // 52 + 1))

        # Update cyclical features
        df_forecast["week_sin"] = np.sin(2 * np.pi * forecast_week / 52.0)
        df_forecast["week_cos"] = np.cos(2 * np.pi * forecast_week / 52.0)

        # Shift lag features forward (current rainfall becomes lag_2w in 2 weeks)
        if w >= 2:
            df_forecast["rainfall_lag_2w"] = df_current["rainfall_mm"]
            df_forecast["temp_lag_2w"]     = df_current["temp_mean_c"]
            df_forecast["humidity_lag_2w"] = df_current["humidity_pct"]
        if w >= 4:
            df_forecast["rainfall_lag_4w"] = df_current["rainfall_mm"]

        # Predict
        X_forecast = df_forecast[FEATURE_COLUMNS].fillna(0).values
        predictions = model.predict(
            pd.DataFrame(X_forecast, columns=FEATURE_COLUMNS)
        )
        df_forecast["predicted_cases"] = np.clip(predictions, 0, None)

        # Confidence intervals
        ci_lower, ci_upper = compute_confidence_intervals(model, X_forecast)
        df_forecast["ci_lower"] = ci_lower
        df_forecast["ci_upper"] = ci_upper

        # Risk scoring
        df_forecast = compute_risk_scores(df_forecast)

        # Build GeoJSON
        title = f"Dengue Risk Forecast — Week {forecast_year}-W{forecast_week:02d} (+{w}w)"
        geojson = build_geojson(df_forecast, geometries, title=title)
        save_geojson(geojson, f"forecast_{w}w.geojson")
        forecast_geojsons.append(geojson)

        # Cache
        cache_key = f"dengue:forecast:{forecast_year}:{forecast_week}"
        cache_to_redis(geojson, cache_key)

    return forecast_geojsons


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    weeks_ahead: int = 4,
    model_stage: str = DEFAULT_MODEL_STAGE,
    source_table: str = "features.forecast_mart",
):
    """
    Full prediction pipeline:
    1. Load model from MLflow
    2. Load current week features
    3. Predict current week
    4. Compute risk scores + tiers
    5. Generate GeoJSON + cache
    6. Generate multi-week forecast
    7. Log predictions to DB
    """
    log.info("=" * 60)
    log.info("DENGUE PREDICTION PIPELINE — STARTING")
    log.info("=" * 60)

    # 1. Load model
    model, actual_stage = load_model(model_stage)
    model_version = f"{MODEL_REGISTRY_NAME}/{actual_stage}"

    # 2. Load features (read-only path uses SQLAlchemy engine internally);
    # keep a separate psycopg2 conn for the write paths below.
    log.info("  Source table: %s", source_table)
    df_current = load_current_features(source_table=source_table)
    conn = get_db_conn()
    try:

        if len(df_current) == 0:
            log.error(
                "No features found for the most recent week. "
                "Run build_features.py first."
            )
            return

        current_year = int(df_current["year"].iloc[0])
        current_week = int(df_current["week"].iloc[0])
        log.info("Current week: %d-W%02d (%d districts)",
                 current_year, current_week, len(df_current))

        # 3. Predict current week
        X_current = df_current[FEATURE_COLUMNS].fillna(0).values
        predictions = model.predict(
            pd.DataFrame(X_current, columns=FEATURE_COLUMNS)
        )
        df_current["predicted_cases"] = np.clip(predictions, 0, None)

        # Confidence intervals
        ci_lower, ci_upper = compute_confidence_intervals(model, X_current)
        df_current["ci_lower"] = ci_lower
        df_current["ci_upper"] = ci_upper

        # 4. Risk scoring
        df_current = compute_risk_scores(df_current)

        # Summary
        log.info("-" * 50)
        log.info("CURRENT WEEK PREDICTIONS (%d-W%02d)", current_year, current_week)
        log.info("-" * 50)
        log.info("  %-20s %8s %8s %10s",
                 "District", "Pred", "CI", "Risk")
        log.info("  " + "-" * 50)
        for _, row in df_current.sort_values("risk_score", ascending=False).iterrows():
            log.info(
                "  %-20s %8.1f  [%4.1f–%4.1f]  %-10s",
                row.get("district_name", f"ID-{row['district_id']}"),
                row["predicted_cases"],
                row.get("ci_lower", 0),
                row.get("ci_upper", 0),
                row["risk_tier"],
            )

        # 5. GeoJSON for current week
        geometries = load_district_geometries(conn)

        current_geojson = build_geojson(
            df_current, geometries,
            title=f"Dengue Risk Map — {current_year}-W{current_week:02d}",
        )
        save_geojson(current_geojson, "risk_current.geojson")
        cache_to_redis(current_geojson, "dengue:risk:current")

        # 6. Multi-week forecast
        if weeks_ahead > 0:
            generate_forecast(
                model, conn, df_current, weeks_ahead,
                geometries, model_version,
            )

        # 7. Log to DB
        log_predictions_to_db(conn, df_current, model_version)

        log.info("=" * 60)
        log.info("DENGUE PREDICTION PIPELINE — COMPLETE")
        log.info("  Current week: %d-W%02d", current_year, current_week)
        log.info("  Forecasts: +%d weeks", weeks_ahead)
        log.info("  GeoJSON files: %s/", OUTPUT_DIR)
        log.info("=" * 60)

    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dengue risk predictions")
    parser.add_argument(
        "--weeks-ahead", type=int, default=4,
        help="Number of weeks to forecast ahead (default: 4)",
    )
    parser.add_argument(
        "--model-stage", type=str, default=DEFAULT_MODEL_STAGE,
        choices=["Production", "Staging"],
        help=f"MLflow model stage to use (default: {DEFAULT_MODEL_STAGE})",
    )
    parser.add_argument(
        "--use-training-mart", action="store_true",
        help=(
            "Predict against features.mart (training/back-test data, ends "
            "at 2023-W52) instead of features.forecast_mart (current data "
            "from real ERA5). Useful for back-testing."
        ),
    )
    args = parser.parse_args()

    run(
        weeks_ahead=args.weeks_ahead,
        model_stage=args.model_stage,
        source_table="features.mart" if args.use_training_mart else "features.forecast_mart",
    )
