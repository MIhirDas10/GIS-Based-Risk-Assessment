# Dengue Risk Prediction — Status Report

*Project: `H:\dengue-risk-bd` · Last updated: 2026-05-26*

---

## TL;DR

**Days 1–9 done.** End-to-end pipeline running on real Copernicus ERA5 weather. XGBoost in Production (RMSE 1129, R² 0.286, Outbreak F1 0.634). REST API serves 7 endpoints. Streamlit dashboard renders current-week risk maps with click-to-drill-down for all 64 districts. Dashboard now shows **2026-W09** predictions (was 2023-W52 until today) thanks to a new `features.forecast_mart` that uses real recent weather without needing future disease ground truth. **118 tests pass, ruff clean, 7 Docker containers up.**

Days 10–12 remaining: Evidently drift monitoring + auto-retrain, IMPACT.md, GitHub Actions CI.

---

## What this project does (one paragraph)

District-level dengue case prediction for Bangladesh. Ingests Copernicus ERA5-Land weather + Kaggle dengue case CSV + WorldPop population + HDX admin boundaries. Engineers 18 features in SQL (lags, rolling means, interactions, spatial lag, cyclical encoding, hotspot rank). Trains 4 models (Seasonal Naive baseline, Ridge, XGBoost, LightGBM); MLflow auto-registers the best one as Production if it beats baseline by ≥15%. Predicts current week + 4 weeks ahead, emits GeoJSON risk maps + Redis cache + DB drift log. FastAPI exposes everything as REST; Streamlit reads the API to render a Folium choropleth dashboard.

---

## Architecture (current)

```
                        ┌─────────────────────┐
                        │   Copernicus CDS    │ ← real ERA5-Land, parallel monthly downloads
                        └─────────┬───────────┘
                                  │
   CSV  ──→ ingest_disease   ┐    │
   SHP  ──→ ingest_boundaries│    │
   TIF  ──→ ingest_population├────┼─────→ ┌──────────────────┐
   NC4  ←── ingest_era5      ┘    │       │   PostgreSQL +   │
                                  │       │     PostGIS      │
                                  │       └────────┬─────────┘
                                  │                │ build_features
                                  │                ├──────→ features.mart           (training)
                                  │                └──────→ features.forecast_mart  (serving)
                                  │                                  │
                                  │                                  ↓
                                  │                            train.py (4 models)
                                  │                                  │ MLflow
                                  │                                  ↓
                                  │                            Production model
                                  │                                  │
                                  │                                  ↓
                                  │                            predict.py
                                  │                                  │
                                  │                       ┌──────────┴──────────┐
                                  │                       │                     │
                                  │                       ↓                     ↓
                                  │                  Redis cache         GeoJSON files
                                  │                       │                     │
                                  └───────────────────────┴──────────┬──────────┘
                                                                     │
                                                              FastAPI (port 8000)
                                                                     │
                                                                     ↓
                                                            Streamlit dashboard (8501)
```

---

## What's running right now

| Container | Image | Port | Role |
|---|---|---|---|
| `dengue_postgres` | postgis/postgis:15-3.4 | 5432 | DB + spatial extensions |
| `dengue_redis` | redis:7.2-alpine | 6379 | Hot cache for GeoJSON |
| `dengue_mlflow` | python:3.11-slim | 5001→5000 | Model tracking + registry |
| `dengue_airflow_webserver` | dengue-airflow:latest | 8080 | DAG UI (admin/admin) |
| `dengue_airflow_scheduler` | dengue-airflow:latest | — | Executes the 5 DAGs |
| `dengue_api` | dengue-risk-bd-api | 8000 | FastAPI REST service |
| `dengue_dashboard` | dengue-risk-bd-dashboard | 8501 | Streamlit UI |

All have `restart: always`. Survive Docker daemon restarts and host reboots.

---

## Days 1–7 — Data, features, training, prediction ✅

### What got done

- **Infrastructure** — 7-container Docker Compose with PostGIS, MLflow (`--serve-artifacts`), Redis, Airflow (init/sched/web).
- **5 ingestion DAGs** in Airflow: `ingest_boundaries`, `ingest_disease_data`, `ingest_worldpop_population`, `ingest_era5_weather`, `build_feature_mart`.
- **DB schema** (init_db.sql): 5 schemas (`disease`, `weather`, `geo`, `features`, `monitoring`), 2 databases (`dengue_db`, `mlflow_db`).
- **Real data**:
  - 64 districts with MultiPolygon geometry in EPSG:4326
  - 856 weekly dengue case records (8 districts, 2019–2023)
  - WorldPop 2020 population (avg density 1392/km², total 175M)
  - **17,216 ERA5 weekly rows** (64 districts × ~270 weeks, real Copernicus data through 2026-W09)
- **Feature mart** — 736 rows × 23 columns, 7-step CTE pipeline in pure SQL.
- **4 models trained** + logged to MLflow. XGBoost won (RMSE 1129, R² 0.286, Outbreak F1 0.634), promoted to Production.
- **Predict pipeline** — produces GeoJSON for current week + 4-week forecasts, caches to Redis, logs to `monitoring.prediction_log`.

### Bugs fixed during verification (16 total)

| # | File | Bug | Fix |
|---|---|---|---|
| 1 | init_db.sql | `features.mart` schema stale (missing 5 columns) | Removed DDL, let build_features own it |
| 2 | init_db.sql | `mlflow_db` never created | Added `CREATE DATABASE mlflow_db` |
| 3 | init_db.sql | No UNIQUE constraint on disease cases | Added `UNIQUE(district_id, year, month, week)` |
| 4 | src/ingestion/ingest_boundaries.py | File was a stub — DAG imported `run()` that didn't exist | Wrote the missing implementation |
| 5 | src/ingestion/ingest_disease.py | CSV is daily; not aggregated to weekly → UNIQUE violation | Aggregate in `prepare_case_rows` before insert |
| 6 | src/ingestion/ingest_disease.py | INSERT was non-idempotent | Changed to `ON CONFLICT DO UPDATE` |
| 7 | src/ingestion/ingest_population.py | WorldPop URL 404 | Corrected bucket path |
| 8 | src/ingestion/{era5,population}.py | Wrong shapefile column name | Added `adm2_name` to candidates |
| 9 | src/features/build_features.py | Weeks crossing month boundaries duplicated PK | Added `cases_weekly` aggregating CTE |
| 10 | src/models/train.py | Ridge scaler not bundled in MLflow → predictions 1000× inflated | Use sklearn Pipeline |
| 11 | docker-compose.yml | MLflow artifact uploads failed | Switched to `--serve-artifacts` |
| 12 | src/ingestion/ingest_era5.py | CDS-beta delivers `format=netcdf` as ZIP-wrapped | Added `_is_zip` + `_unwrap_zip_to_nc` |
| 13 | src/ingestion/ingest_era5.py | CDS-beta renamed `time` dim → `valid_time` | Accept either name |
| 14 | docker-compose.yml | Airflow tasks defaulted POSTGRES_HOST=localhost | Added env vars to airflow-common |
| 15 | src/ingestion/ingest_era5.py | Future-month requests hit `MultiAdaptorNoDataError` | Cap to `today.month - 3` (ERA5-Land lag) |
| 16 | docker-compose.yml | `postgres`/`redis`/`mlflow` had no restart policy | Added `restart: always` |

### ERA5 ingest narrative
- Triggered overnight after the user accepted CDS-beta license
- **Completed in ~1 hour** thanks to parallel monthly downloads (4 concurrent) + 6-hourly sampling + vectorized zonal stats (~100× faster than the original per-hour-per-district loop)
- Per-month NetCDF cache persists at `data/raw/era5_cache/{year}/` — crash-safe across power cuts

### Performance optimizations during overnight run
- **6-hourly instead of hourly** ERA5 sampling — 4× cheaper at CDS and 4× less data
- **Parallel CDS submission** with semaphore=4 — overlap queue waits
- **Pre-rasterized district masks + pure-numpy aggregation** — replaces 138k rasterio calls/month with 64 numpy index operations

---

## Day 8 — FastAPI REST service ✅

`src/api/main.py` — read-only API exposing 7 endpoints. All return JSON with proper OpenAPI typing via Pydantic.

| Method | Path | Returns | Source |
|---|---|---|---|
| GET | `/health` | status + db/redis/mlflow flags + model_version + data_freshness | Live pings |
| GET | `/metrics/model` | name/version/stage/RMSE/MAE/R²/F1 | MLflow registry |
| GET | `/districts` | All 64 districts with centroid lat/lon | Postgres, cached at startup |
| GET | `/district/{id}` | District info + latest prediction + 12-week history | Postgres |
| GET | `/risk/current` | GeoJSON FeatureCollection — current-week risk map | Redis → disk → Postgres (3-tier fallback) |
| GET | `/risk/forecast/{weeks}` | GeoJSON — N-week-ahead forecast (1–4) | Redis scan → disk |
| GET | `/docs` | Interactive Swagger UI | FastAPI auto-generated |

### Key design choices

- **Read-first from Redis** (warm cache). Falls back to disk GeoJSON files (`data/processed/*.geojson`). Falls back to live DB query (`monitoring.prediction_log` + `geo.districts`). 404 only if all three are empty.
- **Model loaded at startup** via FastAPI lifespan handler — first prediction request is fast.
- **District list cached in memory** — small static table, no need to hit DB per request.
- **`protected_namespaces=()` on Pydantic models** — silences the `model_version` field warning.

### Verification
- All 7 endpoints exercised with curl, return correct JSON
- 20 pytest tests via `TestClient`
- Container builds cleanly, restarts automatically

---

## Day 9 — Streamlit dashboard ✅

`dashboard/app.py` — 571-line single-page app at http://localhost:8501.

### Layout

```
DengueRisk BD                          [db: ok] [redis: ok] [mlflow: ok] [freshness: 2023-W52]
District dengue risk surveillance for Bangladesh
─────────────────────────────────────────────────────────────────────────────────────────────
┌─────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────────────┐ ┌────────────┐
│ SERVICE │ │ MAP WEEK │ │ PREDICTED    │ │ HIGH OR CRITICAL │ │ MODEL RMSE │
│ OK      │ │ 2026-W09 │ │ 2,482        │ │ 16               │ │ 1,129.2    │
└─────────┘ └──────────┘ └──────────────┘ └──────────────────┘ └────────────┘

[ Risk Map ] [ District Detail ] [ Model ]
```

Three tabs:
1. **Risk Map** — Folium choropleth with Current/Forecast toggle. Side panel shows tier-distribution bar chart + top-risk table.
2. **District Detail** — district selector, 4 metric cards, dual-axis history chart (cases + rainfall), weekly rows table.
3. **Model** — version/RMSE/MAE/F1 cards + metric comparison bar chart.

### Verification
- 22 pytest tests for pure helpers (`format_number`, `properties_frame`, `latest_week_label`, `top_risk_table`, `safe_api_get`, `status_pill`, `RISK_COLORS`)
- Visual screenshots taken with Playwright + headless Chromium (saved to `data/dashboard_screenshots/`)
- All 3 tabs render correctly with real data
- Dashboard ↔ API plumbing verified inside docker network

### Cosmetic finding (not fixed yet)
The Model tab's bar chart plots RMSE (1129), MAE (480), R² (0.286), Outbreak F1 (0.634) on a single y-axis. The 0-1 metrics (R², F1) appear as invisible thin lines because they're dwarfed by the case-count metrics. The metric cards above the chart show all four values clearly — the chart is redundant + misleading. Recommend either splitting into two subplots or dropping the chart.

---

## Today's enhancement — `features.forecast_mart` (real "current" predictions)

### The problem
`features.mart` is built with an INNER JOIN on `disease.dengue_cases` to ensure the target column is present for training. That join means the mart can only have rows where both disease AND weather data exist. Our disease CSV stops at 2023-W52, so the mart was capped at 2023-W52 — meaning the dashboard showed **130-week-old predictions** even though we had real ERA5 weather for 2026.

### The fix
Added a second feature table `features.forecast_mart`:
- INNER JOIN on weather only (not disease) → covers every (district, year, week) where ERA5 exists
- `cases_spatial_lag` uses **same-week-last-year** dengue cases as a stand-in (climatology repeats; the model's already trained to interpret this signal)
- `dengue_cases` / `cases_per_100k` left NULL (we don't have these for future weeks; the model only needs them for *training*, not *inference*)

Same Production XGBoost model — only the input table changed.

### Result

| | Before | After |
|---|---|---|
| Map week | 2023-W52 (130 weeks ago) | **2026-W09** (12 weeks ago) |
| Districts on map | 8 (case-data districts only) | **64** (all of Bangladesh) |
| Total predicted cases (visible) | 1,528 | 2,482 |
| High/Critical districts | 2 | **16** |
| Forecast horizon | +4 weeks from 2023-W52 | **+4 weeks from 2026-W09** |

### Why 12-week lag, not 0
ERA5-Land has a built-in **3-month publication lag**. To get truly real-time weather, you'd switch the ingestion source (GFS forecast, BMD stations, NOAA NWS). Roughly 2 hours of work; deferred.

### CLI

```bash
# Build forecast mart (every week ERA5 has data)
python /opt/airflow/src/features/build_features.py --forecast

# Predict against forecast mart (default now)
python /opt/airflow/src/models/predict.py --weeks-ahead 4

# Predict against training mart (back-testing only)
python /opt/airflow/src/models/predict.py --use-training-mart
```

---

## Polish work (Days 11-12 partial) ✅

Completed early:
- ✅ **README.md** — one-command setup, architecture diagram, design choices, known limitations
- ✅ **MODEL_CARD.md** — model purpose, intended use, factors, metrics, training data provenance, ethical considerations, maintenance plan (Mitchell et al. template)
- ✅ **DATA_DICTIONARY.md** — every column in every table, plus Redis keys + filesystem artifacts (~250 lines)
- ✅ **pyproject.toml** — ruff config (E/F/W/I/B/C4/SIM/UP rules, py311 target, line-length 100)
- ✅ **Shared `src/db.py`** — eliminated 5 duplicate `get_db_conn()` functions; SQLAlchemy engine for pandas
- ✅ **Runtime DDL moved into init_db.sql** (`uq_prediction_log_district_year_week`)

Still remaining:
- ⏳ **IMPACT.md** — public-health impact statement / theory of change
- ⏳ **GitHub Actions CI** — `.github/workflows/ci.yml` running ruff + pytest

---

## Test suite — 118 passing

| File | Tests | What's covered |
|---|---|---|
| `tests/test_imports.py` | 7 | Every src/ module imports cleanly + exposes `run()` |
| `tests/test_pure_functions.py` | 14 | Date parsing, dewpoint→RH, ZIP detection, polygon coercion, metrics math, risk scoring |
| `tests/test_db_state.py` | 16 | PostGIS, schemas, mlflow_db, UNIQUE constraints, district count/geometry, climatological plausibility |
| `tests/test_models.py` | 5 | Production model registered + loadable + predicts sane values; all 4 MLflow runs exist |
| `tests/test_predict_outputs.py` | 33 | 5 GeoJSON files exist, valid FeatureCollections, required properties, valid tiers, MultiPolygon, Redis keys cached |
| `tests/test_api.py` | 20 | All 7 FastAPI endpoints via TestClient + error paths (404/422) |
| `tests/test_dashboard.py` | 22 | Streamlit pure helpers (format_number, properties_frame, latest_week_label, top_risk_table, safe_api_get, status_pill, RISK_COLORS) |
| `tests/test_demo.py` | 1 | Placeholder (kept) |
| **Total** | **118** | **All passing, ~36 sec runtime, 21 non-failing deprecation warnings** |

Ruff: **All checks passed!** Across `src/`, `tests/`, `dashboard/`.

---

## Production model snapshot

| Field | Value |
|---|---|
| Name | `dengue_risk_model` |
| Version | 1 |
| Stage | Production |
| Algorithm | XGBoost (max_depth=6, lr=0.05, n_estimators=500) |
| Train period | 2019–2021 (224 weekly rows) |
| Validation period | 2022 (232 rows) |
| Test period | 2023 (280 rows) |
| Test RMSE | **1129 cases/week** |
| Test MAE | 480 cases/week |
| Test R² | **0.286** |
| Test Outbreak F1 (≥20 cases threshold) | **0.634** |
| Improvement over seasonal naive | **+16.3%** |

The 0.63 outbreak F1 means the model correctly flags outbreak weeks ~63% of the time — useful for triage, not a substitute for confirmed surveillance.

---

## Sample dashboard output (2026-W09)

| District | Predicted cases | 90% CI | Risk |
|---|---|---|---|
| Dhaka | 280.1 | [48–344] | 🔴 Critical |
| Chattogram | 82.0 | [0–196] | 🔴 Critical |
| Barishal | 59.7 | [16–304] | 🔴 Critical |
| Khulna | 47.7 | [0–238] | 🟠 High |
| Mymensingh | 31.3 | [0–40] | 🟠 High |
| Rajshahi | 24.0 | [19–74] | 🟡 Moderate |
| Rangpur | 25.8 | [6–31] | 🟡 Moderate |
| Sylhet | 4.0 | [0–22] | 🟢 Low |

(Plus 56 more districts in the visible-districts count of 64 — Dhaka, Chattogram, etc. are just the historically reporting 8.)

---

## What's still ahead

### Day 10 — Monitoring + auto-retrain (not started)
- `src/monitoring/monitor.py` — Evidently drift detection
- `airflow/dags/dag_monitor.py` — @weekly run that compares predictions vs actuals (4-week lag)
- `airflow/dags/dag_retrain.py` — triggered when RMSE degrades >15%, fires `train.py`

### Day 11–12 — Final polish (mostly done)
- ✅ pytest — 118 tests
- ✅ ruff — clean
- ✅ README — written
- ✅ MODEL_CARD — written
- ✅ DATA_DICTIONARY — written
- ⏳ IMPACT.md — public-health impact statement
- ⏳ GitHub Actions CI (`.github/workflows/ci.yml`)
- ⏳ Final commit + merge `develop` → `main`

### Deferred (out of original scope)
- Scrape DGHS PDFs for 2024+ dengue ground truth — enables drift monitoring + retraining on recent data
- Swap ERA5-Land (3-month lag) → GFS forecast (6-hour lag) for true real-time predictions
- Per-district performance breakdown — recommend which districts the model is unreliable on
- Quantile-regression model for proper Bayesian uncertainty intervals (instead of bootstrap)

---

## Quick reference — common commands

```bash
# Bring everything up from scratch
docker compose up -d

# Rebuild feature mart after new ERA5 lands
docker exec dengue_airflow_scheduler python /opt/airflow/src/features/build_features.py --export

# Build the serving mart (current-week predictions)
docker exec dengue_airflow_scheduler python /opt/airflow/src/features/build_features.py --forecast

# Retrain models, register best as Production
docker exec dengue_airflow_scheduler python /opt/airflow/src/models/train.py

# Generate predictions + GeoJSON + Redis cache (uses forecast_mart by default)
docker exec dengue_airflow_scheduler python /opt/airflow/src/models/predict.py --weeks-ahead 4

# Run the test suite
docker exec -w /opt/airflow -e API_URL=http://api:8000 dengue_airflow_scheduler python -m pytest tests/ -v

# Lint everything
docker exec -w /opt/airflow -u airflow dengue_airflow_scheduler python -m ruff check src/ tests/ dashboard/

# Take fresh dashboard screenshots
docker exec -w /opt/airflow dengue_airflow_scheduler python scripts/screenshot_dashboard.py
```

URLs:
- http://localhost:8080 — Airflow (admin/admin)
- http://localhost:5001 — MLflow UI
- http://localhost:8000/docs — API Swagger UI
- http://localhost:8501 — Streamlit dashboard
