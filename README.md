# Dengue Risk Prediction — Bangladesh

District-level dengue case prediction for Bangladesh using ERA5 weather, WorldPop population, and historical case data. Trains 4 models (Seasonal Naive baseline, Ridge, XGBoost, LightGBM), promotes the best to MLflow Production, and produces weekly risk-tier GeoJSON maps + 4-week forecasts.

| Layer | Tech | Port |
|---|---|---|
| Orchestration | Apache Airflow 2.8 (LocalExecutor) | `:8080` |
| Data warehouse | PostgreSQL 15 + PostGIS 3.4 | `:5432` |
| Model tracking | MLflow 2.10 (proxy-artifacts mode) | `:5001` |
| Cache | Redis 7.2 | `:6379` |
| REST API | FastAPI 0.109 + uvicorn | `:8000` |
| Dashboard | Streamlit + Folium + Plotly | `:8501` |

---

## Quickstart (one command, ~10 minutes)

**Prerequisites:** Docker Desktop running, ≈ 6 GB free disk, internet for ERA5/WorldPop downloads. On Windows you also need to keep "Sleep" disabled while the ingest runs.

```bash
# Bring everything up — postgres bootstraps the schema, mlflow starts the
# tracking server, airflow-init creates the admin user, scheduler+webserver
# pick up the 5 DAGs.
docker compose up -d
```

Then visit:
- **Airflow** http://localhost:8080 — login `admin` / `admin`
- **MLflow** http://localhost:5001
- **API docs** http://localhost:8000/docs — interactive Swagger UI
- **Dashboard** http://localhost:8501

### Populate the DB end-to-end

```bash
# 1. Districts (64 polygons from bgd_admin2.shp)
docker exec dengue_airflow_scheduler python /opt/airflow/src/ingestion/ingest_boundaries.py

# 2. Disease cases (4,776 daily rows → 856 weekly rows)
docker exec dengue_airflow_scheduler python /opt/airflow/src/ingestion/ingest_disease.py

# 3. Population (WorldPop 2020, ≈10 MB download)
docker exec dengue_airflow_scheduler python /opt/airflow/src/ingestion/ingest_population.py --year 2020

# 4. Weather — REAL ERA5 (5 years × 12 months parallel from Copernicus).
#    Requires a Copernicus CDS account + accepted ERA5-Land licence; put
#    your PAT in .cdsapirc at the project root. Takes ~1 hr first time.
docker exec dengue_airflow_scheduler airflow dags unpause ingest_era5_weather
docker exec dengue_airflow_scheduler airflow dags trigger ingest_era5_weather

# 5. Feature mart (combines all of the above into features.mart)
docker exec dengue_airflow_scheduler python /opt/airflow/src/features/build_features.py --export

# 6. Train 4 models, register best in MLflow
docker exec dengue_airflow_scheduler python /opt/airflow/src/models/train.py

# 7. Predict current week + 4-week forecast; write GeoJSON + cache to Redis
docker exec dengue_airflow_scheduler python /opt/airflow/src/models/predict.py --weeks-ahead 4
```

After step 7 you'll have:

```
data/processed/risk_current.geojson      # current-week risk map
data/processed/forecast_{1..4}w.geojson  # N-week-ahead forecasts
data/processed/features_mart.csv         # 736-row offline-training export
```

And in Redis: `dengue:risk:current`, `dengue:forecast:2024:{1..4}`.

---

## Dashboard (Day 9)

The dashboard is an API-only Streamlit client. It talks to `API_URL` (`http://api:8000` in Docker, `http://localhost:8000` by default outside Docker) and never connects to Postgres, Redis, or MLflow directly.

Views included:

- **Risk Map:** current risk choropleth, 1-4 week forecast selector, tier distribution, top-risk district table
- **District Detail:** district selector, latest prediction, area/division metadata, last 12 weeks of cases and rainfall
- **Model:** Production model version, RMSE, MAE, R2, outbreak F1, data freshness
- **Service status:** API, DB, Redis, and MLflow health surfaced from `/health`

Run it with the stack:

```bash
docker compose up -d --build dashboard
```

Then open:

```text
http://localhost:8501
```

---

## REST API (Day 8)

A read-only FastAPI service exposes the model + predictions over HTTP. Live at `http://localhost:8000` once the stack is up.

| Method | Path | Returns |
|---|---|---|
| `GET` | `/health` | `{status, db_ok, redis_ok, mlflow_ok, model_version, data_freshness, uptime_s}` |
| `GET` | `/metrics/model` | Current Production model's RMSE/MAE/R²/Outbreak-F1 |
| `GET` | `/districts` | List all 64 districts with name, division, centroid |
| `GET` | `/district/{id}` | One district + latest prediction + last 12 weeks of cases/weather |
| `GET` | `/risk/current` | GeoJSON FeatureCollection — current-week risk map |
| `GET` | `/risk/forecast/{1..4}` | GeoJSON FeatureCollection — N-week-ahead forecast |
| `GET` | `/docs` | Interactive Swagger UI |

### Examples

```bash
# Liveness + freshness
curl http://localhost:8000/health
# → {"status":"ok","db_ok":true,"redis_ok":true,"mlflow_ok":true,
#    "model_version":"dengue_risk_model/v1/Production","data_freshness":"2023-W52",...}

# Production model metrics
curl http://localhost:8000/metrics/model
# → {"name":"dengue_risk_model","version":"1","stage":"Production",
#    "rmse":1129.17,"mae":479.88,"r2":0.286,"outbreak_f1":0.634}

# Dhaka drill-down
curl http://localhost:8000/district/18
# → {"district":{"district_name":"Dhaka",...},
#    "latest_prediction":{"year":2023,"week":52,"predicted_cases":386.47,"risk_tier":"Critical",...},
#    "history":[...12 weekly rows...]}

# Current risk map (full GeoJSON, ~6 MB with geometry)
curl http://localhost:8000/risk/current > current_map.geojson

# 2-week-ahead forecast
curl http://localhost:8000/risk/forecast/2 > forecast_2w.geojson
```

### Source-of-truth chain
The API reads map forecasts from **Redis first** (the cache `predict.py` populates), then falls back to **disk** (`data/processed/*.geojson`). For `/risk/current`, it can also rebuild a GeoJSON response from **Postgres** if both cache and disk are cold. Per-district details always query Postgres. The MLflow Production model is loaded into memory at startup.

---

## Architecture

```
                 ┌─────────────────┐
                 │  Copernicus CDS │ (real ERA5-Land hourly weather)
                 └────────┬────────┘
                          │ download (per-month parallel)
                          ↓
   CSV ─→ ingest_disease ─┐
   SHP ─→ ingest_boundaries│           ┌──────────────┐
   TIF ─→ ingest_population├──────────→│  PostgreSQL  │
   NC4 ─→ ingest_era5     ─┘          │   + PostGIS  │
                                       └──────┬───────┘
                                              │ build_features (7-step CTE)
                                              ↓
                                       features.mart (736 rows × 20 features)
                                              │
                                              ↓
                                       train.py (4 models)
                                              │ logs → MLflow
                                              ↓
                                       Production model
                                              │
                                              ↓
                                       predict.py
                                          │   │
                                          ↓   ↓
                                      GeoJSON Redis cache
```

Each ingestion module exposes a `run()` callable that's wrapped by an Airflow DAG (`airflow/dags/dag_ingest_*.py`). The DAGs are paused by default; the typical production setup unpauses the `@weekly` ones (`ingest_era5_weather`, `ingest_disease_data`, `build_feature_mart`) so they form a weekly heartbeat.

---

## Development workflow

```bash
# Run the test suite (96 tests; needs the containers up)
docker exec -w /opt/airflow dengue_airflow_scheduler python -m pytest tests/ -v

# Lint
docker exec -w /opt/airflow dengue_airflow_scheduler python -m ruff check src/ tests/

# Hot-reload code: src/, tests/, dags/ are bind-mounted, so edits on the
# host show up in the container immediately. No rebuild needed for Python.

# Rebuild image only when Dockerfile.airflow changes (new pip deps)
docker compose build airflow-init
docker compose up -d --force-recreate airflow-scheduler airflow-webserver
```

### Project structure

```
src/
├── db.py                          # Shared psycopg2 + SQLAlchemy connection helpers
├── api/
│   └── main.py                    # FastAPI service for risk maps, districts, model metrics
├── ingestion/
│   ├── ingest_boundaries.py       # bgd_admin2.shp → geo.districts
│   ├── ingest_disease.py          # dengue_cases.csv → disease.dengue_cases
│   ├── ingest_population.py       # WorldPop 1km → geo.district_population
│   └── ingest_era5.py             # Copernicus → weather.era5_district_weekly
├── features/
│   └── build_features.py          # 7-step CTE → features.mart
└── models/
    ├── train.py                   # 4 models → MLflow → Production
    └── predict.py                 # MLflow → GeoJSON + Redis + prediction_log

airflow/dags/                       # One DAG file per src/ entrypoint
dashboard/app.py                    # Streamlit API client dashboard
tests/                              # 96 pytest tests (pure-fn + integration)
init_db.sql                         # PostgreSQL schema (auto-runs on first DB volume)
docker-compose.yml                  # 7 services
Dockerfile.airflow                  # Airflow + GDAL + ML deps
.cdsapirc                           # Copernicus PAT (gitignored in real deploys)
```

---

## Key design choices

- **Features built in SQL, not pandas.** `build_features.py` is a 7-step CTE pipeline that runs inside Postgres. Faster, auditable, and the feature mart is a single source of truth — both training and serving read the same rows.
- **6-hourly ERA5 sampling (not hourly).** Weekly aggregates don't need diurnal resolution; 6-hourly is 4× cheaper at the CDS API and 4× faster to process.
- **Parallel CDS submission.** Each year's task submits 12 monthly requests concurrently (bounded by a semaphore at 4). Reduces CDS queue wait dominance.
- **Per-month NetCDF cache.** Downloads land at `data/raw/era5_cache/{year}/`; a retry skips already-completed months, so a crash mid-year loses nothing.
- **MLflow proxy artifacts.** `mlflow server --serve-artifacts` so the Airflow tasks can log artifacts via HTTP without sharing a filesystem with the MLflow container.
- **Restart policies on all stateful containers** — survives Docker daemon restarts and Windows reboots.

---

## Related docs

- [`STATUS_REPORT.md`](STATUS_REPORT.md) — chronological story of bugs found + fixed during build
- [`MODEL_CARD.md`](MODEL_CARD.md) — model purpose, training data, limitations, ethics
- [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) — every column in every table

---

## Known limitations

- Disease data is at **division-level only** (8 of Bangladesh's 64 districts have case rows). The feature mart and model coverage are correspondingly limited.
- 2020/2021 had unusually low case counts (COVID disruption?), creating a class-imbalance challenge — outbreak F1 of 0.63 reflects this.
- The `current_year` ERA5 task caps at `today.month - 3` because ERA5-Land has a ~3-month publication lag.
- Confidence intervals are bootstrap-based (5% feature noise injection), not Bayesian. They're an uncertainty proxy, not a true posterior.
