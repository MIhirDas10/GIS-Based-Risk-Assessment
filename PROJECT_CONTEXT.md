# PROJECT CONTEXT — Dengue Risk Prediction, Bangladesh

> **Master context document.** Everything needed to understand where this project is, how it got here, what the data actually contains, what's been decided and why, and where it's going. Read this first.

**Repo:** `H:\dengue-risk-bd` · **Branch:** `develop` · **Last updated:** 2026-09-30

---

## Table of contents

1. [What this project is](#1-what-this-project-is)
2. [Current state at a glance](#2-current-state-at-a-glance)
3. [System architecture](#3-system-architecture)
4. [Data assets — full inventory](#4-data-assets--full-inventory)
5. [The new dataset (critical — mostly unexploited)](#5-the-new-dataset--critical--mostly-unexploited)
6. [Model state](#6-model-state)
7. [What's done / not done](#7-whats-done--not-done)
8. [Bugs found and fixed (16)](#8-bugs-found-and-fixed-16)
9. [Research positioning & literature findings](#9-research-positioning--literature-findings)
10. [Business model](#10-business-model)
11. [Key decisions and rationale](#11-key-decisions-and-rationale)
12. [Known issues, gotchas, ops notes](#12-known-issues-gotchas-ops-notes)
13. [Command reference](#13-command-reference)
14. [Next steps](#14-next-steps)
15. [Related documents](#15-related-documents)

---

## 1. What this project is

District-level dengue outbreak forecasting for Bangladesh. Ingests climate reanalysis, population rasters, administrative boundaries, and historical dengue case counts; engineers features in SQL; trains and registers ML models; serves weekly risk-tier maps and 1–4 week forecasts through a REST API and a Streamlit dashboard.

**Originally scoped** as a 12-day portfolio project. **Now being repositioned** as a thesis/publication-grade research project plus a potentially sustainable product (see §9 and §10).

**The core value proposition:** turn free satellite weather data into 2–4 weeks of lead time for public-health decisions (vector control deployment, hospital surge planning, supply stockpiling).

---

## 2. Current state at a glance

| Dimension | Status |
|---|---|
| Pipeline | ✅ End-to-end working on real data |
| Containers | 8 services, all `restart: always` |
| Tests | ✅ 118 passing |
| Lint | ✅ ruff clean |
| Model | XGBoost v1 in MLflow **Production** |
| Model quality | ⚠️ R² 0.286, Outbreak F1 0.634 — demo-grade, not decision-grade |
| Dashboard | ✅ Live, shows current-week (2026-W09) predictions for all 64 districts |
| Git | ❌ **Nothing committed** — all work is uncommitted in working tree |
| CI | ❌ Broken (see §12) |
| Day 10 (monitoring) | ❌ Not started |
| New DGHS data | ⚠️ In repo, **~8% extracted** — the biggest untapped asset |

---

## 3. System architecture

```
                        ┌─────────────────────┐
                        │   Copernicus CDS    │  real ERA5-Land, parallel monthly pulls
                        └─────────┬───────────┘
                                  │
   CSV  ──→ ingest_disease   ┐    │
   SHP  ──→ ingest_boundaries│    │
   TIF  ──→ ingest_population├────┼─────→ ┌──────────────────┐
   NC4  ←── ingest_era5      ┘    │       │   PostgreSQL 15  │
                                  │       │   + PostGIS 3.4  │
                                  │       └────────┬─────────┘
                                  │                │ build_features.py
                                  │                ├──→ features.mart          (training; INNER JOIN on cases)
                                  │                └──→ features.forecast_mart (serving; INNER JOIN on weather only)
                                  │                                  │
                                  │                            train.py → 4 models → MLflow
                                  │                                  │
                                  │                            Production model
                                  │                                  │
                                  │                            predict.py
                                  │                       ┌──────────┴──────────┐
                                  │                       ↓                     ↓
                                  │                  Redis cache         GeoJSON files
                                  └───────────────────────┴──────────┬──────────┘
                                                                     ↓
                                                            FastAPI  :8000
                                                                     ↓
                                                        Streamlit dashboard :8501
```

### Containers

| Container | Image | Port | Role |
|---|---|---|---|
| `dengue_postgres` | postgis/postgis:15-3.4 | 5432 | Data warehouse + spatial ops |
| `dengue_redis` | redis:7.2-alpine | 6379 | Hot cache for GeoJSON |
| `dengue_mlflow` | python:3.11-slim | 5001→5000 | Tracking + model registry (`--serve-artifacts`) |
| `dengue_airflow_init` | dengue-airflow:latest | — | One-shot: db migrate + create admin user |
| `dengue_airflow_webserver` | dengue-airflow:latest | 8080 | DAG UI (`admin`/`admin`) |
| `dengue_airflow_scheduler` | dengue-airflow:latest | — | Executes DAGs; also our general-purpose Python runner |
| `dengue_api` | dengue-risk-bd-api | 8000 | FastAPI REST |
| `dengue_dashboard` | dengue-risk-bd-dashboard | 8501 | Streamlit UI |

### Database layout

**Databases:** `dengue_db` (application), `mlflow_db` (MLflow backend store)

| Schema | Table | Rows | Notes |
|---|---|---|---|
| `geo` | `districts` | 64 | MultiPolygon EPSG:4326, from HDX `bgd_admin2.shp` |
| `geo` | `district_population` | 64 | WorldPop 2020; avg density 1392/km², total 175M |
| `disease` | `dengue_cases` | 856 | **Weekly**, 8 divisions only, 2019→2023 |
| `weather` | `era5_district_weekly` | ~17,216 | Real ERA5-Land, 64 districts, 2019→2026-W09 |
| `features` | `mart` | 736 | Training set — INNER JOIN on cases caps it at 2023-W52 |
| `features` | `forecast_mart` | 17,280 | Serving set — weather-only join, extends to 2026-W09 |
| `monitoring` | `prediction_log` | — | Prediction audit trail; `UNIQUE(district_id, year, week)` |

### Repo structure

```
src/
├── db.py                       # Shared psycopg2 + SQLAlchemy connection helpers
├── api/main.py                 # FastAPI, 7 endpoints
├── ingestion/
│   ├── ingest_boundaries.py    # bgd_admin2.shp → geo.districts
│   ├── ingest_disease.py       # dengue_cases.csv → disease.dengue_cases
│   ├── ingest_population.py    # WorldPop 1km → geo.district_population
│   └── ingest_era5.py          # Copernicus → weather.era5_district_weekly
├── features/build_features.py  # 7-step CTE → mart + forecast_mart
├── models/
│   ├── train.py                # 4 models → MLflow → auto-promote
│   └── predict.py              # MLflow → GeoJSON + Redis + prediction_log
└── monitoring/                 # EMPTY — Day 10 not started

airflow/dags/                   # 5 DAGs (ingest ×4, build_features ×1)
dashboard/app.py                # Streamlit, 571 lines, user-redesigned
tests/                          # 118 tests across 8 files
scripts/                        # Utility scripts (screenshots, smoke tests, seeding)
new dataset/                    # NEW DGHS data — see §5
```

---

## 4. Data assets — full inventory

### 4.1 Currently loaded in Postgres

| Source | Coverage | Granularity | Notes |
|---|---|---|---|
| **ERA5-Land** (Copernicus) | 2019-01 → 2026-W09, 64 districts | Weekly (6-hourly sampled) | Real reanalysis. ~3-month publication lag. |
| **Dengue cases** (Kaggle CSV) | 2019 → 2023-W52, 8 divisions | Weekly | **This is the binding constraint on training.** |
| **WorldPop** | 2020 snapshot, 64 districts | Static | Total 175M, avg density 1392/km² |
| **HDX boundaries** | `bgd_admin2.shp` | Static | 64 districts, EPSG:4326 |

### 4.2 The constraint that shaped everything

`features.mart` INNER JOINs `disease.dengue_cases`, so it can only cover weeks where **both** case data and weather exist. Cases stop at 2023-W52 → **the training mart stops at 2023-W52**, even though ERA5 runs to 2026.

This is why `features.forecast_mart` was created (weather-only join, last-year spatial-lag stand-in) so the dashboard could show *current* predictions. See §11.

### 4.3 ERA5 ingestion specifics

- **6-hourly sampling** (00/06/12/18 UTC), not hourly — 4× cheaper at CDS, statistically equivalent for weekly aggregates
- **Parallel monthly downloads**, semaphore bounded at 4 concurrent
- **Per-month NetCDF cache** at `data/raw/era5_cache/{year}/` — crash-safe, retries skip completed months
- **Vectorized zonal stats** — pre-rasterized district masks + numpy indexing, ~100× faster than the original per-hour-per-district rasterio loop
- Full 5-year ingest completed in **~1 hour**
- Current-year requests capped at `today.month - 3` to respect ERA5-Land publication lag

---

## 5. The new dataset — CRITICAL — mostly unexploited

Location: `new dataset/`

### 5.1 Structure

```
new dataset/
├── Dengue data/
│   ├── 2024/
│   │   ├── 2. Daily Dengue Dhaka City 2024 (1).xlsx
│   │   └── 3. Daily Dengue Bangladesh 2024 (1).xlsx   ← 360 daily sheets
│   ├── 2025/
│   │   ├── 2. Daily Dengue Dhaka City 2025.xlsx
│   │   └── 3. Daily Dengue Bangladesh 2025.xlsx       ← 365 daily sheets
│   ├── all_csv/                                        ← arranger output
│   └── data_arranger.py                                ← extracts only ~8%
└── External data/
    └── 2024_Dhaka.csv                                  ← Meteostat, low value
```

### 5.2 ⭐ The big finding: the raw DGHS files contain 64 districts daily

`3. Daily Dengue Bangladesh 20XX.xlsx` has **one sheet per day** (360 for 2024, 365 for 2025), each ~137 rows × 21 cols, covering **all 64 districts**.

**Per district, per day, the sheets contain:**

| Col | Bangla | Meaning |
|---|---|---|
| 5 | সরকারী | New admissions — **government** hospitals (24h) |
| 6 | বেসরকারী | New admissions — **private** hospitals (24h) |
| 7 | মোট | Total new admissions (24h) ← *only this is currently extracted* |
| 8 | সর্বমোট ভর্তি | Cumulative admissions YTD |
| 9 | **মৃত্যু** | **Deaths** (cumulative) |
| 10 | ছাড়পত্র প্রাপ্ত | Discharged |
| 11 | **বর্তমানে ভর্তি** | **Currently hospitalized** (live occupancy) |

**Scale:** ~725 day-sheets × 64 districts ≈ **46,000 district-day records × 7 metrics**.

**Current extraction:** `data_arranger.py` pulls only 5 division totals + Dhaka city, total column only. **Roughly 8% of available information.**

### 5.3 Why this matters enormously

1. **64 districts instead of 8** — the model currently predicts for 8 divisions only
2. **Daily instead of weekly** — ~7× more training rows
3. **2024–2025 ground truth** — closes the gap that caps training at 2023
4. **Deaths + occupancy** — enables *burden* forecasting, not just case counts
5. **Gov/private split** — a rare variable; see Niche 2 in §9
6. **Each sheet is a dated snapshot** → **the revision history can be reconstructed**. See Niche 1 in §9. This may be the single most valuable property of the dataset.

### 5.4 The arranged files (`all_csv/`)

| File | Date range | Cases end | Weather end | Total cases |
|---|---|---|---|---|
| `Daily_Dengue_Climate_PollutionDataDhakaCity_2016_2024_updated.xlsx` | 2016-01-01 → 2025-12-31 | **2025-12-31** | 2024-09-30 | 316,861 |
| `ctg_daily_21_24.xlsx` | 2021-01-01 → 2025-12-31 | 2025-12-31 | 2024-11-25 | 14,052 |
| `khulna_daily_21_24.xlsx` | same | 2025-12-31 | 2024-06-30 | 8,931 |
| `raj_daily_21_24.xlsx` | same | 2025-12-31 | 2024-11-25 | 8,078 |
| `rangpur_daily_21_24.xlsx` | same | 2025-12-31 | 2024-11-25 | 1,274 |
| `sylhet_daily_21_24.xlsx` | same | 2025-12-31 | 2024-11-25 | 627 |

**Note:** the weather columns in these files are stale, but that doesn't matter — real ERA5 through 2026 is already in Postgres for all 64 districts. **Only the case counts are needed from these files.**

Regional files have 8 missing days each. Sylhet has a typo column: `DengeuCases`.

### 5.5 Dhaka yearly case totals (sanity-check against reality)

| Year | Cases | Note |
|---|---|---|
| 2016 | 5,776 | |
| 2017 | 2,653 | |
| 2018 | 10,143 | |
| 2019 | 51,803 | First major epidemic |
| 2020 | 1,224 | COVID — suppressed reporting and/or transmission |
| 2021 | 23,615 | |
| 2022 | 39,220 | |
| 2023 | **110,008** | Record year — matches known national total of ~321k / 1,705 deaths |
| 2024 | 40,765 | |
| 2025 | 31,654 | |

Totals align with published figures, so the extraction logic is trustworthy.

### 5.6 Pollution data — limited use

The Dhaka workbook has `SolarRad, SO2, NO2, CO, O3, PM2.5, PM10, HI` but **only for 2016-01-01 → 2018-07-31 (943 days)**.

This does **not overlap** the current 2019+ training window. To use pollution features, training would need to start from 2016 — and even then pollution would cover only ~26% of rows. Better to source fresh pollution data (OpenAQ, Copernicus CAMS) than to rely on this.

### 5.7 External data verdict: ❌ not useful as-is

`2024_Dhaka.csv` is a **Meteostat daily export** (schema `date,tavg,tmin,tmax,prcp,snow,wdir,wspd,wpgt,pres,tsun` is Meteostat's exact format).

| Problem | Detail |
|---|---|
| Coverage | 1 year (2024), 1 city (Dhaka) |
| Empty columns | `snow`, `wdir`, `wpgt`, `tsun` — 100% null |
| **No humidity** | Meteostat *daily* omits RH entirely. Humidity is one of 4 core model inputs. |
| Redundant | ERA5 already covers 64 districts, 2019–2026, better quality |

**The empty columns are NOT the problem** — `snow` is irrelevant in Bangladesh; `wdir`/`wpgt` have no established dengue link; `tsun` is minor.

**The real problem is missing humidity.** Meteostat's **hourly** endpoint includes `rhum` and `dwpt`; the daily endpoint does not. If pursuing Meteostat, pull hourly and aggregate.

**But the source has one real virtue:** ~1-day lag vs ERA5's ~3 months. That's the fix for stale predictions — though **Open-Meteo is the better tool** for that job (see §14).

⚠️ **Train/serve skew warning:** ERA5 is a grid-cell average from a reanalysis model; Meteostat is a point station observation. They systematically disagree. A model trained on ERA5 will be miscalibrated on Meteostat input. Bias correction (using the 2024–2026 overlap) would be mandatory.

---

## 6. Model state

### Production model

| Field | Value |
|---|---|
| Registry name | `dengue_risk_model` |
| Version | 1 |
| Stage | **Production** |
| Algorithm | XGBoost (`max_depth=6, lr=0.05, n_estimators=500, subsample=0.8, colsample_bytree=0.8, min_child_weight=3`) |
| Train / validate / test | 2019–2021 (224 rows) / 2022 (232) / 2023 (280) |
| Split strategy | **Temporal — no random shuffling** (leakage-safe) |

### Test-set results (2023)

| Model | RMSE | MAE | R² | Outbreak F1 |
|---|---|---|---|---|
| Seasonal Naive (baseline) | 1349 | 353 | −0.019 | 0.218 |
| Ridge (Pipeline w/ scaler) | 1169 | 378 | 0.235 | 0.662 |
| **XGBoost (Production)** | **1129** | 480 | **0.286** | 0.634 |
| LightGBM | 1137 | 544 | 0.276 | 0.626 |

Improvement over baseline: **+16.3%** (auto-promotion threshold is 15%).

### The 18 features

```
Direct weather:  temp_mean_c, temp_max_c, rainfall_mm, humidity_pct
Lag (2/4 wk):    rainfall_lag_2w, rainfall_lag_4w, temp_lag_2w, humidity_lag_2w
Rolling (4 wk):  temp_rolling_4w, rainfall_rolling_4w
Interactions:    humidity_x_temp, rainfall_x_density
Spatial:         cases_spatial_lag
Cyclical/cal:    week_sin, week_cos, month
Static:          population_density, hotspot_rank
```

### ⚠️ Honest assessment

**R² 0.286 and Outbreak F1 0.634 is demo-grade, not decision-grade.** Roughly 1 in 3 outbreak weeks is missed or false-alarmed. Nobody would pay for this, and it won't carry a paper on accuracy alone.

This is **fine** under the research repositioning (§9) — the contribution there is honest evaluation methodology and rare data, not model performance. But it must not be oversold.

**Why performance is limited:**
- Only 8 districts with case data (56 districts contribute nothing to training)
- Weekly granularity, 736 training rows
- 2020–2021 are COVID-distorted (1,224 and 23,615 cases vs 51,803 in 2019)
- No mosquito-control, serotype, immunity, or mobility data
- Single severe-outbreak year (2023) in test

---

## 7. What's done / not done

### ✅ Done

**Days 1–7 — pipeline**
- 8-container Docker stack, all with restart policies
- PostGIS schema, auto-bootstrapped via `init_db.sql`
- 4 ingestion modules, all working
- 5 Airflow DAGs
- 7-step CTE feature pipeline in pure SQL
- 4 models + MLflow registry with auto-promotion
- Prediction → GeoJSON + Redis + DB log
- **Real ERA5 ingested**, 2019→2026, 64 districts

**Day 8 — API**
- `src/api/main.py`, 7 endpoints
- 3-tier fallback for GeoJSON: Redis → disk → live Postgres reconstruction
- Model loaded at startup via lifespan handler
- 20 tests

**Day 9 — dashboard**
- `dashboard/app.py`, 571 lines, editorial UI (user-redesigned)
- 3 tabs: Risk Map (Folium choropleth), District Detail, Model
- 22 tests

**Enhancement — `forecast_mart`**
- Current-week predictions for all 64 districts (2026-W09)

**Docs & quality**
- 118 tests, ruff clean, `pyproject.toml` config
- README, MODEL_CARD, DATA_DICTIONARY, STATUS_REPORT, ROADMAP_AND_BUSINESS

### ❌ Not done

**Day 10 — monitoring (nothing exists)**
- `src/monitoring/monitor.py` — Evidently drift detection
- `airflow/dags/dag_monitor.py`
- `airflow/dags/dag_retrain.py`
- `evidently==0.4.16` is in requirements but unused

**Day 11–12 remainder**
- `IMPACT.md`
- CI fix (currently broken — see §12)
- **Git commit + merge `develop` → `main`**

**Orchestration gaps**
- `forecast_mart` build not wired into any DAG (manual `--forecast`)
- No prediction DAG (`predict.py` runs manually despite its docstring referencing one)

**The big one**
- **64-district daily extraction from raw DGHS files** — see §5

---

## 8. Bugs found and fixed (16)

Found during verification of "Days 1–7 done but untested."

| # | File | Bug | Fix |
|---|---|---|---|
| 1 | `init_db.sql` | `features.mart` schema stale, missing 5 columns | Removed DDL; `build_features.py` owns it |
| 2 | `init_db.sql` | `mlflow_db` never created | Added `CREATE DATABASE mlflow_db` |
| 3 | `init_db.sql` | No UNIQUE on disease cases | `UNIQUE(district_id, year, month, week)` |
| 4 | `ingest_boundaries.py` | **File was a stub** — DAG imported nonexistent `run()` | Wrote the implementation |
| 5 | `ingest_disease.py` | CSV is daily, not aggregated → UNIQUE violation | Aggregate to weekly before insert |
| 6 | `ingest_disease.py` | INSERT non-idempotent | `ON CONFLICT DO UPDATE` |
| 7 | `ingest_population.py` | WorldPop URL 404 | Corrected bucket path |
| 8 | `ingest_era5/population.py` | Wrong shapefile column (`ADM2_EN`) | Added `adm2_name` |
| 9 | `build_features.py` | Weeks crossing month boundaries duplicated PK | Added `cases_weekly` aggregating CTE |
| 10 | `train.py` | **Ridge scaler not logged** → predictions 1000× inflated (9M cases!) | Wrapped in sklearn `Pipeline` |
| 11 | `docker-compose.yml` | MLflow artifact uploads failed | `--serve-artifacts` |
| 12 | `ingest_era5.py` | CDS-beta returns ZIP-wrapped NetCDF | `_is_zip` + `_unwrap_zip_to_nc` |
| 13 | `ingest_era5.py` | CDS-beta renamed `time` → `valid_time` | Accept either |
| 14 | `docker-compose.yml` | Airflow tasks defaulted `POSTGRES_HOST=localhost` | Added env vars to `airflow-common` |
| 15 | `ingest_era5.py` | Future-month requests → `MultiAdaptorNoDataError` | Cap at `today.month - 3` |
| 16 | `docker-compose.yml` | postgres/redis/mlflow had no restart policy | `restart: always` |

**Most dangerous:** #10 (silent 1000× prediction inflation) and #4 (a stub file that would have blocked everything). There's now a regression test asserting `preds.max() < 50000`.

---

## 9. Research positioning & literature findings

### 9.1 Literature search verdict (2026-09-30)

**⚠️ The base project is NOT novel.** Bangladesh district-level ML dengue forecasting is saturated, mostly within the last 12 months:

- [District-Level Dengue EWS in Bangladesh](https://pmc.ncbi.nlm.nih.gov/articles/PMC13030265/) — hybrid explainable AI + Bayesian deep learning
- [Bayesian hybrid statistical/ML dengue forecasting Bangladesh](https://www.medrxiv.org/content/10.1101/2025.09.14.25335716v1) — SARIMA-XGBoost, Sept 2025
- [Interpretable tree-based dengue EWS Bangladesh](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12063067/) — LightGBM + SHAP
- [Data-driven ML dengue forecasting Bangladesh](https://onlinelibrary.wiley.com/doi/10.1002/hsr2.72147)

**Also already covered:**
- Dengue nowcasting / reporting-delay correction is an active field — [NobBS](https://www.biorxiv.org/content/10.1101/663823.full.pdf), [Brazil](https://journals.plos.org/plosntds/article?id=10.1371%2Fjournal.pntd.0012501), [Thailand](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7055098/), [São Paulo](https://www.medrxiv.org/content/10.1101/2025.05.05.25326971.full.pdf)
- Hospitalization forecasting under reporting delay — [published Jan 2026, Brazil, PLOS Digital Health](https://journals.plos.org/digitalhealth/article?id=10.1371%2Fjournal.pdig.0001206)
- Dengue mortality prediction — exists but **patient-level clinical only** ([bedside score](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8142072/), [lab-value models](https://bmcinfectdis.biomedcentral.com/articles/10.1186/s12879-018-3141-6))

### 9.2 ⭐ The opening

From the backfill literature, a key quoted finding:

> **"At the time of recent publication, there are no dengue surveillance systems that maintain publicly available records of backfilling and updated case counts."**

The 725 dated DGHS daily snapshots in `new dataset/` can reconstruct exactly this. Existing nowcasting papers must **model** delay distributions because they cannot **observe** them. This project can observe them.

### 9.3 Five claimable niches (ranked)

**🥇 1 — First public dengue data-revision (vintage) corpus for an LMIC**
Reconstruct per-district, per-day revision history from dated snapshots. Publish (a) the dataset, (b) empirical delay/backfill distributions by district and outbreak phase, (c) a test of whether existing nowcasting methods recover truth when ground truth is available.
*Venue:* Scientific Data, Data in Brief, Epidemics. *Risk:* low. *Defensibility:* high — irreplicable without the archive.

**🥈 2 — Private-sector care-seeking share as a real-time surveillance-bias signal**
Literature treats private care as a *limitation causing underreporting* ([systematic review](https://pmc.ncbi.nlm.nih.gov/articles/PMC4253126/), [Malaysia facility-use study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6831957/) — 54% of inpatient care was private). Nobody uses the *observed* gov/private ratio as a time-varying correction factor or capacity-stress indicator.
*Questions:* Does private share rise as public capacity saturates? Does it predict subsequent underreporting? Can it correct incidence in real time?
*Why strong:* inverts a known limitation into a usable signal; health-systems audience, not just ML.

**🥉 3 — Population-level dengue mortality & CFR forecasting**
All existing death prediction is patient-level clinical (needs lab values). Nobody forecasts district-level deaths / CFR as a time series. A *rising CFR at constant case counts* is the clearest health-system-failure signal and the thing a health ministry most needs.

**4 — Direct occupancy (prevalence) forecasting instead of incidence**
Standard practice forecasts new cases then multiplies by assumed length-of-stay. The DGHS data contains "currently admitted" directly observed. Forecasting the prevalence state directly avoids the LOS assumption — and the derived-vs-observed gap is itself a publishable negative result.

**5 — Benchmarking nowcasting methods against observed ground truth**
Every nowcasting paper validates against *eventual finalised* counts. With a vintage corpus you can ask whether these methods recover the *true* delay distribution or merely fit a plausible one. Pairs naturally with Niche 1.

### 9.4 Proposed thesis framing

> **"Beyond case counts: surveillance-aware dengue forecasting under imperfect reporting in Bangladesh"**
>
> Three contributions: (i) first public LMIC dengue revision corpus + empirical delay characterisation; (ii) public/private care-seeking ratio as real-time bias-correction and capacity-stress signal; (iii) district-level mortality/CFR forecasting as a health-system failure indicator.

**Narrative:** *everyone else forecasts the number; this work forecasts the number AND characterises how wrong the number is when it arrives.*

**Tactic:** carve Niche 1 out as a standalone data paper early — fast, low-risk, produces a citable artifact while the larger work develops.

### 9.5 Liberating reframe

Under this positioning, **model accuracy is not the contribution.** A modest model with rigorous real-time validation is *more* publishable than a strong model with inflated retrospective numbers, because it corrects the field rather than adding to the pile. The current R² 0.286 becomes part of the finding: *"even a well-specified model achieves only X under realistic conditions vs Y reported retrospectively."*

### 9.6 ⚠️ Verification required

The above came from ~6 searches, not a systematic review. **Must be independently verified** — especially Niche 2, which is the least certain. A proper systematic review is step one of the thesis regardless.

---

## 10. Business model

Full detail in [`ROADMAP_AND_BUSINESS.md`](ROADMAP_AND_BUSINESS.md). Summary:

**Recommended sequence — don't pick one buyer, stage them so each funds the next:**

| Phase | Timeline | Model | Rationale |
|---|---|---|---|
| **1. Wedge** | 0–6 mo | Grant-funded pilot (icddr,b, Wellcome, Gates, ADB) | Non-dilutive money + real data + credibility partner. Closes the accuracy gap. |
| **2. Engine** | 6–18 mo | B2B subscription — private hospital chains ("surge planner") + FMCG/pharma data licensing | Hospitals have clearest ROI; FMCG is the cash cow (dengue season = their sales season) |
| **3. Moat** | 18+ mo | Parametric insurance data licensing (reinsurers, ADB) | Highest margin; only credible after a validated multi-year track record |

**Why sustainable:** data network effects (every week compounds and can't be bought retroactively); one engine serves many buyers at near-zero marginal cost; the public-good version *generates the data* that powers the commercial versions.

**The gating item:** *"nobody pays for a 63%-accurate outbreak flag."* Step 0 is real recent data → retrain → backtest slide (*"X weeks lead time, Y% of outbreak-weeks caught"*). No backtest = no sale.

**Note the alignment:** the burden-forecasting research direction (§9, Niches 3–4) is *also* the strongest product direction. "How many beds will I need?" is what a hospital pays for and what nobody has published. Research and revenue point the same way.

---

## 11. Key decisions and rationale

| Decision | Why |
|---|---|
| **Features built in SQL, not pandas** | Runs inside Postgres; auditable; single source of truth for both training and serving |
| **6-hourly ERA5, not hourly** | 4× cheaper at CDS, 4× faster processing; weekly aggregates don't need diurnal resolution |
| **Parallel CDS submission (sem=4)** | Queue waits dominate; overlapping them cut a multi-hour job to ~1 hour |
| **Per-month NetCDF cache on disk** | Crash/power-cut safe; retries skip completed months (this machine has had repeated power cuts) |
| **Pre-rasterized masks + numpy** | Replaced 138k rasterio calls/month with 64 numpy ops — ~100× faster |
| **MLflow `--serve-artifacts`** | Airflow container can log artifacts over HTTP without sharing a filesystem with MLflow |
| **Separate `forecast_mart`** | Training mart needs the case join (for the target); serving doesn't. Splitting them let the dashboard show 2026 data without polluting the training set. |
| **Last-year spatial-lag stand-in** | For future weeks, neighbour case counts are unknown. Climatology repeats, and the model is already trained to read this signal. Approximation is acknowledged. |
| **`restart: always` everywhere** | Power cuts during development repeatedly killed the stack |
| **Ridge wrapped in `Pipeline`** | The scaler must travel with the model into MLflow — otherwise raw features hit scaled coefficients (bug #10) |

---

## 12. Known issues, gotchas, ops notes

### 🔴 Blocking / important

| Issue | Detail |
|---|---|
| **Nothing committed to git** | All Days 1–9 work, all bug fixes, all docs are uncommitted. On disk only. |
| **CI is broken** | `.github/workflows/ci.yml` installs only `ruff pytest` — but tests import pandas, fastapi, streamlit, psycopg2. It will fail. Also pins Python 3.10 while the stack runs 3.11. |
| **Docker goes down frequently** | Power cuts on this machine. Data survives (named volumes), but the stack needs `docker compose up -d` after each. |
| **Model not decision-grade** | R² 0.286, F1 0.634. Do not oversell. |

### 🟡 Minor

| Issue | Detail |
|---|---|
| `psutil` in `Dockerfile.api` | Installed but never imported. Harmless, removable. |
| Model tab bar chart | Plots RMSE (1129) and MAE (480) on the same axis as R² (0.286) and F1 (0.634) — the 0–1 metrics render as invisible lines. Metric cards above already show them. Split or drop the chart. |
| `forecast_mart` not in a DAG | Manual `--forecast` invocation |
| No prediction DAG | `predict.py` docstring references an orchestration DAG that doesn't exist |
| Sylhet column typo | `DengeuCases` in `sylhet_daily_21_24.xlsx` |
| 8 missing days | In each regional arranged file |

### ⚙️ Ops notes

- **Windows console can't print Bangla** — `UnicodeEncodeError` with cp1252. Write output to a UTF-8 file and read it back.
- **Git Bash mangles container paths** — `/opt/airflow/...` becomes `C:/Program Files/Git/opt/airflow/...`. Use PowerShell for `docker exec`, or prefix paths with `//`.
- **`docker cp` into an existing dir nests it** — `docker cp tests container:/opt/airflow/tests` creates `/opt/airflow/tests/tests`. Delete first or copy individual files.
- **User-installed pip packages vanish on container recreate** — `pytest`, `ruff`, `fastapi`, `httpx` were installed with `pip install --user` into the scheduler. Recreating the container loses them. They're now pinned in `Dockerfile.airflow`, but a rebuild is needed for that to take effect.
- **Container file permissions** — the `airflow` user (UID 50000) can't always write to bind-mounted Windows paths. Use `-u root` for `chown`/`rm` operations.
- **Port 5000 was taken** on this machine (a node process), so MLflow maps to **5001** on the host.

### 🔑 Credentials / endpoints

| Service | URL | Credentials |
|---|---|---|
| Airflow | http://localhost:8080 | `admin` / `admin` |
| MLflow | http://localhost:5001 | none |
| API docs | http://localhost:8000/docs | none |
| Dashboard | http://localhost:8501 | none |
| Postgres | `localhost:5432` | `dengue_admin` / `dengue_pass_2024` / db `dengue_db` |
| Copernicus CDS | `.cdsapirc` at repo root | PAT — **licence must be accepted on the CDS website per dataset** |

> ⚠️ These are development credentials committed in plain text. Rotate and externalise before any deployment.

---

## 13. Command reference

```bash
# Bring the whole stack up (after a power cut / reboot)
docker compose up -d

# --- Ingestion ---
docker exec dengue_airflow_scheduler python /opt/airflow/src/ingestion/ingest_boundaries.py
docker exec dengue_airflow_scheduler python /opt/airflow/src/ingestion/ingest_disease.py
docker exec dengue_airflow_scheduler python /opt/airflow/src/ingestion/ingest_population.py --year 2020

# ERA5 (via Airflow — long-running, ~1 hr for 5 years)
docker exec dengue_airflow_scheduler airflow dags unpause ingest_era5_weather
docker exec dengue_airflow_scheduler airflow dags trigger ingest_era5_weather

# --- Features ---
docker exec dengue_airflow_scheduler python /opt/airflow/src/features/build_features.py --export     # training mart
docker exec dengue_airflow_scheduler python /opt/airflow/src/features/build_features.py --forecast    # serving mart

# --- Model ---
docker exec dengue_airflow_scheduler python /opt/airflow/src/models/train.py
docker exec dengue_airflow_scheduler python /opt/airflow/src/models/predict.py --weeks-ahead 4
docker exec dengue_airflow_scheduler python /opt/airflow/src/models/predict.py --use-training-mart   # back-test mode

# --- Quality ---
docker exec -w /opt/airflow -e API_URL=http://api:8000 dengue_airflow_scheduler python -m pytest tests/ -v
docker exec -w /opt/airflow -u airflow dengue_airflow_scheduler python -m ruff check src/ tests/ dashboard/

# --- Screenshots (Playwright + headless Chromium in the scheduler container) ---
docker exec -w /opt/airflow dengue_airflow_scheduler python scripts/screenshot_dashboard.py
```

> **Note:** Run `docker exec` from **PowerShell**, not Git Bash, to avoid path mangling.

---

## 14. Next steps

### Immediate (protects existing work)
1. **Fix CI** — install real dependencies, align Python to 3.11
2. **Commit everything** — Days 1–9, 16 bug fixes, all docs are uncommitted

### The unlock (research + product foundation)
3. **Build the 64-district DGHS extractor**, preserving vintage structure:
   ```
   (district, report_date, as_of_date,
    gov_admissions, private_admissions, total_admissions,
    cumulative_admissions, deaths, currently_admitted, discharged)
   ```
   **Critical:** preserve the snapshot dimension (`as_of_date`) rather than flattening to a final series. That distinction is what turns a dashboard dataset into a research dataset.

### Then
4. **Systematic literature review** to confirm the niches in §9
5. **Open-Meteo integration** — kills both the 3-month lag *and* the weather-persistence assumption (16-day real forecasts). Archive is ERA5-derived → minimal train/serve skew. https://open-meteo.com
6. **Retrain on 2016–2025 daily, 64 districts** → produce the backtest
7. **Day 10** — Evidently drift monitoring + auto-retrain DAGs
8. `IMPACT.md`, merge `develop` → `main`

### Additional data sources worth adding

| Purpose | Source |
|---|---|
| Recent + forecast weather | [Open-Meteo](https://open-meteo.com) — free, no key, 16-day forecast, ERA5-derived archive |
| Station observations (~1-day lag) | [Meteostat](https://meteostat.net/en/) — use **hourly** endpoint for humidity |
| Solar / met, global | [NASA POWER](https://power.larc.nasa.gov/) |
| Air quality | [OpenAQ](https://openaq.org), [Copernicus CAMS](https://ads.atmosphere.copernicus.eu) |
| Vector habitat (NDVI/NDWI, LST, standing water) | [Google Earth Engine](https://earthengine.google.com) |
| Early behavioural signal | Google Trends via `pytrends` — searches for "ডেঙ্গু"/"dengue"/"platelet" can lead official reports |
| Boundaries, population, facilities | [HDX Bangladesh](https://data.humdata.org/group/bgd) |
| Upstream surveillance | [DGHS](https://dghs.gov.bd), [IEDCR](https://iedcr.gov.bd) |

> ⚠️ Verify exact endpoint paths — organisations and datasets are real, but URL structures change.

---

## 15. Related documents

| File | Contents |
|---|---|
| [`README.md`](README.md) | Setup, quickstart, architecture, API examples, dev workflow |
| [`STATUS_REPORT.md`](STATUS_REPORT.md) | Chronological build narrative, bug-fix log, day-by-day state |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Model purpose, intended use, metrics, training data, ethics, maintenance (Mitchell et al. template) |
| [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) | Every column in every table, plus Redis keys and filesystem artifacts |
| [`ROADMAP_AND_BUSINESS.md`](ROADMAP_AND_BUSINESS.md) | Technical roadmap tiers, business-model survey, implementation phases, risks |
| **`PROJECT_CONTEXT.md`** | **This file — master context** |

---

## One-paragraph summary for a newcomer

A working end-to-end dengue forecasting system for Bangladesh: real Copernicus ERA5 weather ingested via Airflow into PostGIS, features engineered in SQL, four models trained with MLflow auto-promotion (XGBoost in Production), served through FastAPI and a Streamlit dashboard showing current-week risk maps for all 64 districts. 118 tests pass and the code is lint-clean, but **nothing is committed to git**, the CI is broken, and Day 10 (monitoring) was never started. The model is demo-grade (R² 0.286) because training data covers only 8 divisions weekly through 2023. **The decisive asset is a newly-added folder of raw DGHS daily reports containing all 64 districts × 725 days × 7 metrics — including deaths, hospital occupancy, and a government/private care split — of which only ~8% is currently extracted.** Because each file is a dated snapshot, the surveillance revision history can be reconstructed, which the literature says no public dengue system provides. That property, not the model, is the project's path to publishable novelty and to a defensible product.
