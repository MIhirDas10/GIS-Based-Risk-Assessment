# Data Dictionary

Authoritative reference for every column in every table the pipeline reads or writes. Tables live in the `dengue_db` Postgres database; MLflow has its own database (`mlflow_db`) which is not documented here (managed by MLflow).

---

## Schema overview

| Schema | Purpose | Tables |
|---|---|---|
| `geo` | Geographic reference data | `districts`, `district_population` |
| `disease` | Raw case observations | `dengue_cases` |
| `weather` | ERA5 weather aggregates | `era5_district_weekly` |
| `features` | ML-ready feature mart | `mart` |
| `monitoring` | Prediction tracking + drift | `prediction_log` |

---

## `geo.districts`

64 Bangladesh second-administrative-level units (`zila`/`district`).

| Column | Type | Source | Description |
|---|---|---|---|
| `district_id` | `SERIAL PK` | auto | Internal surrogate key. |
| `district_name` | `VARCHAR(100) NOT NULL` | shapefile `adm2_name` | English name. Note historical variants are mapped to a canonical form (e.g. `Chittagong`/`Chottogram` → `Chattogram`). |
| `division_name` | `VARCHAR(100)` | shapefile `adm1_name` | Parent administrative division. |
| `geometry` | `GEOMETRY(MultiPolygon, 4326)` | shapefile | District boundary in WGS-84 (EPSG:4326). Single-polygon districts are wrapped in MultiPolygon for schema uniformity. |
| `area_km2` | `FLOAT` | shapefile `area_sqkm` | Computed area in km². Used by `ingest_population` for density. |
| `created_at` | `TIMESTAMP DEFAULT NOW()` | auto | First-insert timestamp. |

**Constraints:** `PRIMARY KEY (district_id)`. No UNIQUE on `district_name` — duplicate names would indicate a data-quality bug rather than a constraint to enforce.

---

## `geo.district_population`

Static population reference per district (annual snapshots).

| Column | Type | Source | Description |
|---|---|---|---|
| `district_id` | `INT FK → geo.districts` | join | District. |
| `population` | `BIGINT` | WorldPop sum-of-pixels | Total persons living in the district that year. |
| `population_density` | `FLOAT` | computed (`population / area_km2`) | Persons per km². Used by the risk-scoring formula. |
| `year` | `INT` | WorldPop release | Reference year (currently 2020). |

**Constraints:** `PRIMARY KEY (district_id, year)`.

---

## `disease.dengue_cases`

Weekly aggregated dengue case counts. Source CSV is daily; `prepare_case_rows` sums by ISO week.

| Column | Type | Source | Description |
|---|---|---|---|
| `id` | `SERIAL PK` | auto | Row id. |
| `district_id` | `INT FK → geo.districts` | name lookup | District. |
| `year` | `INT NOT NULL` | parsed from CSV `Month` | ISO year. |
| `month` | `INT NOT NULL` | parsed from CSV `Month` | Calendar month (1–12). The earliest month of weeks that span two months. |
| `week` | `INT` | ISO week of `Month` date | ISO 8601 week (1–53). |
| `dengue_cases` | `INT NOT NULL` | CSV `Patients` summed | Total cases reported in the district that week. |
| `cases_per_100k` | `FLOAT` | unused (NULL) | Reserved — currently the column exists but is not populated; the mart re-computes incidence on the fly. |
| `data_source` | `VARCHAR(50) DEFAULT 'kaggle'` | constant | Provenance tag. |
| `ingested_at` | `TIMESTAMP DEFAULT NOW()` | auto | Last-touched timestamp (updated on ON CONFLICT). |

**Constraints:** `UNIQUE(district_id, year, month, week)` — re-running the ingest is idempotent.

**Known coverage:** Only 8 districts (the divisional capitals) have rows. The other 56 districts are present in `geo.districts` but never appear in `dengue_cases`. This is a data limitation, not a bug.

---

## `weather.era5_district_weekly`

ERA5-Land 6-hourly reanalysis, zonal-aggregated to per-district weekly summaries.

| Column | Type | Source | Description |
|---|---|---|---|
| `id` | `SERIAL PK` | auto | Row id. |
| `district_id` | `INT FK → geo.districts` | shapefile name match | District. |
| `year` | `INT NOT NULL` | timestep ISO year | |
| `week` | `INT NOT NULL` | timestep ISO week | |
| `temp_mean_c` | `FLOAT` | ERA5 `t2m` (K) → mean → °C | Weekly mean of 2-metre air temperature over the district. |
| `temp_max_c` | `FLOAT` | ERA5 `t2m` (K) → max → °C | Weekly maximum (over all 6-hourly timesteps × all pixels). |
| `rainfall_mm` | `FLOAT` | ERA5 `tp` (m/hour) → sum × 1000 | Weekly total precipitation in millimetres. |
| `humidity_pct` | `FLOAT` | derived from `t2m` + `d2m` via Magnus formula | Weekly mean relative humidity (0–100). |
| `ingested_at` | `TIMESTAMP DEFAULT NOW()` | auto | Updated on UPSERT. |

**Constraints:** `UNIQUE(district_id, year, week)` — same year-week from different DAG runs UPSERTs in place.

**Resolution & coverage:**
- Spatial grid: ERA5-Land 0.1° × 0.1° (~10 km), Bangladesh BBOX `[N=26.7, W=88.0, S=20.5, E=92.7]` → 63 lat × 48 lon cells
- Temporal sampling: 4 per day (00, 06, 12, 18 UTC)
- Year coverage: 2019–2026 (partial — current year capped at `today.month - 3` for the ERA5-Land publication lag)

---

## `features.mart` — the ML training table

Built by `src/features/build_features.py` via a 7-step CTE pipeline. Full rebuild on each DAG run (TRUNCATE + INSERT).

| Column | Type | Stage | Description |
|---|---|---|---|
| `district_id` | `INTEGER FK → geo.districts` | identifier | |
| `year` | `INTEGER` | identifier | |
| `week` | `INTEGER` | identifier | |
| **Direct weather features** |
| `temp_mean_c` | `FLOAT` | direct | Current-week mean temperature (°C). |
| `temp_max_c` | `FLOAT` | direct | Current-week max temperature (°C). |
| `rainfall_mm` | `FLOAT` | direct | Current-week total rainfall (mm). |
| `humidity_pct` | `FLOAT` | direct | Current-week mean relative humidity (%). |
| **Lag features (mosquito breeding chain is 2–4 weeks)** |
| `rainfall_lag_2w` | `FLOAT` | LAG window | Rainfall 2 weeks ago. NULL for the first 2 weeks per district. |
| `rainfall_lag_4w` | `FLOAT` | LAG window | Rainfall 4 weeks ago. NULL for the first 4 weeks per district. |
| `temp_lag_2w` | `FLOAT` | LAG window | Temperature 2 weeks ago. |
| `humidity_lag_2w` | `FLOAT` | LAG window | Humidity 2 weeks ago. |
| **Rolling features (sustained conditions)** |
| `temp_rolling_4w` | `FLOAT` | AVG window 4w | 4-week trailing mean temperature. |
| `rainfall_rolling_4w` | `FLOAT` | SUM window 4w | 4-week trailing total rainfall. |
| **Interaction features** |
| `humidity_x_temp` | `FLOAT` | `humidity * temp` | Heat-humidity stress proxy. Both individually permissive of mosquitoes; together = ideal. |
| `rainfall_x_density` | `FLOAT` | `rainfall * pop_density` | Rainfall in dense urban areas creates more breeding sites (clogged drains, containers). |
| **Spatial features** |
| `cases_spatial_lag` | `FLOAT` | `ST_Touches` neighbours | Mean of dengue_cases in adjacent districts THIS week. 0 if no reporting neighbours. |
| **Cyclical / calendar features** |
| `week_sin` | `FLOAT` | `sin(2π·week/52)` | Sine encoding so week 52 is "close" to week 1. |
| `week_cos` | `FLOAT` | `cos(2π·week/52)` | Cosine pair. |
| `month` | `INTEGER` | calendar | 1–12. Kept for human readability (model also has sin/cos). |
| **Static / structural** |
| `population_density` | `FLOAT` | from `geo.district_population` (2020) | Persons/km². |
| `hotspot_rank` | `FLOAT` | `PERCENT_RANK` over all-time mean cases | 0 = historically lowest-risk district, 1 = highest (Dhaka). |
| **Targets** |
| `dengue_cases` | `INTEGER` | from `disease.dengue_cases` (summed if week spans two months) | Regression target. |
| `cases_per_100k` | `FLOAT` | from disease ingest | Per-capita rate (currently NULL — feature not used by model). |

**Constraints:** `PRIMARY KEY (district_id, year, week)`. Re-running `build_features.py` TRUNCATEs and re-INSERTs.

**Coverage:** Only districts present in BOTH `disease.dengue_cases` AND `weather.era5_district_weekly` survive the INNER JOIN. In practice that means 8 districts × ~92 weeks each ≈ 736 rows.

**Indexes:**
- `idx_mart_year (year)` — for the train/val/test temporal split
- `idx_mart_district (district_id)` — for per-district dashboard queries
- `idx_mart_district_year_week (district_id, year, week)` — composite for the most common query pattern

---

## `monitoring.prediction_log`

One row per (district, year, week) prediction; `predict.py` UPSERTs.

| Column | Type | Description |
|---|---|---|
| `id` | `SERIAL PK` | auto |
| `district_id` | `INT` | District |
| `year` | `INT` | ISO year of the predicted week |
| `week` | `INT` | ISO week |
| `predicted_cases` | `FLOAT` | Model output (clipped at 0) |
| `actual_cases` | `INT` | Backfilled later by the (planned) `dag_monitor.py` with a 4-week lag |
| `risk_tier` | `VARCHAR(20)` | One of `Low`, `Moderate`, `High`, `Critical` |
| `model_version` | `VARCHAR(50)` | e.g. `dengue_risk_model/Production` |
| `predicted_at` | `TIMESTAMP DEFAULT NOW()` | Last write time (refreshed on UPSERT) |

**Constraints:** `uq_prediction_log_district_year_week UNIQUE(district_id, year, week)` — keeps each (district, week) row monotonically updated.

---

## File-system artifacts (not in PostgreSQL)

| Path | Producer | Description |
|---|---|---|
| `data/processed/features_mart.csv` | `build_features.py --export` | Snapshot of `features.mart` for offline experimentation |
| `data/processed/risk_current.geojson` | `predict.py` | Current-week risk map; one Feature per district with `properties.{predicted_cases, risk_tier, risk_color, hotspot_rank, …}` |
| `data/processed/forecast_{1..4}w.geojson` | `predict.py` | N-week-ahead forecasts |
| `data/raw/era5_cache/{year}/era5_bangladesh_{year}_{MM}.nc` | `ingest_era5.py` | Persistent per-month NetCDF cache; cache hits skip CDS re-download |
| `data/raw/worldpop_bgd_{year}.tif` | `ingest_population.py` | WorldPop GeoTIFF cache |

## Redis keys

| Key pattern | TTL | Producer | Description |
|---|---|---|---|
| `dengue:risk:current` | 7 days | `predict.py` | JSON-serialized current-week GeoJSON |
| `dengue:forecast:{year}:{week}` | 7 days | `predict.py` | JSON-serialized forecast for the given (year, week) |
