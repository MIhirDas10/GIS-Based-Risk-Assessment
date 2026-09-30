"""
src/features/build_features.py

Builds the ML-ready feature mart (features.mart) by joining all source
tables and engineering lag, rolling, spatial, cyclical, and interaction
features entirely in SQL via PostGIS window functions.

This keeps the transformation logic inside the database — fast, auditable,
and reproducible — rather than loading everything into pandas.

Pipeline steps
--------------
1. Pre-flight checks  — verify source tables have data before starting
2. DDL                — ensure features.mart + indexes exist
3. Truncate           — full rebuild (simpler than incremental at this scale)
4. Feature SQL        — single INSERT...SELECT with 7-step CTE pipeline
5. Post-insert index  — create indexes for downstream query performance
6. Validation         — row counts, null rates, distribution stats
7. Export (optional)  — dump mart to CSV for offline model training

Feature catalogue (20 columns)
-------------------------------
Direct:        temp_mean_c, temp_max_c, rainfall_mm, humidity_pct,
               population_density
Lag:           rainfall_lag_2w, rainfall_lag_4w, temp_lag_2w,
               humidity_lag_2w
Rolling:       temp_rolling_4w, rainfall_rolling_4w
Interaction:   humidity_x_temp, rainfall_x_density
Spatial:       cases_spatial_lag
Cyclical:      week_sin, week_cos, month
Hotspot:       hotspot_rank
Target:        dengue_cases, cases_per_100k

Usage (standalone):
    python build_features.py                  # build only
    python build_features.py --export         # build + export CSV
    python build_features.py --export-only    # skip build, just export

Usage (via Airflow):
    Called by dag_build_features.py weekly.
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import psycopg2

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("build_features")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EXPORT_DIR = os.getenv("FEATURES_EXPORT_DIR", "/opt/airflow/data/processed")
EXPORT_FILENAME = "features_mart.csv"

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

def get_db_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "dengue_db"),
        user=os.getenv("POSTGRES_USER", "dengue_admin"),
        password=os.getenv("POSTGRES_PASSWORD", "dengue_pass_2024"),
    )


# ---------------------------------------------------------------------------
# Step 1 — Pre-flight: verify source tables have data
# ---------------------------------------------------------------------------

PREFLIGHT_QUERIES = {
    "disease.dengue_cases": "SELECT COUNT(*) FROM disease.dengue_cases;",
    "weather.era5_district_weekly": "SELECT COUNT(*) FROM weather.era5_district_weekly;",
    "geo.districts": "SELECT COUNT(*) FROM geo.districts;",
    "geo.district_population": "SELECT COUNT(*) FROM geo.district_population;",
}

def preflight_checks(conn) -> dict:
    """
    Verify each source table has rows. Returns {table: count}.
    Raises RuntimeError if disease or weather tables are empty (no point
    building features without the two core sources).
    """
    log.info("Running pre-flight source checks …")
    counts = {}
    with conn.cursor() as cur:
        for table, sql in PREFLIGHT_QUERIES.items():
            cur.execute(sql)
            count = cur.fetchone()[0]
            counts[table] = count
            status = "OK" if count > 0 else "EMPTY"
            log.info("  %-35s %6d rows  [%s]", table, count, status)

    # Hard fail if core tables are empty
    if counts["disease.dengue_cases"] == 0:
        raise RuntimeError(
            "disease.dengue_cases is empty — run ingest_disease DAG first"
        )
    if counts["weather.era5_district_weekly"] == 0:
        raise RuntimeError(
            "weather.era5_district_weekly is empty — run ingest_era5 DAG first"
        )

    # Soft warning if population is missing (features still build, density = 0)
    if counts["geo.district_population"] == 0:
        log.warning(
            "geo.district_population is empty — population_density will be 0 "
            "for all rows. Run ingest_population DAG to fix this."
        )

    log.info("Pre-flight checks passed")
    return counts


# ---------------------------------------------------------------------------
# Step 2 — DDL: create table + indexes
# ---------------------------------------------------------------------------

DDL_FEATURES_MART = """
CREATE TABLE IF NOT EXISTS features.mart (
    district_id         INTEGER  NOT NULL REFERENCES geo.districts(district_id),
    year                INTEGER  NOT NULL,
    week                INTEGER  NOT NULL,

    -- Direct weather features
    temp_mean_c         FLOAT,
    temp_max_c          FLOAT,
    rainfall_mm         FLOAT,
    humidity_pct        FLOAT,

    -- Lag features (weather N weeks prior)
    rainfall_lag_2w     FLOAT,
    rainfall_lag_4w     FLOAT,
    temp_lag_2w         FLOAT,
    humidity_lag_2w     FLOAT,

    -- Rolling features (sustained conditions)
    temp_rolling_4w     FLOAT,
    rainfall_rolling_4w FLOAT,

    -- Interaction features
    humidity_x_temp     FLOAT,   -- heat-humidity stress index proxy
    rainfall_x_density  FLOAT,   -- high rain in dense areas = breeding risk

    -- Spatial features
    cases_spatial_lag   FLOAT,   -- mean cases in adjacent districts

    -- Cyclical encoding
    week_sin            FLOAT,
    week_cos            FLOAT,
    month               INTEGER, -- month number (1-12) for interpretability

    -- Population
    population_density  FLOAT,

    -- Hotspot rank (0-1 percentile)
    hotspot_rank        FLOAT,

    -- Target variables
    dengue_cases        INTEGER,
    cases_per_100k      FLOAT,

    PRIMARY KEY (district_id, year, week)
);
"""

DDL_INDEXES = """
-- Index for temporal queries (model training splits by year)
CREATE INDEX IF NOT EXISTS idx_mart_year
    ON features.mart (year);

-- Index for district lookups (dashboard per-district views)
CREATE INDEX IF NOT EXISTS idx_mart_district
    ON features.mart (district_id);

-- Composite index for the most common query pattern
CREATE INDEX IF NOT EXISTS idx_mart_district_year_week
    ON features.mart (district_id, year, week);
"""


# ---------------------------------------------------------------------------
# Step 4 — Feature SQL (7-step CTE pipeline)
# ---------------------------------------------------------------------------

FEATURE_SQL = """
INSERT INTO features.mart (
    district_id, year, week,
    temp_mean_c, temp_max_c, rainfall_mm, humidity_pct,
    rainfall_lag_2w, rainfall_lag_4w, temp_lag_2w, humidity_lag_2w,
    temp_rolling_4w, rainfall_rolling_4w,
    humidity_x_temp, rainfall_x_density,
    cases_spatial_lag,
    week_sin, week_cos, month,
    population_density,
    hotspot_rank,
    dengue_cases, cases_per_100k
)

WITH

-- ---------------------------------------------------------------
-- CTE 1 — base: join disease + weather + population
--
-- INNER JOIN on weather means we only produce rows where both
-- case data AND ERA5 data exist. This is intentional: a row
-- without weather features would be useless for training.
--
-- LEFT JOIN on population because it's static supplementary
-- data — missing population just means density=0, which the
-- model can still learn from.
-- ---------------------------------------------------------------
base AS (
    SELECT
        dc.district_id,
        dc.year,
        dc.week,
        dc.month,

        -- Weather (direct pass-through)
        w.temp_mean_c,
        w.temp_max_c,
        w.rainfall_mm,
        w.humidity_pct,

        -- Population (2020 reference year for all rows)
        COALESCE(dp.population_density, 0) AS population_density,

        -- Target
        dc.dengue_cases,
        dc.cases_per_100k

    FROM disease.dengue_cases dc

    -- Only rows where ERA5 weather exists
    INNER JOIN weather.era5_district_weekly w
        ON w.district_id = dc.district_id
        AND w.year       = dc.year
        AND w.week       = dc.week

    -- Population is optional (static)
    LEFT JOIN geo.district_population dp
        ON dp.district_id = dc.district_id
        AND dp.year       = 2020
),

-- ---------------------------------------------------------------
-- CTE 2 — lag_features: temporal lags via SQL window functions
--
-- WHY LAGS MATTER FOR DENGUE:
--   Rainfall 2-4 weeks ago creates standing water → mosquito
--   breeding → biting → incubation → symptom onset → reporting.
--   The biological chain is 2-4 weeks, so rainfall_lag_2w and
--   rainfall_lag_4w are the strongest predictors in most dengue
--   literature.
--
-- PARTITION BY district_id: lags never bleed across districts.
-- ORDER BY year, week: correct temporal ordering. Even at year
-- boundaries (week 52 of 2020 → week 1 of 2021) the combination
-- of year+week sorts correctly.
--
-- FIRST 2/4 ROWS PER DISTRICT: will be NULL (no prior data).
-- This is expected and the validation step checks the rate.
-- ---------------------------------------------------------------
lag_features AS (
    SELECT
        *,

        -- Rainfall 2 weeks ago (main mosquito breeding signal)
        LAG(rainfall_mm, 2) OVER w_district AS rainfall_lag_2w,

        -- Rainfall 4 weeks ago (peak breeding-to-biting lag)
        LAG(rainfall_mm, 4) OVER w_district AS rainfall_lag_4w,

        -- Temperature 2 weeks ago (affects mosquito development speed)
        LAG(temp_mean_c, 2) OVER w_district AS temp_lag_2w,

        -- Humidity 2 weeks ago (affects mosquito survival)
        LAG(humidity_pct, 2) OVER w_district AS humidity_lag_2w,

        -- 4-week rolling mean temperature (sustained heat matters more
        -- than a single hot week — mosquitoes need consistent warmth)
        AVG(temp_mean_c) OVER (
            PARTITION BY district_id
            ORDER BY year, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS temp_rolling_4w,

        -- 4-week rolling total rainfall (cumulative water accumulation)
        SUM(rainfall_mm) OVER (
            PARTITION BY district_id
            ORDER BY year, week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rainfall_rolling_4w

    FROM base

    -- Named window for cleaner LAG calls
    WINDOW w_district AS (
        PARTITION BY district_id ORDER BY year, week
    )
),

-- ---------------------------------------------------------------
-- CTE 3 — interaction + cyclical features
--
-- INTERACTIONS capture non-linear effects that tree models
-- COULD learn but giving them explicitly helps linear models
-- and speeds up tree convergence:
--
--   humidity_x_temp:     High humidity + high temp = ideal for
--                        Aedes aegypti. Neither alone is enough.
--   rainfall_x_density:  Heavy rain in a dense urban area creates
--                        more breeding sites (clogged drains,
--                        containers) than in rural areas.
--
-- CYCLICAL ENCODING: week 52 should be "close" to week 1, but a
-- raw integer treats them as far apart. Sin/cos mapping puts them
-- on a circle so the model sees the true distance.
-- ---------------------------------------------------------------
enriched AS (
    SELECT
        *,

        -- Heat-humidity interaction
        humidity_pct * temp_mean_c AS humidity_x_temp,

        -- Rainfall × density interaction
        rainfall_mm * population_density AS rainfall_x_density,

        -- Cyclical week encoding
        SIN(2 * PI() * week / 52.0) AS week_sin,
        COS(2 * PI() * week / 52.0) AS week_cos

    FROM lag_features
),

-- ---------------------------------------------------------------
-- CTE 4 — spatial_lag: mean cases in adjacent (touching) districts
--
-- WHY THIS MATTERS: dengue outbreaks spread geographically.
-- If Dhaka has a surge, Gazipur and Narayanganj are next.
-- ST_Touches returns districts that share a boundary line.
--
-- LEFT JOIN ensures districts with no neighbours in the case data
-- (most of the 64 districts have no reported cases) get NULL
-- instead of disappearing from the result set.
-- ---------------------------------------------------------------
spatial_lag AS (
    SELECT
        e.district_id,
        e.year,
        e.week,
        AVG(dc_n.dengue_cases) AS cases_spatial_lag
    FROM enriched e
    INNER JOIN geo.districts self_d
        ON self_d.district_id = e.district_id
    INNER JOIN geo.districts neighbour_d
        ON ST_Touches(self_d.geometry, neighbour_d.geometry)
        AND neighbour_d.district_id != e.district_id
    INNER JOIN disease.dengue_cases dc_n
        ON dc_n.district_id = neighbour_d.district_id
        AND dc_n.year       = e.year
        AND dc_n.week       = e.week
    GROUP BY e.district_id, e.year, e.week
),

-- ---------------------------------------------------------------
-- CTE 5 — hotspot_rank: structural district-level risk
--
-- Some districts are structurally high-risk (Dhaka — dense,
-- poor drainage, urban heat island) regardless of this week's
-- weather. PERCENT_RANK across all districts' historical mean
-- cases gives a 0-to-1 score that captures this.
--
-- This is computed ONCE over all history, not per-year, so it
-- represents the district's long-term risk profile.
-- ---------------------------------------------------------------
hotspot AS (
    SELECT
        district_id,
        PERCENT_RANK() OVER (
            ORDER BY AVG(dengue_cases)
        ) AS hotspot_rank
    FROM disease.dengue_cases
    GROUP BY district_id
)

-- ---------------------------------------------------------------
-- CTE 6 — Final assembly: LEFT JOIN spatial_lag and hotspot
--
-- LEFT JOINs ensure all enriched rows survive even if a district
-- has no spatial neighbours with case data (most won't — we only
-- have 8 districts with case data out of 64).
-- ---------------------------------------------------------------
SELECT
    e.district_id,
    e.year,
    e.week,

    -- Direct weather
    e.temp_mean_c,
    e.temp_max_c,
    e.rainfall_mm,
    e.humidity_pct,

    -- Lags
    e.rainfall_lag_2w,
    e.rainfall_lag_4w,
    e.temp_lag_2w,
    e.humidity_lag_2w,

    -- Rolling
    e.temp_rolling_4w,
    e.rainfall_rolling_4w,

    -- Interactions
    e.humidity_x_temp,
    e.rainfall_x_density,

    -- Spatial (0 if no neighbour data — not NULL, so models don't choke)
    COALESCE(sl.cases_spatial_lag, 0) AS cases_spatial_lag,

    -- Cyclical + calendar
    e.week_sin,
    e.week_cos,
    e.month,

    -- Population
    e.population_density,

    -- Hotspot (0 if district somehow missing from case table)
    COALESCE(h.hotspot_rank, 0) AS hotspot_rank,

    -- Target
    e.dengue_cases,
    e.cases_per_100k

FROM enriched e

LEFT JOIN spatial_lag sl
    ON sl.district_id = e.district_id
    AND sl.year       = e.year
    AND sl.week       = e.week

LEFT JOIN hotspot h
    ON h.district_id = e.district_id

ON CONFLICT (district_id, year, week) DO UPDATE SET
    temp_mean_c         = EXCLUDED.temp_mean_c,
    temp_max_c          = EXCLUDED.temp_max_c,
    rainfall_mm         = EXCLUDED.rainfall_mm,
    humidity_pct        = EXCLUDED.humidity_pct,
    rainfall_lag_2w     = EXCLUDED.rainfall_lag_2w,
    rainfall_lag_4w     = EXCLUDED.rainfall_lag_4w,
    temp_lag_2w         = EXCLUDED.temp_lag_2w,
    humidity_lag_2w     = EXCLUDED.humidity_lag_2w,
    temp_rolling_4w     = EXCLUDED.temp_rolling_4w,
    rainfall_rolling_4w = EXCLUDED.rainfall_rolling_4w,
    humidity_x_temp     = EXCLUDED.humidity_x_temp,
    rainfall_x_density  = EXCLUDED.rainfall_x_density,
    cases_spatial_lag   = EXCLUDED.cases_spatial_lag,
    week_sin            = EXCLUDED.week_sin,
    week_cos            = EXCLUDED.week_cos,
    month               = EXCLUDED.month,
    population_density  = EXCLUDED.population_density,
    hotspot_rank        = EXCLUDED.hotspot_rank,
    dengue_cases        = EXCLUDED.dengue_cases,
    cases_per_100k      = EXCLUDED.cases_per_100k;
"""


# ---------------------------------------------------------------------------
# Step 6 — Validation queries
# ---------------------------------------------------------------------------

VALIDATION_SQL = """
SELECT
    COUNT(*)                                                     AS total_rows,
    COUNT(DISTINCT district_id)                                  AS n_districts,
    COUNT(DISTINCT year)                                         AS n_years,
    MIN(year)                                                    AS year_min,
    MAX(year)                                                    AS year_max,
    MIN(week)                                                    AS week_min,
    MAX(week)                                                    AS week_max,

    -- Target stats
    ROUND(AVG(dengue_cases)::NUMERIC, 2)                         AS avg_cases,
    MAX(dengue_cases)                                            AS max_cases,
    ROUND(STDDEV(dengue_cases)::NUMERIC, 2)                      AS std_cases,

    -- Feature coverage (count of non-NULL values)
    COUNT(rainfall_lag_2w)                                       AS nonnull_lag_2w,
    COUNT(rainfall_lag_4w)                                       AS nonnull_lag_4w,
    COUNT(humidity_lag_2w)                                       AS nonnull_lag_2w_hum,
    COUNT(temp_rolling_4w)                                       AS nonnull_rolling_4w,

    -- Null counts (for diagnosing issues)
    SUM(CASE WHEN rainfall_lag_2w IS NULL THEN 1 ELSE 0 END)     AS null_lag_2w,
    SUM(CASE WHEN rainfall_lag_4w IS NULL THEN 1 ELSE 0 END)     AS null_lag_4w,

    -- Feature ranges (sanity checks)
    ROUND(MIN(temp_mean_c)::NUMERIC, 1)                          AS temp_min,
    ROUND(MAX(temp_mean_c)::NUMERIC, 1)                          AS temp_max,
    ROUND(MIN(rainfall_mm)::NUMERIC, 1)                          AS rain_min,
    ROUND(MAX(rainfall_mm)::NUMERIC, 1)                          AS rain_max,
    ROUND(MIN(humidity_pct)::NUMERIC, 1)                         AS hum_min,
    ROUND(MAX(humidity_pct)::NUMERIC, 1)                         AS hum_max,
    ROUND(AVG(cases_spatial_lag)::NUMERIC, 2)                    AS avg_spatial_lag,
    ROUND(MAX(hotspot_rank)::NUMERIC, 3)                         AS max_hotspot

FROM features.mart;
"""

DISTRIBUTION_SQL = """
-- Per-year row count — helps catch missing years
SELECT year, COUNT(*) AS rows, SUM(dengue_cases) AS total_cases
FROM features.mart
GROUP BY year
ORDER BY year;
"""

DISTRICT_COVERAGE_SQL = """
-- Per-district summary — shows which districts have sparse data
SELECT
    d.district_name,
    COUNT(m.*) AS rows,
    COALESCE(SUM(m.dengue_cases), 0) AS total_cases,
    ROUND(AVG(m.population_density)::NUMERIC, 1) AS pop_density,
    ROUND(AVG(m.hotspot_rank)::NUMERIC, 3) AS hotspot_rank
FROM geo.districts d
LEFT JOIN features.mart m ON m.district_id = d.district_id
WHERE d.district_id IN (SELECT DISTINCT district_id FROM disease.dengue_cases)
GROUP BY d.district_name
ORDER BY total_cases DESC;
"""


# ---------------------------------------------------------------------------
# Step 7 — Export mart to CSV (for offline model training without DB)
# ---------------------------------------------------------------------------

EXPORT_SQL = """
SELECT
    d.district_name,
    m.*
FROM features.mart m
JOIN geo.districts d ON d.district_id = m.district_id
ORDER BY m.district_id, m.year, m.week;
"""


def export_mart(conn, export_dir: str = EXPORT_DIR) -> Path:
    """
    Export features.mart to CSV for offline model training.
    Returns path to the exported file.
    """
    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / EXPORT_FILENAME

    log.info("Exporting features.mart to %s …", out_path)

    with conn.cursor() as cur:
        cur.execute(EXPORT_SQL)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    log.info("Exported %d rows × %d columns to %s", len(rows), len(columns), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(export: bool = False):
    """
    Full feature mart build pipeline:
    1. Pre-flight checks
    2. DDL (create table if not exists)
    3. Truncate (full rebuild)
    4. Feature SQL (CTE pipeline)
    5. Create indexes
    6. Validate
    7. Export (optional)

    Returns row count inserted.
    """
    log.info("=" * 60)
    log.info("FEATURE MART BUILD — STARTING")
    log.info("=" * 60)

    conn = get_db_conn()
    try:
        # 1. Pre-flight
        source_counts = preflight_checks(conn)

        with conn.cursor() as cur:

            # 2. DDL
            log.info("Step 2: Ensuring features.mart table exists …")
            cur.execute(DDL_FEATURES_MART)
            conn.commit()

            # 3. Truncate
            log.info("Step 3: Truncating features.mart for full rebuild …")
            cur.execute("TRUNCATE TABLE features.mart;")
            conn.commit()

            # 4. Feature SQL
            log.info("Step 4: Building feature mart (CTE pipeline) …")
            log.info("  This joins %d case rows with %d weather rows …",
                     source_counts["disease.dengue_cases"],
                     source_counts["weather.era5_district_weekly"])
            cur.execute(FEATURE_SQL)
            rows_inserted = cur.rowcount
            conn.commit()
            log.info("  → Inserted %d rows into features.mart", rows_inserted)

            if rows_inserted == 0:
                log.error(
                    "Zero rows inserted! Most likely cause: disease and weather "
                    "tables have no overlapping (district_id, year, week) keys. "
                    "Check that ERA5 data covers the same years as dengue cases "
                    "(2019–2023)."
                )
                return 0

            # 5. Indexes
            log.info("Step 5: Creating indexes …")
            for stmt in DDL_INDEXES.split(";"):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt + ";")
            conn.commit()

            # 6. Validate
            log.info("Step 6: Validation …")
            cur.execute(VALIDATION_SQL)
            val = cur.fetchone()
            cols = [desc[0] for desc in cur.description]
            result = dict(zip(cols, val))

        log.info("-" * 50)
        log.info("VALIDATION RESULTS")
        log.info("-" * 50)
        for k, v in result.items():
            log.info("  %-25s %s", k, v)

        # Null lag diagnostics
        total = result["total_rows"] or 1
        null_pct_2w = (result["null_lag_2w"] or 0) / total * 100
        null_pct_4w = (result["null_lag_4w"] or 0) / total * 100
        log.info("-" * 50)
        log.info("  NULL rainfall_lag_2w:   %5.1f%%  (expected ~3%% from year boundaries)",
                 null_pct_2w)
        log.info("  NULL rainfall_lag_4w:   %5.1f%%  (expected ~5%% from year boundaries)",
                 null_pct_4w)

        if null_pct_2w > 20:
            log.warning(
                "High null rate on lag features (%.1f%%) — check ERA5 data completeness",
                null_pct_2w,
            )

        # Sanity: Bangladesh temp should be 10-40°C, humidity 30-100%
        temp_min = result.get("temp_min", 0) or 0
        temp_max = result.get("temp_max", 0) or 0
        if temp_min < 0 or temp_max > 50:
            log.warning(
                "Temperature range [%.1f, %.1f]°C looks suspect — "
                "check ERA5 unit conversion (should be °C not K)",
                temp_min, temp_max,
            )

        # Per-year breakdown
        log.info("-" * 50)
        log.info("PER-YEAR BREAKDOWN")
        log.info("-" * 50)
        with conn.cursor() as cur:
            cur.execute(DISTRIBUTION_SQL)
            for row in cur.fetchall():
                log.info("  Year %d: %4d rows, %6d total cases", row[0], row[1], row[2])

        # Per-district breakdown
        log.info("-" * 50)
        log.info("PER-DISTRICT COVERAGE")
        log.info("-" * 50)
        with conn.cursor() as cur:
            cur.execute(DISTRICT_COVERAGE_SQL)
            log.info("  %-20s %5s %8s %8s %7s", "District", "Rows", "Cases", "Density", "Hotspot")
            for row in cur.fetchall():
                log.info("  %-20s %5d %8d %8.1f %7.3f",
                         row[0], row[1], row[2], row[3] or 0, row[4] or 0)

        # 7. Export
        if export:
            export_mart(conn)

        log.info("=" * 60)
        log.info("FEATURE MART BUILD — COMPLETE (%d rows)", rows_inserted)
        log.info("=" * 60)
        return rows_inserted

    except Exception as e:
        conn.rollback()
        log.error("Feature mart build FAILED: %s", e, exc_info=True)
        raise
    finally:
        conn.close()


def run_export_only():
    """Export existing mart to CSV without rebuilding."""
    log.info("Export-only mode — skipping build")
    conn = get_db_conn()
    try:
        export_mart(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the ML feature mart")
    parser.add_argument(
        "--export", action="store_true",
        help="Also export features.mart to CSV after building",
    )
    parser.add_argument(
        "--export-only", action="store_true",
        help="Skip build, just export existing mart to CSV",
    )
    args = parser.parse_args()

    if args.export_only:
        run_export_only()
    else:
        run(export=args.export)