-- Enable PostGIS
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Separate database for MLflow backend store
-- (docker-compose passes POSTGRES_DB=dengue_db which only auto-creates
--  the main DB; MLflow points at mlflow_db so we create it here.
--  init_db.sql only runs on a fresh data dir, so a plain CREATE is safe.)
CREATE DATABASE mlflow_db;

-- Schemas
CREATE SCHEMA IF NOT EXISTS disease;
CREATE SCHEMA IF NOT EXISTS weather;
CREATE SCHEMA IF NOT EXISTS geo;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- District boundaries (spatial)
CREATE TABLE IF NOT EXISTS geo.districts (
    district_id   SERIAL PRIMARY KEY,
    district_name VARCHAR(100) NOT NULL,
    division_name VARCHAR(100),
    geometry      GEOMETRY(MULTIPOLYGON, 4326),
    area_km2      FLOAT,
    created_at    TIMESTAMP DEFAULT NOW()
);

-- Population
CREATE TABLE IF NOT EXISTS geo.district_population (
    district_id        INT REFERENCES geo.districts(district_id),
    population         BIGINT,
    population_density FLOAT,
    year               INT,
    PRIMARY KEY (district_id, year)
);

-- Disease cases
CREATE TABLE IF NOT EXISTS disease.dengue_cases (
    id             SERIAL PRIMARY KEY,
    district_id    INT REFERENCES geo.districts(district_id),
    year           INT NOT NULL,
    month          INT NOT NULL,
    week           INT,
    dengue_cases   INT NOT NULL,
    cases_per_100k FLOAT,
    data_source    VARCHAR(50) DEFAULT 'kaggle',
    ingested_at    TIMESTAMP DEFAULT NOW(),
    UNIQUE (district_id, year, month, week)
);

-- Weather
CREATE TABLE IF NOT EXISTS weather.era5_district_weekly (
    id           SERIAL PRIMARY KEY,
    district_id  INT REFERENCES geo.districts(district_id),
    year         INT NOT NULL,
    week         INT NOT NULL,
    temp_mean_c  FLOAT,
    temp_max_c   FLOAT,
    rainfall_mm  FLOAT,
    humidity_pct FLOAT,
    ingested_at  TIMESTAMP DEFAULT NOW(),
    UNIQUE (district_id, year, week)
);

-- Feature mart + forecast mart: created by src/features/build_features.py.
-- (The schemas live next to the SQL pipeline that owns them so the two
--  can't drift out of sync — earlier versions kept the DDL here and
--  it became stale relative to the INSERT list. Do not re-add them.)
--
-- features.mart           — training data; INNER JOIN on disease.dengue_cases
--                            so it only covers weeks with case ground truth.
-- features.forecast_mart  — serving data; INNER JOIN on weather only, so it
--                            covers every (district, year, week) where ERA5
--                            exists, including weeks past the disease cutoff.

-- Monitoring
CREATE TABLE IF NOT EXISTS monitoring.prediction_log (
    id              SERIAL PRIMARY KEY,
    district_id     INT,
    year            INT,
    week            INT,
    predicted_cases FLOAT,
    actual_cases    INT,
    risk_tier       VARCHAR(20),
    model_version   VARCHAR(50),
    predicted_at    TIMESTAMP DEFAULT NOW(),
    -- predict.py upserts on (district_id, year, week). Constraint defined
    -- here (not at runtime in the script) so the contract lives next to
    -- the schema and Airflow doesn't need ALTER privileges.
    CONSTRAINT uq_prediction_log_district_year_week
        UNIQUE (district_id, year, week)
);