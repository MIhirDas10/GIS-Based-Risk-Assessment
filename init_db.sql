-- Enable PostGIS
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

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
    ingested_at    TIMESTAMP DEFAULT NOW()
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

-- Feature mart
CREATE TABLE IF NOT EXISTS features.mart (
    district_id        INT,
    year               INT,
    week               INT,
    temp_mean_c        FLOAT,
    rainfall_mm        FLOAT,
    humidity_pct       FLOAT,
    rainfall_lag_2w    FLOAT,
    rainfall_lag_4w    FLOAT,
    temp_lag_2w        FLOAT,
    temp_rolling_4w    FLOAT,
    humidity_x_temp    FLOAT,
    cases_spatial_lag  FLOAT,
    week_sin           FLOAT,
    week_cos           FLOAT,
    population_density FLOAT,
    hotspot_rank       FLOAT,
    dengue_cases       INT,
    cases_per_100k     FLOAT,
    PRIMARY KEY (district_id, year, week)
);

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
    predicted_at    TIMESTAMP DEFAULT NOW()
);