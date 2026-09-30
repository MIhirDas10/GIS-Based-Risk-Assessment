"""
tests/test_db_state.py

Integration tests against the live PostGIS instance — verify the data
ingested overnight is shaped correctly and within climatologically
plausible ranges. These will fail if the schema drifts or if a future
ingest run produces obviously broken values.
"""
import pytest

# Every test here queries the live PostGIS instance.
pytestmark = pytest.mark.integration


def _scalar(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()[0]


def _all(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------

def test_postgis_extension_installed(pg_conn):
    extname = _scalar(pg_conn, "SELECT extname FROM pg_extension WHERE extname='postgis'")
    assert extname == "postgis"


@pytest.mark.parametrize("schema", ["disease", "weather", "geo", "features", "monitoring"])
def test_schemas_exist(pg_conn, schema):
    found = _scalar(
        pg_conn,
        f"SELECT schema_name FROM information_schema.schemata WHERE schema_name='{schema}'",
    )
    assert found == schema


def test_mlflow_db_exists(pg_conn):
    found = _scalar(pg_conn, "SELECT datname FROM pg_database WHERE datname='mlflow_db'")
    assert found == "mlflow_db", "mlflow_db must exist for the MLflow server"


def test_disease_unique_constraint_present(pg_conn):
    """Re-running ingest_disease must be idempotent — needs the UNIQUE."""
    n = _scalar(
        pg_conn,
        """
        SELECT COUNT(*) FROM pg_constraint c
        JOIN pg_class t ON c.conrelid = t.oid
        JOIN pg_namespace ns ON t.relnamespace = ns.oid
        WHERE ns.nspname='disease' AND t.relname='dengue_cases' AND c.contype='u'
        """,
    )
    assert n >= 1, "disease.dengue_cases needs a UNIQUE(district_id,year,month,week)"


# ---------------------------------------------------------------------------
# Districts
# ---------------------------------------------------------------------------

def test_64_districts_with_geometry(pg_conn):
    n_total = _scalar(pg_conn, "SELECT COUNT(*) FROM geo.districts")
    n_geom = _scalar(pg_conn, "SELECT COUNT(*) FROM geo.districts WHERE geometry IS NOT NULL")
    assert n_total == 64, f"Expected 64 Bangladesh districts, got {n_total}"
    assert n_geom == 64, "Every district must have geometry"


def test_districts_in_wgs84(pg_conn):
    srid = _scalar(
        pg_conn,
        "SELECT ST_SRID(geometry) FROM geo.districts WHERE geometry IS NOT NULL LIMIT 1",
    )
    assert srid == 4326, "Geometry must be EPSG:4326 for GeoJSON output"


# ---------------------------------------------------------------------------
# Disease cases
# ---------------------------------------------------------------------------

def test_disease_cases_present(pg_conn):
    n = _scalar(pg_conn, "SELECT COUNT(*) FROM disease.dengue_cases")
    # 8 districts × 52 weeks × 5 years ≈ ~850 (some weeks span months so a bit fewer)
    assert n >= 800, f"Expected ~850 weekly disease records, got {n}"
    assert n <= 1000, f"Too many disease rows ({n}) — possible duplication"


def test_disease_covers_full_year_range(pg_conn):
    years = [r[0] for r in _all(pg_conn, "SELECT DISTINCT year FROM disease.dengue_cases ORDER BY year")]
    assert years == [2019, 2020, 2021, 2022, 2023]


def test_disease_no_null_targets(pg_conn):
    n_null = _scalar(pg_conn, "SELECT COUNT(*) FROM disease.dengue_cases WHERE dengue_cases IS NULL")
    assert n_null == 0


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

def test_population_all_64_districts(pg_conn):
    n = _scalar(pg_conn, "SELECT COUNT(*) FROM geo.district_population WHERE year=2020")
    assert n == 64


def test_population_density_plausible(pg_conn):
    avg, mn, mx = _all(
        pg_conn,
        "SELECT AVG(population_density), MIN(population_density), MAX(population_density) FROM geo.district_population",
    )[0]
    # Bangladesh district densities range from ~400 (Khagrachhari) to ~50k (Dhaka)
    assert 300 < float(avg) < 5000, f"Avg density {avg} outside plausible range"
    assert float(mn) > 0
    assert float(mx) > 1000


# ---------------------------------------------------------------------------
# Weather (real ERA5 — overnight)
# ---------------------------------------------------------------------------

def test_era5_full_grid_present(pg_conn):
    """Should have approximately 64 districts × 52 weeks × 5 years ≈ 16,640
    rows, plus partial current year (2026 Jan–Feb)."""
    n = _scalar(pg_conn, "SELECT COUNT(*) FROM weather.era5_district_weekly")
    assert n >= 16640, f"Expected ≥16640 ERA5 weekly rows, got {n}"


def test_era5_covers_full_year_range(pg_conn):
    years = [r[0] for r in _all(
        pg_conn,
        "SELECT DISTINCT year FROM weather.era5_district_weekly ORDER BY year",
    )]
    for y in (2019, 2020, 2021, 2022, 2023):
        assert y in years, f"Missing ERA5 year {y}"


def test_era5_climatologically_plausible(pg_conn):
    """Sanity-check Bangladesh weather values against real ERA5."""
    row = _all(
        pg_conn,
        """SELECT
             AVG(temp_mean_c), MIN(temp_mean_c), MAX(temp_mean_c),
             AVG(humidity_pct), MIN(humidity_pct), MAX(humidity_pct),
             AVG(rainfall_mm), MAX(rainfall_mm)
           FROM weather.era5_district_weekly
           WHERE year BETWEEN 2019 AND 2023""",
    )[0]
    avg_t, min_t, max_t, avg_h, min_h, max_h, avg_r, max_r = [float(v) for v in row]
    # Bangladesh annual mean temp ~ 25°C; winter lows around 10°C, summer highs ~35°C
    assert 22 <= avg_t <= 28, f"avg temp {avg_t} outside Bangladesh range"
    assert 5 <= min_t <= 20, f"min temp {min_t} suspicious"
    assert 28 <= max_t <= 40, f"max temp {max_t} suspicious"
    # Humidity 60-95% in tropics
    assert 60 <= avg_h <= 90, f"avg humidity {avg_h} suspicious"
    assert min_h >= 0 and max_h <= 100
    # Rainfall can spike during monsoon
    assert avg_r > 30, f"avg weekly rain {avg_r} too dry for Bangladesh"
    assert max_r > 200, f"max weekly rain {max_r} too low — monsoon should hit 500-1000+"


# ---------------------------------------------------------------------------
# Feature mart
# ---------------------------------------------------------------------------

def test_feature_mart_populated(pg_conn):
    n = _scalar(pg_conn, "SELECT COUNT(*) FROM features.mart")
    # 8 case-bearing districts × ~92 weeks each
    assert 500 <= n <= 2000, f"Feature mart has {n} rows — outside expected range"


def test_feature_mart_no_pk_duplicates(pg_conn):
    """The (district_id, year, week) PK must hold — fails if build_features
    re-introduces the week-spanning-month bug we already fixed."""
    n_dup = _scalar(
        pg_conn,
        """SELECT COUNT(*) FROM (
             SELECT district_id, year, week, COUNT(*) c
             FROM features.mart GROUP BY 1,2,3 HAVING COUNT(*) > 1
           ) t""",
    )
    assert n_dup == 0


def test_feature_mart_critical_columns_not_null(pg_conn):
    """Targets and current-week weather must always be present."""
    for col in ("dengue_cases", "temp_mean_c", "rainfall_mm", "humidity_pct"):
        n_null = _scalar(pg_conn, f"SELECT COUNT(*) FROM features.mart WHERE {col} IS NULL")
        assert n_null == 0, f"{col} has {n_null} NULLs"


def test_feature_mart_lag_nulls_only_at_year_start(pg_conn):
    """rainfall_lag_2w nulls should only be 1-3% (first 2 weeks per district)."""
    total = _scalar(pg_conn, "SELECT COUNT(*) FROM features.mart")
    n_null = _scalar(pg_conn, "SELECT COUNT(*) FROM features.mart WHERE rainfall_lag_2w IS NULL")
    pct = n_null / total * 100
    assert pct < 10, f"rainfall_lag_2w {pct:.1f}% null — too high"
