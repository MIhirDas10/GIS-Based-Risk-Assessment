"""
src/ingestion/ingest_era5.py

Downloads ERA5-Land hourly data from Copernicus CDS for Bangladesh,
performs zonal statistics per district polygon, aggregates to weekly
means/sums, and bulk-inserts into weather.era5_district_weekly.

Usage (standalone):
    python ingest_era5.py --year 2019

Usage (via Airflow):
    Called by dag_ingest_era5.py, which passes year as an argument.

Dependencies (already in requirements.txt):
    cdsapi, netCDF4, rasterio, geopandas, numpy, psycopg2-binary, shapely
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import cdsapi
import geopandas as gpd
import netCDF4 as nc
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import rasterio
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import mapping

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("ingest_era5")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Bangladesh bounding box (lat: 20.5–26.7, lon: 88.0–92.7)
# ERA5 uses [N, W, S, E] order
BBOX = [26.7, 88.0, 20.5, 92.7]

# ERA5-Land variables we need
ERA5_VARIABLES = [
    "2m_temperature",          # → temp_mean_c, temp_max_c
    "total_precipitation",     # → rainfall_mm
    "2m_dewpoint_temperature", # → used to derive relative humidity
]

# Boundaries shapefile path (inside container)
BOUNDARIES_SHP = os.getenv(
    "BOUNDARIES_SHP_PATH",
    "/opt/airflow/data/raw/boundaries/bgd_admin2.shp",
)

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db_conn():
    """Return a psycopg2 connection using env vars."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "dengue_db"),
        user=os.getenv("POSTGRES_USER", "dengue_admin"),
        password=os.getenv("POSTGRES_PASSWORD", "dengue_pass_2024"),
    )


def get_district_map(conn):
    """
    Returns {district_name_lower: district_id} from geo.districts.
    Used to map shapefile names to DB ids.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT district_id, district_name FROM geo.districts;")
        rows = cur.fetchall()
    return {name.lower(): did for did, name in rows}


# ---------------------------------------------------------------------------
# ERA5 download
# ---------------------------------------------------------------------------

def download_era5(year: int, tmp_dir: str) -> Path:
    """
    Download ERA5-Land hourly data for Bangladesh for a given year.
    Returns path to the downloaded NetCDF file.

    Pulls all 12 months at once for the year but only for the Bangladesh
    bounding box — keeps file size manageable (~200–400 MB per year).
    """
    out_path = Path(tmp_dir) / f"era5_bangladesh_{year}.nc"

    if out_path.exists():
        log.info("ERA5 file already exists, skipping download: %s", out_path)
        return out_path

    log.info("Downloading ERA5-Land data for year %d …", year)

    c = cdsapi.Client()  # reads ~/.cdsapirc automatically
    c.retrieve(
        "reanalysis-era5-land",
        {
            "variable": ERA5_VARIABLES,
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],   # all 24 hours
            "area": BBOX,                                   # [N, W, S, E]
            "format": "netcdf",
        },
        str(out_path),
    )

    log.info("Download complete: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def dewpoint_to_rh(temp_k: np.ndarray, dewpoint_k: np.ndarray) -> np.ndarray:
    """
    Approximate relative humidity (%) from temperature and dewpoint (Kelvin).
    Uses the August–Roche–Magnus approximation.
    """
    t_c  = temp_k  - 273.15
    td_c = dewpoint_k - 273.15
    # Magnus formula constants (for liquid water, valid 0–60°C)
    a, b = 17.625, 243.04
    gamma_t  = (a * t_c)  / (b + t_c)
    gamma_td = (a * td_c) / (b + td_c)
    rh = 100.0 * np.exp(gamma_td - gamma_t)
    return np.clip(rh, 0, 100)


# ---------------------------------------------------------------------------
# Zonal statistics
# ---------------------------------------------------------------------------

def zonal_mean(data_2d: np.ndarray, transform, district_geom) -> float:
    """
    Compute the mean of data_2d (2-D array: lat × lon) over a district polygon.
    Returns np.nan if no valid pixels fall inside the polygon.

    Parameters
    ----------
    data_2d   : 2-D numpy array with shape (rows, cols)
    transform : rasterio Affine transform matching data_2d
    district_geom : shapely geometry (any type)
    """
    # rasterio.mask expects a list of GeoJSON-like dicts
    geojson_geom = [mapping(district_geom)]

    # Write data to an in-memory raster so rasterio.mask can clip it
    with rasterio.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=data_2d.shape[0],
            width=data_2d.shape[1],
            count=1,
            dtype=data_2d.dtype,
            crs="EPSG:4326",
            transform=transform,
            nodata=np.nan,
        ) as dataset:
            dataset.write(data_2d.astype(np.float32), 1)

        with memfile.open() as dataset:
            try:
                out_image, _ = rasterio_mask(
                    dataset,
                    geojson_geom,
                    crop=True,
                    nodata=np.nan,
                    all_touched=True,   # include edge pixels
                )
            except ValueError:
                # geometry does not overlap raster extent
                return np.nan

    pixels = out_image[0]
    valid  = pixels[~np.isnan(pixels)]
    return float(np.mean(valid)) if len(valid) > 0 else np.nan


# ---------------------------------------------------------------------------
# NetCDF → weekly district aggregates
# ---------------------------------------------------------------------------

def process_netcdf(nc_path: Path, districts_gdf: gpd.GeoDataFrame) -> list[dict]:
    """
    Read the ERA5 NetCDF file, compute zonal statistics per district per hour,
    then aggregate hourly values to ISO week-level summaries.

    Returns a list of dicts ready for DB insertion:
        {district_name, year, week, temp_mean_c, temp_max_c,
         rainfall_mm, humidity_pct}
    """
    log.info("Opening NetCDF: %s", nc_path)
    ds = nc.Dataset(str(nc_path))

    # ERA5 dimension names
    lats = ds.variables["latitude"][:]   # 1-D
    lons = ds.variables["longitude"][:]  # 1-D
    times = ds.variables["time"]         # hours since 1900-01-01 00:00

    import cftime
    time_vals = nc.num2date(times[:], times.units, times.calendar
                            if hasattr(times, "calendar") else "standard")

    # Variable arrays: shape (time, lat, lon)
    t2m  = ds.variables["t2m"][:]   # 2m temperature (K)
    tp   = ds.variables["tp"][:]    # total precipitation (m/hour → convert to mm)
    d2m  = ds.variables["d2m"][:]   # 2m dewpoint (K)

    # Build rasterio Affine transform from the lat/lon grid
    # ERA5 grid is regular — use spacing of first two cells
    lon_res = float(lons[1] - lons[0])
    lat_res = float(lats[1] - lats[0])   # negative (N→S)
    from rasterio.transform import from_origin
    transform = from_origin(
        west=float(lons[0]) - lon_res / 2,
        north=float(lats[0]) - lat_res / 2,  # lats[0] is northernmost
        xsize=abs(lon_res),
        ysize=abs(lat_res),
    )

    n_times = t2m.shape[0]
    log.info("NetCDF has %d time steps, %d districts to process", n_times, len(districts_gdf))

    # Accumulator: {(district_name, year, isoweek): {lists of hourly values}}
    from collections import defaultdict
    import datetime

    accum: dict = defaultdict(lambda: {
        "temps": [],
        "precip": [],
        "humidity": [],
    })

    for t_idx in range(n_times):
        dt = time_vals[t_idx]
        # Convert cftime → Python datetime → isoweek
        py_dt = datetime.datetime(dt.year, dt.month, dt.day, dt.hour)
        iso = py_dt.isocalendar()          # (year, week, weekday)
        year, week = iso[0], iso[1]

        # Extract 2-D slices for this time step
        temp_2d     = np.array(t2m[t_idx])  # (lat, lon)
        precip_2d   = np.array(tp[t_idx])   # (lat, lon), metres/hour
        dewpt_2d    = np.array(d2m[t_idx])  # (lat, lon)

        # Rh needs temp in K and dewpoint in K
        rh_2d = dewpoint_to_rh(temp_2d, dewpt_2d)

        # Convert precipitation: m/hour → mm
        precip_mm_2d = precip_2d * 1000.0

        for _, district_row in districts_gdf.iterrows():
            dname = district_row["district_name"]
            geom  = district_row["geometry"]

            key = (dname, year, week)
            accum[key]["temps"].append(zonal_mean(temp_2d, transform, geom))
            accum[key]["precip"].append(zonal_mean(precip_mm_2d, transform, geom))
            accum[key]["humidity"].append(zonal_mean(rh_2d, transform, geom))

        if t_idx % 100 == 0:
            log.info("  Processed time step %d / %d", t_idx, n_times)

    ds.close()

    # Aggregate hourly → weekly
    records = []
    for (dname, year, week), vals in accum.items():
        temps    = [v for v in vals["temps"]    if not np.isnan(v)]
        precips  = [v for v in vals["precip"]   if not np.isnan(v)]
        humids   = [v for v in vals["humidity"] if not np.isnan(v)]

        if not temps:
            log.warning("No valid pixels for %s week %d-%d — skipping", dname, year, week)
            continue

        # Temperature: mean and max (K → °C)
        temp_arr   = np.array(temps)
        temp_mean_c = float(np.mean(temp_arr) - 273.15)
        temp_max_c  = float(np.max(temp_arr)  - 273.15)

        # Rainfall: sum of hourly mm over the week
        rainfall_mm = float(np.sum(precips)) if precips else float("nan")

        # Humidity: mean over the week
        humidity_pct = float(np.mean(humids)) if humids else float("nan")

        records.append({
            "district_name": dname,
            "year":          year,
            "week":          week,
            "temp_mean_c":   round(temp_mean_c, 3),
            "temp_max_c":    round(temp_max_c,  3),
            "rainfall_mm":   round(rainfall_mm, 3),
            "humidity_pct":  round(humidity_pct, 3),
        })

    log.info("Produced %d weekly district records", len(records))
    return records


# ---------------------------------------------------------------------------
# Database insertion
# ---------------------------------------------------------------------------

def insert_weather(conn, records: list[dict], district_map: dict[str, int]) -> int:
    """
    Bulk-insert weekly weather records into weather.era5_district_weekly.
    Skips records whose district_name is not in district_map.
    Returns count of rows inserted/updated.
    """
    rows = []
    skipped = 0
    for rec in records:
        did = district_map.get(rec["district_name"].lower())
        if did is None:
            log.warning("Unknown district '%s' — skipping", rec["district_name"])
            skipped += 1
            continue
        rows.append((
            did,
            rec["year"],
            rec["week"],
            rec["temp_mean_c"],
            rec["temp_max_c"],
            rec["rainfall_mm"],
            rec["humidity_pct"],
        ))

    if skipped:
        log.warning("Skipped %d records with unrecognised district names", skipped)

    if not rows:
        log.warning("Nothing to insert — all records skipped")
        return 0

    sql = """
        INSERT INTO weather.era5_district_weekly
            (district_id, year, week, temp_mean_c, temp_max_c,
             rainfall_mm, humidity_pct)
        VALUES %s
        ON CONFLICT (district_id, year, week)
        DO UPDATE SET
            temp_mean_c  = EXCLUDED.temp_mean_c,
            temp_max_c   = EXCLUDED.temp_max_c,
            rainfall_mm  = EXCLUDED.rainfall_mm,
            humidity_pct = EXCLUDED.humidity_pct,
            ingested_at  = NOW();
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=500)
    conn.commit()

    log.info("Inserted/updated %d rows into weather.era5_district_weekly", len(rows))
    return len(rows)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def run(year: int):
    """Full pipeline for one year: download → process → insert."""
    log.info("=== ERA5 ingestion starting for year %d ===", year)

    # 1. Load district boundaries (needed for zonal stats)
    log.info("Loading district boundaries from %s", BOUNDARIES_SHP)
    districts_gdf = gpd.read_file(BOUNDARIES_SHP).to_crs("EPSG:4326")

    # Normalise name column — same logic as ingest_boundaries.py
    name_candidates = ["ADM2_EN", "NAME_2", "DIST_NAME", "district", "name"]
    name_col = next((c for c in name_candidates if c in districts_gdf.columns), None)
    if name_col is None:
        raise ValueError(
            f"Cannot find district name column. Available: {list(districts_gdf.columns)}"
        )
    districts_gdf = districts_gdf.rename(columns={name_col: "district_name"})
    districts_gdf = districts_gdf[["district_name", "geometry"]].copy()

    log.info("Loaded %d district polygons", len(districts_gdf))

    # 2. Download ERA5 (skips if already downloaded)
    with tempfile.TemporaryDirectory(prefix="era5_") as tmp_dir:
        nc_path = download_era5(year, tmp_dir)

        # 3. Process NetCDF → weekly records
        records = process_netcdf(nc_path, districts_gdf)

    # 4. Insert into DB
    conn = get_db_conn()
    try:
        district_map = get_district_map(conn)
        inserted = insert_weather(conn, records, district_map)
        log.info("=== ERA5 ingestion complete for year %d — %d rows ===", year, inserted)
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest ERA5 weather data for one year")
    parser.add_argument(
        "--year", type=int, required=True,
        help="Year to ingest (e.g. 2019). Run once per year, 2019–2023.",
    )
    args = parser.parse_args()

    if args.year < 2019 or args.year > 2023:
        log.error("Year must be between 2019 and 2023 (to match dengue case data range)")
        sys.exit(1)

    run(args.year)