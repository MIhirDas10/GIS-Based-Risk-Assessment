"""
src/ingestion/ingest_population.py

Downloads WorldPop Bangladesh population GeoTIFF (100m resolution),
computes zonal statistics (total population + density) per district
polygon, and inserts into geo.district_population.

This is a @once DAG — population data is static and only needs to run
once (or re-run if WorldPop releases a new year).

Source: https://hub.worldpop.org/geodata/country?iso3=BGD
File:   BGD_ppp_2020_1km_Aggregated.tif  (1km aggregated, ~10 MB)

Usage (standalone):
    python ingest_population.py --year 2020

Usage (via Airflow):
    Called by dag_ingest_population.py once on first deploy.
"""

import argparse
import logging
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import geopandas as gpd
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
log = logging.getLogger("ingest_population")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# WorldPop direct download URLs for Bangladesh (unconstrained, 1km aggregated)
# Using 2020 as the reference year — most complete dataset for Bangladesh
WORLDPOP_URLS = {
    2015: "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/2015/BGD/bgd_ppp_2015_1km_Aggregated.tif",
    2020: "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/2020/BGD/bgd_ppp_2020_1km_Aggregated.tif",
}

# Default year to use
DEFAULT_YEAR = 2020

# Boundaries shapefile path (inside container)
BOUNDARIES_SHP = os.getenv(
    "BOUNDARIES_SHP_PATH",
    "/opt/airflow/data/raw/boundaries/bgd_admin2.shp",
)

# Local cache path — avoid re-downloading if already present
POPULATION_TIF_PATH = os.getenv(
    "POPULATION_TIF_PATH",
    "/opt/airflow/data/raw/worldpop_bgd_{year}.tif",
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


def get_district_map(conn) -> dict[str, int]:
    """Returns {district_name_lower: district_id} from geo.districts."""
    with conn.cursor() as cur:
        cur.execute("SELECT district_id, district_name FROM geo.districts;")
        rows = cur.fetchall()
    return {name.lower(): did for did, name in rows}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_worldpop(year: int, out_path: Path) -> Path:
    """
    Download WorldPop GeoTIFF for Bangladesh.
    Skips download if file already exists at out_path.
    Returns path to the downloaded file.
    """
    if out_path.exists():
        log.info("WorldPop file already exists, skipping download: %s", out_path)
        return out_path

    url = WORLDPOP_URLS.get(year)
    if url is None:
        raise ValueError(
            f"No WorldPop URL configured for year {year}. "
            f"Available years: {list(WORLDPOP_URLS.keys())}"
        )

    log.info("Downloading WorldPop Bangladesh %d from %s …", year, url)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def progress(block_num, block_size, total_size):
        if total_size > 0 and block_num % 50 == 0:
            pct = min(100, block_num * block_size * 100 / total_size)
            log.info("  %.0f%%", pct)

    urllib.request.urlretrieve(url, str(out_path), reporthook=progress)
    log.info(
        "Download complete: %s (%.1f MB)",
        out_path,
        out_path.stat().st_size / 1e6,
    )
    return out_path


# ---------------------------------------------------------------------------
# Zonal statistics
# ---------------------------------------------------------------------------

def zonal_population_stats(
    tif_path: Path,
    district_geom,
    pixel_area_km2: float,
) -> dict:
    """
    Compute total population and population density for one district polygon.

    WorldPop pixel values = estimated number of people in that pixel.
    Sum of pixels = total population in district.
    Density = total_population / district_area_km2.

    Parameters
    ----------
    tif_path        : path to WorldPop GeoTIFF
    district_geom   : shapely geometry of the district (EPSG:4326)
    pixel_area_km2  : area of one pixel in km² (used for sanity check only)

    Returns
    -------
    dict with keys: total_population, population_density
    """
    geojson_geom = [mapping(district_geom)]

    with rasterio.open(str(tif_path)) as src:
        try:
            out_image, _ = rasterio_mask(
                src,
                geojson_geom,
                crop=True,
                nodata=src.nodata,
                all_touched=True,
            )
        except ValueError:
            # District polygon doesn't overlap raster
            log.warning("District geometry does not overlap raster — returning 0")
            return {"total_population": 0, "population_density": 0.0}

    pixels = out_image[0].astype(np.float64)

    # Mask out nodata values (WorldPop uses -99999 or nan)
    nodata = src.nodata if src.nodata is not None else -99999
    valid_mask = (pixels != nodata) & (~np.isnan(pixels)) & (pixels >= 0)
    valid_pixels = pixels[valid_mask]

    total_population = int(np.sum(valid_pixels))
    return {
        "total_population": total_population,
        # density will be computed using the actual district area from DB
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def compute_district_populations(
    tif_path: Path,
    districts_gdf: gpd.GeoDataFrame,
) -> list[dict]:
    """
    Run zonal stats for all 64 districts.
    Returns list of dicts: {district_name, total_population, area_km2}
    """
    log.info("Computing zonal population statistics for %d districts …", len(districts_gdf))

    with rasterio.open(str(tif_path)) as src:
        transform = src.transform
        # Approximate pixel area in km² (WorldPop 1km grid)
        pixel_width_deg  = abs(transform.a)
        pixel_height_deg = abs(transform.e)
        # At Bangladesh's latitude (~23.5°N), 1° lon ≈ 102 km, 1° lat ≈ 111 km
        pixel_area_km2 = pixel_width_deg * 102 * pixel_height_deg * 111

    records = []
    for idx, row in districts_gdf.iterrows():
        dname = row["district_name"]
        geom  = row["geometry"]
        area_km2 = row.get("area_km2", None)

        stats = zonal_population_stats(tif_path, geom, pixel_area_km2)
        total_pop = stats["total_population"]

        # Compute density using actual district area
        if area_km2 and area_km2 > 0:
            density = round(total_pop / area_km2, 2)
        else:
            density = None

        log.info(
            "  %-20s  pop=%8d  area=%7.1f km²  density=%6.1f /km²",
            dname, total_pop, area_km2 or 0, density or 0,
        )

        records.append({
            "district_name":      dname,
            "total_population":   total_pop,
            "population_density": density,
            "area_km2":           area_km2,
        })

    log.info("Computed population stats for %d districts", len(records))
    return records


# ---------------------------------------------------------------------------
# Database insertion
# ---------------------------------------------------------------------------

def insert_population(
    conn,
    records: list[dict],
    district_map: dict[str, int],
    year: int,
) -> int:
    """
    Bulk-insert population records into geo.district_population.
    Uses ON CONFLICT DO UPDATE so re-running is safe.
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
            rec["total_population"],
            rec["population_density"],
            year,
        ))

    if skipped:
        log.warning("Skipped %d records with unrecognised district names", skipped)

    if not rows:
        log.warning("Nothing to insert")
        return 0

    sql = """
        INSERT INTO geo.district_population
            (district_id, population, population_density, year)
        VALUES %s
        ON CONFLICT (district_id, year)
        DO UPDATE SET
            population          = EXCLUDED.population,
            population_density  = EXCLUDED.population_density;
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=100)
    conn.commit()

    log.info("Inserted/updated %d rows into geo.district_population", len(rows))
    return len(rows)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run(year: int = DEFAULT_YEAR):
    """Full pipeline: download → zonal stats → insert."""
    log.info("=== WorldPop ingestion starting for year %d ===", year)

    # 1. Resolve output path for the GeoTIFF
    tif_path = Path(POPULATION_TIF_PATH.format(year=year))

    # 2. Download (skips if already present)
    tif_path = download_worldpop(year, tif_path)

    # 3. Load district boundaries
    log.info("Loading district boundaries from %s", BOUNDARIES_SHP)
    districts_gdf = gpd.read_file(BOUNDARIES_SHP).to_crs("EPSG:4326")

    # Normalise name column
    name_candidates = ["ADM2_EN", "NAME_2", "DIST_NAME", "district", "name"]
    name_col = next((c for c in name_candidates if c in districts_gdf.columns), None)
    if name_col is None:
        raise ValueError(
            f"Cannot find district name column. Available: {list(districts_gdf.columns)}"
        )
    districts_gdf = districts_gdf.rename(columns={name_col: "district_name"})

    # Pull area_km2 from DB if not in shapefile (computed during boundaries ingestion)
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT district_name, area_km2 FROM geo.districts;")
            area_map = {name.lower(): area for name, area in cur.fetchall()}

        districts_gdf["area_km2"] = districts_gdf["district_name"].apply(
            lambda n: area_map.get(n.lower())
        )

        log.info("Loaded %d district polygons", len(districts_gdf))

        # 4. Compute zonal statistics
        records = compute_district_populations(tif_path, districts_gdf)

        # 5. Insert into DB
        district_map = get_district_map(conn)
        inserted = insert_population(conn, records, district_map, year)

        log.info("=== WorldPop ingestion complete — %d rows inserted ===", inserted)

    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest WorldPop population data for Bangladesh"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        choices=list(WORLDPOP_URLS.keys()),
        help=f"WorldPop reference year (default: {DEFAULT_YEAR})",
    )
    args = parser.parse_args()
    run(args.year)