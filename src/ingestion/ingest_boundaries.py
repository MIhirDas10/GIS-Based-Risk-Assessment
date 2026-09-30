import logging
import os

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon

from db import get_db_conn as get_connection  # noqa: E402  (legacy alias)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Map HUMDATA district names → our standardized names
DISTRICT_MAPPING = {
    "Chittagong": "Chittagong",
    "Comilla": "Comilla",
    "Cox's Bazar": "Cox's Bazar",
    "Feni": "Feni",
    "Khagrachhari": "Khagrachhari",
    "Lakshmipur": "Lakshmipur",
    "Noakhali": "Noakhali",
    "Rangamati": "Rangamati",
    "Brahmanbaria": "Brahmanbaria",
    "Chandpur": "Chandpur",
    "Dhaka": "Dhaka",
    "Faridpur": "Faridpur",
    "Gazipur": "Gazipur",
    "Gopalganj": "Gopalganj",
    "Kishoreganj": "Kishoreganj",
    "Madaripur": "Madaripur",
    "Manikganj": "Manikganj",
    "Munshiganj": "Munshiganj",
    "Narayanganj": "Narayanganj",
    "Narsingdi": "Narsingdi",
    "Rajbari": "Rajbari",
    "Shariatpur": "Shariatpur",
    "Tangail": "Tangail",
    "Bagerhat": "Bagerhat",
    "Chuadanga": "Chuadanga",
    "Jessore": "Jessore",
    "Jhenaidah": "Jhenaidah",
    "Khulna": "Khulna",
    "Kushtia": "Kushtia",
    "Magura": "Magura",
    "Meherpur": "Meherpur",
    "Narail": "Narail",
    "Satkhira": "Satkhira",
    "Barguna": "Barguna",
    "Barisal": "Barisal",
    "Bhola": "Bhola",
    "Jhalokati": "Jhalokati",
    "Patuakhali": "Patuakhali",
    "Pirojpur": "Pirojpur",
    "Bandarban": "Bandarban",
    "Habiganj": "Habiganj",
    "Moulvibazar": "Moulvibazar",
    "Sunamganj": "Sunamganj",
    "Sylhet": "Sylhet",
    "Bogra": "Bogra",
    "Joypurhat": "Joypurhat",
    "Naogaon": "Naogaon",
    "Natore": "Natore",
    "Nawabganj": "Nawabganj",
    "Pabna": "Pabna",
    "Rajshahi": "Rajshahi",
    "Sirajganj": "Sirajganj",
    "Dinajpur": "Dinajpur",
    "Gaibandha": "Gaibandha",
    "Kurigram": "Kurigram",
    "Lalmonirhat": "Lalmonirhat",
    "Nilphamari": "Nilphamari",
    "Panchagarh": "Panchagarh",
    "Rangpur": "Rangpur",
    "Thakurgaon": "Thakurgaon",
    "Jamalpur": "Jamalpur",
    "Mymensingh": "Mymensingh",
    "Netrakona": "Netrakona",
    "Sherpur": "Sherpur",
}

# Division mapping by district
DISTRICT_DIVISION_MAP = {
    "Chittagong": "Chittagong", "Comilla": "Chittagong",
    "Cox's Bazar": "Chittagong", "Feni": "Chittagong",
    "Khagrachhari": "Chittagong", "Lakshmipur": "Chittagong",
    "Noakhali": "Chittagong", "Rangamati": "Chittagong",
    "Brahmanbaria": "Chittagong", "Chandpur": "Chittagong",
    "Bandarban": "Chittagong",
    "Dhaka": "Dhaka", "Faridpur": "Dhaka", "Gazipur": "Dhaka",
    "Gopalganj": "Dhaka", "Kishoreganj": "Dhaka", "Madaripur": "Dhaka",
    "Manikganj": "Dhaka", "Munshiganj": "Dhaka", "Narayanganj": "Dhaka",
    "Narsingdi": "Dhaka", "Rajbari": "Dhaka", "Shariatpur": "Dhaka",
    "Tangail": "Dhaka",
    "Bagerhat": "Khulna", "Chuadanga": "Khulna", "Jessore": "Khulna",
    "Jhenaidah": "Khulna", "Khulna": "Khulna", "Kushtia": "Khulna",
    "Magura": "Khulna", "Meherpur": "Khulna", "Narail": "Khulna",
    "Satkhira": "Khulna",
    "Barguna": "Barisal", "Barisal": "Barisal", "Bhola": "Barisal",
    "Jhalokati": "Barisal", "Patuakhali": "Barisal", "Pirojpur": "Barisal",
    "Habiganj": "Sylhet", "Moulvibazar": "Sylhet",
    "Sunamganj": "Sylhet", "Sylhet": "Sylhet",
    "Bogra": "Rajshahi", "Joypurhat": "Rajshahi", "Naogaon": "Rajshahi",
    "Natore": "Rajshahi", "Nawabganj": "Rajshahi", "Pabna": "Rajshahi",
    "Rajshahi": "Rajshahi", "Sirajganj": "Rajshahi",
    "Dinajpur": "Rangpur", "Gaibandha": "Rangpur", "Kurigram": "Rangpur",
    "Lalmonirhat": "Rangpur", "Nilphamari": "Rangpur", "Panchagarh": "Rangpur",
    "Rangpur": "Rangpur", "Thakurgaon": "Rangpur",
    "Jamalpur": "Mymensingh", "Mymensingh": "Mymensingh",
    "Netrakona": "Mymensingh", "Sherpur": "Mymensingh",
}


def to_multipolygon(geom):
    # ensuring that geometry is always a MultiPolygon
    if geom is None:
        return None
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    return geom # else if it's Multipolygon


# Candidate name columns across vintages of the HDX shapefile.
# 2023+ release uses lowercase adm2_name; older releases used ADM2_EN.
NAME_COL_CANDIDATES = ["adm2_name", "ADM2_EN", "NAME_2", "DIST_NAME", "district", "name"]
DIVISION_COL_CANDIDATES = ["adm1_name", "ADM1_EN", "NAME_1", "division"]
AREA_COL_CANDIDATES = ["area_sqkm", "AREA_SQKM", "AREA_KM2"]

SHAPEFILE_PATH = os.getenv(
    "BOUNDARIES_SHP_PATH",
    "/opt/airflow/data/raw/boundaries/bgd_admin2.shp",
)


def _pick_col(gdf, candidates, role):
    for c in candidates:
        if c in gdf.columns:
            return c
    raise ValueError(
        f"Could not find {role} column in shapefile. "
        f"Tried {candidates}, available: {list(gdf.columns)}"
    )


def run():
    """
    Load bgd_admin2 shapefile (64 districts), reproject to EPSG:4326,
    convert to MultiPolygon, and upsert into geo.districts keyed by
    district_name. Re-running is safe — geometry/division/area are updated
    in place.
    """
    logger.info("Loading boundaries from %s", SHAPEFILE_PATH)
    gdf = gpd.read_file(SHAPEFILE_PATH)

    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        logger.info("Reprojecting %s -> EPSG:4326", gdf.crs)
        gdf = gdf.to_crs("EPSG:4326")

    name_col = _pick_col(gdf, NAME_COL_CANDIDATES, "district name")
    div_col = _pick_col(gdf, DIVISION_COL_CANDIDATES, "division name")
    area_col = next((c for c in AREA_COL_CANDIDATES if c in gdf.columns), None)

    logger.info(
        "Shapefile columns: name=%s division=%s area=%s rows=%d",
        name_col, div_col, area_col, len(gdf),
    )

    rows = []
    for _, r in gdf.iterrows():
        raw_name = (r[name_col] or "").strip()
        if not raw_name:
            continue
        std_name = DISTRICT_MAPPING.get(raw_name, raw_name)
        div = DISTRICT_DIVISION_MAP.get(std_name) or (r[div_col] or "").strip()
        geom = to_multipolygon(r["geometry"])
        if geom is None or geom.is_empty:
            logger.warning("Empty geometry for %s — skipping", std_name)
            continue
        area_km2 = float(r[area_col]) if area_col and r[area_col] is not None else None
        rows.append((std_name, div, geom.wkt, area_km2))

    if not rows:
        raise RuntimeError("No district rows produced from shapefile — check column names")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # First-time insert (table is empty on a fresh DB); if rows exist,
            # update geometry/division/area in place keyed by district_name.
            # We can't use ON CONFLICT because there's no unique index on name.
            for std_name, div, wkt, area_km2 in rows:
                cur.execute(
                    """
                    INSERT INTO geo.districts (district_name, division_name, geometry, area_km2)
                    SELECT %s, %s, ST_Multi(ST_GeomFromText(%s, 4326))::geometry(MultiPolygon, 4326), %s
                    WHERE NOT EXISTS (
                        SELECT 1 FROM geo.districts WHERE district_name = %s
                    )
                    """,
                    (std_name, div, wkt, area_km2, std_name),
                )
                cur.execute(
                    """
                    UPDATE geo.districts
                    SET division_name = %s,
                        geometry = ST_Multi(ST_GeomFromText(%s, 4326))::geometry(MultiPolygon, 4326),
                        area_km2 = COALESCE(%s, area_km2)
                    WHERE district_name = %s
                    """,
                    (div, wkt, area_km2, std_name),
                )
        conn.commit()
        logger.info("Loaded %d district boundaries into geo.districts", len(rows))
    finally:
        conn.close()
    return len(rows)


if __name__ == "__main__":
    run()
