"""Smoke test process_netcdf on a real CDS-beta NetCDF file."""
import sys
from pathlib import Path

sys.path.insert(0, "/opt/airflow/src")

import geopandas as gpd

from ingestion.ingest_era5 import process_netcdf

gdf = gpd.read_file("/opt/airflow/data/raw/boundaries/bgd_admin2.shp").to_crs("EPSG:4326")
gdf = gdf.rename(columns={"adm2_name": "district_name"})[["district_name", "geometry"]]

recs = process_netcdf(
    Path("/opt/airflow/data/raw/era5_cache/2019/era5_bangladesh_2019_07.nc"), gdf
)
print(f"GOT {len(recs)} records (expected ~5 weeks * 64 districts = ~320)")
for r in recs[:3]:
    print(r)
