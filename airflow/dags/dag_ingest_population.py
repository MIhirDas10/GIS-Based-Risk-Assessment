"""
airflow/dags/dag_ingest_population.py

Airflow DAG — ingest WorldPop Bangladesh population data.

Schedule: @once  — population is static, only needs to run once.
Re-trigger manually if WorldPop releases a new year's data.
"""

import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow/src")
from ingestion.ingest_population import run as population_run

default_args = {
    "owner": "dengue_team",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="ingest_worldpop_population",
    default_args=default_args,
    description="Download WorldPop Bangladesh GeoTIFF and compute district-level population stats",
    schedule_interval="@once",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ingestion", "population", "static"],
) as dag:

    ingest_population = PythonOperator(
        task_id="ingest_worldpop_2020",
        python_callable=population_run,
        op_kwargs={"year": 2020},
        execution_timeout=timedelta(hours=1),
        doc_md="""
        ### WorldPop population ingestion
        Downloads the WorldPop Bangladesh 1km aggregated GeoTIFF for 2020,
        runs rasterio zonal statistics per district polygon, and upserts
        total population + density into `geo.district_population`.

        Safe to re-run — uses ON CONFLICT DO UPDATE.
        """,
    )