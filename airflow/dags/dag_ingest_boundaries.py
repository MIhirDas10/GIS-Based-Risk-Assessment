from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys

sys.path.insert(0, "/opt/airflow/src")

from ingestion.ingest_boundaries import run as ingest_boundaries

default_args = {
    "owner": "dengue_team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ingest_boundaries",
    description="Load Bangladesh district boundaries into PostGIS",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@once",
    catchup=False,
    tags=["ingestion", "geo"],
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_district_boundaries",
        python_callable=ingest_boundaries,
    )