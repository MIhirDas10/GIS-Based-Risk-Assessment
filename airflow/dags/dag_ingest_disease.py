from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys

# more will done 
sys.path.append(0, "/opt/airflow/src")

from ingestion.ingest_disease import run as ingest_disease

default_args = {
    "owner": "dengue_team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id = "ingest_disease_data",
    desc = "load kaggle dengue CSV into PostGIS",
    start_date = datetime(2024, 1, 1),
    schedule_interval = "@weekly",
    catchup = False,
    default_args = default_args,
    tags = ["ingestion", "disease"],
) as dag:
    
    ingest_task = PythonOperator(
        task_id = "ingest_dengue_cases",
        python_callable = ingest_disease,
    )