"""
airflow/dags/dag_build_features.py

Airflow DAG — build the ML-ready feature mart (features.mart).

Schedule: @weekly — runs after ERA5 and disease ingestion complete.

Design:
- Two tasks: build → export
- Export produces data/processed/features_mart.csv for offline training
- Full rebuild each run (TRUNCATE + INSERT)
- Safe to re-run at any time
"""

import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow/src")
from features.build_features import run as build_features_run
from features.build_features import run_export_only as export_run

default_args = {
    "owner": "dengue_team",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="build_feature_mart",
    default_args=default_args,
    description="Build ML-ready feature mart from disease, weather, and population data",
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["features", "ml", "mart"],
) as dag:

    build_mart = PythonOperator(
        task_id="build_features_mart",
        python_callable=build_features_run,
        op_kwargs={"export": False},  # export handled by next task
        execution_timeout=timedelta(minutes=30),
        doc_md="""
        ### Feature mart build

        Full pipeline:
        1. Pre-flight checks (verify source tables have data)
        2. DDL (create features.mart if not exists)
        3. Truncate (full rebuild)
        4. 7-step CTE pipeline: base → lags → rolling → interaction
           → spatial lag → hotspot → final assembly
        5. Create indexes
        6. Validation (row counts, null rates, distributions)
        """,
    )

    export_csv = PythonOperator(
        task_id="export_features_csv",
        python_callable=export_run,
        execution_timeout=timedelta(minutes=10),
        doc_md="""
        ### Export features mart to CSV

        Dumps features.mart with district names to
        data/processed/features_mart.csv for offline model training
        or notebook exploration.
        """,
    )

    build_mart >> export_csv