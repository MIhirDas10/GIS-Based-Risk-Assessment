"""
airflow/dags/dag_ingest_era5.py

Airflow DAG — ingest ERA5-Land weather data for Bangladesh.

Schedule: @weekly (keeps data fresh going forward)
First run: manually trigger a backfill for years 2019–2023

Backfill command (run once after deploying):
    docker exec -it dengue_airflow_webserver airflow dags backfill \
        --start-date 2019-01-01 --end-date 2023-12-31 ingest_era5_weather

Design notes:
- One task per year (2019–2023) run in sequence to avoid Copernicus API
  rate limits and keep individual task memory usage low.
- Going forward the @weekly schedule pulls the current year only.
- Retries=2 with 10-min delay handles transient CDS API timeouts.
"""

import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow/src")
from ingestion.ingest_era5 import run as era5_run

# ---------------------------------------------------------------------------
# Default args
# ---------------------------------------------------------------------------
default_args = {
    "owner": "dengue_team",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),   # CDS API can be slow to recover
    "email_on_failure": False,
}

# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="ingest_era5_weather",
    default_args=default_args,
    description="Download ERA5-Land weather for Bangladesh and load into PostGIS",
    schedule_interval="@weekly",
    start_date=datetime(2019, 1, 1),
    catchup=False,          # set True only during manual backfill
    max_active_runs=1,      # never run two ERA5 downloads in parallel
    tags=["ingestion", "weather", "era5"],
) as dag:

    # ------------------------------------------------------------------
    # Historical backfill tasks — one per year
    # Run these by setting catchup=True and triggering manually, or by
    # calling era5_run() directly in a one-off script.
    # ------------------------------------------------------------------
    previous_task = None

    for year in range(2019, 2024):   # 2019, 2020, 2021, 2022, 2023

        task = PythonOperator(
            task_id=f"ingest_era5_{year}",
            python_callable=era5_run,
            op_kwargs={"year": year},
            execution_timeout=timedelta(hours=6),   # ERA5 download can be slow
            doc_md=f"""
            ### ERA5 ingestion — {year}
            Downloads ERA5-Land hourly data for Bangladesh ({year}),
            computes zonal statistics per district polygon, aggregates
            to weekly summaries, and upserts into
            `weather.era5_district_weekly`.
            """,
        )

        # Run years sequentially to avoid hammering the CDS API
        if previous_task is not None:
            previous_task >> task
        previous_task = task

    # ------------------------------------------------------------------
    # Weekly refresh task — pulls current year going forward
    # ------------------------------------------------------------------
    import datetime as _dt

    def ingest_current_year(**context):
        """Pull ERA5 for whatever year the DAG is currently running in."""
        current_year = context["logical_date"].year
        era5_run(year=current_year)

    weekly_refresh = PythonOperator(
        task_id="ingest_era5_current_year",
        python_callable=ingest_current_year,
        execution_timeout=timedelta(hours=4),
        doc_md="""
        ### ERA5 weekly refresh
        Runs every week and re-pulls the current year's ERA5 data.
        Uses ON CONFLICT DO UPDATE so rerunning is safe.
        """,
    )

    # Historical years must finish before the weekly refresh starts
    previous_task >> weekly_refresh