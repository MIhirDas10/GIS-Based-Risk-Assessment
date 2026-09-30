"""
tests/conftest.py

Shared pytest fixtures. Tests run inside the airflow container so all
deps (psycopg2, mlflow, redis, geopandas, etc.) are available and the
container env already has POSTGRES_HOST=postgres etc.
"""
import os
import sys
from pathlib import Path

import pytest

# Make src/ importable when running pytest from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(scope="session")
def pg_conn():
    """A psycopg2 connection to the dengue DB, reused across the session."""
    import psycopg2
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "dengue_db"),
        user=os.getenv("POSTGRES_USER", "dengue_admin"),
        password=os.getenv("POSTGRES_PASSWORD", "dengue_pass_2024"),
    )
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def pg_engine():
    """A SQLAlchemy engine for tests that want pandas-friendly reads."""
    from db import get_sqlalchemy_engine
    return get_sqlalchemy_engine()


@pytest.fixture(scope="session")
def mlflow_client():
    """An MLflow tracking client pointed at the in-cluster server."""
    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    return mlflow.tracking.MlflowClient()


@pytest.fixture(scope="session")
def redis_client():
    """A Redis client pointed at the in-cluster Redis."""
    import redis
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "redis"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True,
    )
