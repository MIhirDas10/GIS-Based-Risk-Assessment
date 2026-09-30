"""
src/db.py

Shared database connection helpers.

We previously had identical ``get_db_conn()`` copies in five different
modules; this consolidates them. We also expose a SQLAlchemy engine so
that ``pandas.read_sql`` calls don't trip the
"only supports SQLAlchemy connectable" deprecation warning.
"""
from __future__ import annotations

import os
from functools import lru_cache

import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def _conn_kwargs() -> dict:
    """Resolve connection parameters from environment with sensible defaults."""
    return {
        "host":     os.getenv("POSTGRES_HOST", "localhost"),
        "port":     int(os.getenv("POSTGRES_PORT", 5432)),
        "dbname":   os.getenv("POSTGRES_DB", "dengue_db"),
        "user":     os.getenv("POSTGRES_USER", "dengue_admin"),
        "password": os.getenv("POSTGRES_PASSWORD", "dengue_pass_2024"),
    }


def get_db_conn():
    """Return a raw psycopg2 connection (caller must close it)."""
    return psycopg2.connect(**_conn_kwargs())


def _build_sqlalchemy_url() -> str:
    p = _conn_kwargs()
    return (
        f"postgresql+psycopg2://{p['user']}:{p['password']}"
        f"@{p['host']}:{p['port']}/{p['dbname']}"
    )


@lru_cache(maxsize=1)
def get_sqlalchemy_engine() -> Engine:
    """
    Return a process-wide SQLAlchemy engine, lazily built once.

    Use this with ``pandas.read_sql`` to avoid the psycopg2
    deprecation warning ("pandas only supports SQLAlchemy connectable").
    Pooling defaults are fine for our batch jobs — short-lived scripts
    create a single connection then release it.
    """
    return create_engine(
        _build_sqlalchemy_url(),
        pool_pre_ping=True,   # transparently recover from idle drops
        future=True,
    )
