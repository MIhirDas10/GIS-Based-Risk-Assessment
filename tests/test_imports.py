"""
tests/test_imports.py

Smoke test: every src/ module imports without error.

This catches syntax errors, missing imports, and accidental top-level
side effects that would break Airflow when it loads the DAG file.
"""
import importlib

import pytest


@pytest.mark.parametrize("mod", [
    "ingestion.ingest_disease",
    "ingestion.ingest_boundaries",
    "ingestion.ingest_population",
    "ingestion.ingest_era5",
    "features.build_features",
    "models.train",
    "models.predict",
])
def test_module_imports(mod):
    m = importlib.import_module(mod)
    assert m is not None, f"{mod} returned None"
    # Every module that's also a DAG-callable entrypoint should expose `run`
    if mod in {
        "ingestion.ingest_disease",
        "ingestion.ingest_boundaries",
        "ingestion.ingest_population",
        "ingestion.ingest_era5",
        "features.build_features",
        "models.train",
        "models.predict",
    }:
        assert hasattr(m, "run"), f"{mod} is missing run()"
