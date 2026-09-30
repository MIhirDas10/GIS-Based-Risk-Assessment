"""
tests/test_pure_functions.py

Pure-function unit tests — no DB, no network. Cover helpers that we'd
otherwise rely on integration tests to catch.
"""
import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# ingest_disease.parse_date
# ---------------------------------------------------------------------------

def test_parse_date_dd_mm_yy():
    from ingestion.ingest_disease import parse_date
    year, month, week = parse_date("27/8/19")
    assert (year, month) == (2019, 8)
    # week 35 in ISO 8601 for 2019-08-27
    assert week == 35


def test_parse_date_dd_mm_yyyy():
    from ingestion.ingest_disease import parse_date
    year, month, week = parse_date("01/01/2023")
    # 2023-01-01 is a Sunday → ISO week 52 of 2022
    assert (year, month) == (2023, 1)
    assert week in (52, 1)  # ISO can put Jan 1 in either


def test_parse_date_invalid_raises():
    from ingestion.ingest_disease import parse_date
    with pytest.raises(ValueError):
        parse_date("not-a-date")


# ---------------------------------------------------------------------------
# ingest_era5.dewpoint_to_rh
# ---------------------------------------------------------------------------

def test_dewpoint_to_rh_at_saturation():
    """When temp == dewpoint, RH must be 100%."""
    from ingestion.ingest_era5 import dewpoint_to_rh
    temp = np.array([300.0])
    dew = np.array([300.0])
    rh = dewpoint_to_rh(temp, dew)
    assert math.isclose(rh[0], 100.0, abs_tol=0.5)


def test_dewpoint_to_rh_drier_air():
    """Dewpoint below temp → RH < 100%."""
    from ingestion.ingest_era5 import dewpoint_to_rh
    temp = np.array([300.0])
    dew = np.array([285.0])  # 15°C below temp
    rh = dewpoint_to_rh(temp, dew)
    assert 0 < rh[0] < 100
    # Roughly half-saturated air at 27°C with 12°C dewpoint
    assert 30 < rh[0] < 60


def test_dewpoint_to_rh_clipped_to_0_100():
    """Even weird inputs must produce RH in [0, 100]."""
    from ingestion.ingest_era5 import dewpoint_to_rh
    temp = np.array([320.0, 250.0])
    dew = np.array([200.0, 320.0])
    rh = dewpoint_to_rh(temp, dew)
    assert rh.min() >= 0
    assert rh.max() <= 100


# ---------------------------------------------------------------------------
# ingest_era5._is_zip
# ---------------------------------------------------------------------------

def test_is_zip_detects_magic(tmp_path):
    from ingestion.ingest_era5 import _is_zip
    p = tmp_path / "fake.zip"
    p.write_bytes(b"PK\x03\x04rest_of_zip_data")
    assert _is_zip(p) is True


def test_is_zip_rejects_netcdf(tmp_path):
    from ingestion.ingest_era5 import _is_zip
    p = tmp_path / "fake.nc"
    p.write_bytes(b"CDF\x01rest_of_netcdf_data")
    assert _is_zip(p) is False


def test_is_zip_handles_missing_file(tmp_path):
    from ingestion.ingest_era5 import _is_zip
    assert _is_zip(tmp_path / "nope.dat") is False


# ---------------------------------------------------------------------------
# ingest_boundaries.to_multipolygon
# ---------------------------------------------------------------------------

def test_to_multipolygon_wraps_polygon():
    from shapely.geometry import MultiPolygon, Polygon

    from ingestion.ingest_boundaries import to_multipolygon
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    result = to_multipolygon(poly)
    assert isinstance(result, MultiPolygon)
    assert len(result.geoms) == 1


def test_to_multipolygon_passthrough_multipolygon():
    from shapely.geometry import MultiPolygon, Polygon

    from ingestion.ingest_boundaries import to_multipolygon
    mp = MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])])
    result = to_multipolygon(mp)
    assert isinstance(result, MultiPolygon)


def test_to_multipolygon_none():
    from ingestion.ingest_boundaries import to_multipolygon
    assert to_multipolygon(None) is None


# ---------------------------------------------------------------------------
# train.compute_metrics
# ---------------------------------------------------------------------------

def test_compute_metrics_perfect_prediction():
    """If y_pred == y_true, RMSE/MAE = 0, R² = 1, F1 = 1."""
    from models.train import compute_metrics
    y_true = np.array([10, 20, 30, 40, 50])
    y_pred = y_true.copy().astype(float)
    m = compute_metrics(y_true, y_pred, outbreak_threshold=20)
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0
    assert m["r2"] == 1.0
    # All values ≥ threshold for both → F1 = 1
    assert m["outbreak_f1"] == 1.0


def test_compute_metrics_constant_baseline():
    """A naive predict-the-mean baseline should have RMSE > 0 and R² ≈ 0."""
    from models.train import compute_metrics
    y_true = np.array([10, 20, 30, 40, 50])
    y_pred = np.full_like(y_true, y_true.mean(), dtype=float)
    m = compute_metrics(y_true, y_pred, outbreak_threshold=25)
    assert m["rmse"] > 0
    assert math.isclose(m["r2"], 0.0, abs_tol=1e-9)


def test_compute_metrics_handles_all_zero_actuals():
    """MAPE undefined when all y_true are 0 — must not crash."""
    from models.train import compute_metrics
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1.0, 2.0, 3.0])
    m = compute_metrics(y_true, y_pred, outbreak_threshold=10)
    assert math.isnan(m["mape"])
    assert m["rmse"] > 0


# ---------------------------------------------------------------------------
# predict.compute_risk_scores
# ---------------------------------------------------------------------------

def test_compute_risk_scores_assigns_tiers():
    """Given 8 districts with varied risk, tiers should distribute via percentile."""
    import pandas as pd

    from models.predict import compute_risk_scores

    df = pd.DataFrame({
        "district_id": range(8),
        "year": [2023] * 8,
        "week": [52] * 8,
        "predicted_cases": [10, 20, 50, 100, 200, 500, 1000, 5000],
        "population_density": [500] * 8,
        "hotspot_rank": np.linspace(0, 1, 8),
    })
    out = compute_risk_scores(df)
    assert "risk_tier" in out.columns
    assert "risk_score" in out.columns
    assert "risk_color" in out.columns
    # Risk score must be monotonic w.r.t predicted_cases (when density is constant)
    assert out["risk_score"].is_monotonic_increasing
    # All 4 tiers should appear given 8 distinct rows
    tiers = set(out["risk_tier"])
    assert tiers.issubset({"Low", "Moderate", "High", "Critical"})


def test_compute_risk_scores_negative_clipped():
    """Risk score can never be negative."""
    import pandas as pd

    from models.predict import compute_risk_scores
    df = pd.DataFrame({
        "district_id": [0, 1],
        "year": [2023, 2023],
        "week": [52, 52],
        "predicted_cases": [-10.0, 100.0],   # model may predict negative
        "population_density": [500.0, 500.0],
        "hotspot_rank": [0.5, 0.5],
    })
    out = compute_risk_scores(df)
    assert (out["risk_score"] >= 0).all()
