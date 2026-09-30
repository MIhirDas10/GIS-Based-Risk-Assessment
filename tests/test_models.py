"""
tests/test_models.py

Verify the trained model in MLflow is loadable and produces sensible
predictions on real feature data.
"""
import pandas as pd
import pytest

# Every test here needs the MLflow registry and the live database.
pytestmark = pytest.mark.integration

# Must match the FEATURE_COLUMNS list in src/models/train.py
EXPECTED_FEATURES = [
    "temp_mean_c", "temp_max_c", "rainfall_mm", "humidity_pct",
    "rainfall_lag_2w", "rainfall_lag_4w", "temp_lag_2w", "humidity_lag_2w",
    "temp_rolling_4w", "rainfall_rolling_4w",
    "humidity_x_temp", "rainfall_x_density",
    "cases_spatial_lag",
    "week_sin", "week_cos", "month",
    "population_density", "hotspot_rank",
]


def test_production_model_registered(mlflow_client):
    """A 'Production' stage model must exist."""
    versions = mlflow_client.get_latest_versions("dengue_risk_model", stages=["Production"])
    assert len(versions) == 1, "Expected exactly one Production version"
    assert versions[0].current_stage == "Production"


def test_production_model_loadable(mlflow_client):
    import mlflow
    model = mlflow.pyfunc.load_model("models:/dengue_risk_model/Production")
    assert model is not None


def test_production_model_predicts_sensible_values(mlflow_client, pg_engine):
    """Load the current model and predict on this week's actual features.
    Predictions must be non-negative and within an order of magnitude of
    the observed range (we already saw 9M-case bug — this catches regressions)."""
    import mlflow
    model = mlflow.pyfunc.load_model("models:/dengue_risk_model/Production")

    # Pull current week features straight from the mart
    df = pd.read_sql(
        f"""SELECT {', '.join(EXPECTED_FEATURES)}
            FROM features.mart
            WHERE (year, week) = (
                SELECT year, week FROM features.mart
                ORDER BY year DESC, week DESC LIMIT 1
            )""",
        pg_engine,
    )
    assert len(df) > 0, "feature mart is empty"

    preds = model.predict(df.fillna(0))
    assert len(preds) == len(df)
    # No insane values (the scaler bug we fixed produced ~9_000_000)
    assert preds.max() < 50000, (
        f"Predictions max={preds.max()} suspiciously large — scaler/pipeline bug?"
    )
    # Predictions should be roughly non-negative (allow tiny float noise)
    assert preds.min() > -100


def test_mlflow_has_all_4_model_runs(mlflow_client):
    """train.py logs 4 named runs — verify they exist."""
    exp = mlflow_client.get_experiment_by_name("dengue_risk_prediction")
    assert exp is not None
    runs = mlflow_client.search_runs([exp.experiment_id])
    names = {r.data.tags.get("mlflow.runName") for r in runs}
    assert {"seasonal_naive", "ridge_regression", "xgboost", "lightgbm"}.issubset(names), (
        f"Missing one of the 4 expected runs. Found: {names}"
    )


def test_best_model_beats_baseline(mlflow_client):
    """The Production model's test RMSE must beat the seasonal-naive baseline."""
    exp = mlflow_client.get_experiment_by_name("dengue_risk_prediction")
    runs = mlflow_client.search_runs([exp.experiment_id])

    by_name = {r.data.tags.get("mlflow.runName"): r for r in runs}
    naive_rmse = by_name["seasonal_naive"].data.metrics["rmse"]
    # Find the Production model's source run (xgboost in current state)
    prod_versions = mlflow_client.get_latest_versions(
        "dengue_risk_model", stages=["Production"]
    )
    prod_run_id = prod_versions[0].run_id
    prod_rmse = mlflow_client.get_run(prod_run_id).data.metrics["rmse"]
    assert prod_rmse < naive_rmse, (
        f"Production RMSE {prod_rmse} doesn't beat naive {naive_rmse}"
    )
