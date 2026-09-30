"""
src/models/train.py

Trains dengue prediction models on the features.mart table, logs
everything to MLflow, and promotes the best model to the registry.

Models (in order):
    1. Seasonal Naive  — same week last year (baseline floor)
    2. Ridge Regression — linear baseline with all features
    3. XGBoost          — primary model
    4. LightGBM         — comparison gradient booster

Validation strategy (CRITICAL — no random splits):
    Train:    2019–2021  (3 years)
    Validate: 2022       (tune hyperparams)
    Test:     2023       (final evaluation, never touched during tuning)

Metrics logged per model:
    RMSE, MAE, MAPE, R²
    Outbreak F1 (binary: cases > outbreak_threshold)

Artifacts logged per model:
    SHAP summary plot, feature importance plot,
    confusion matrix (outbreak classification),
    actual vs predicted scatter plot

MLflow registry flow:
    Best model on test set → registered as "dengue_risk_model"
    If it beats baseline by >15% RMSE → promoted to Production

Usage:
    python train.py                          # full pipeline
    python train.py --skip-registry          # train only, don't register
    python train.py --outbreak-threshold 50  # custom outbreak cutoff

Via Airflow:
    Called by dag_retrain.py (Day 10) during auto-retraining.
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

from db import get_sqlalchemy_engine  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("train")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT", "dengue_risk_prediction")
MODEL_REGISTRY_NAME = "dengue_risk_model"

# Temporal split boundaries
TRAIN_YEARS    = [2019, 2020, 2021]
VALIDATE_YEAR  = 2022
TEST_YEAR      = 2023

# Default outbreak threshold (cases per week per district)
DEFAULT_OUTBREAK_THRESHOLD = 20

# Features used by the models (everything except target + identifiers)
FEATURE_COLUMNS = [
    "temp_mean_c", "temp_max_c", "rainfall_mm", "humidity_pct",
    "rainfall_lag_2w", "rainfall_lag_4w", "temp_lag_2w", "humidity_lag_2w",
    "temp_rolling_4w", "rainfall_rolling_4w",
    "humidity_x_temp", "rainfall_x_density",
    "cases_spatial_lag",
    "week_sin", "week_cos", "month",
    "population_density", "hotspot_rank",
]

TARGET_COL = "dengue_cases"

# Identifier columns (kept for analysis but not used as features)
ID_COLUMNS = ["district_id", "year", "week"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features() -> pd.DataFrame:
    """
    Load features.mart from PostGIS into a pandas DataFrame.
    Falls back to CSV if DB is unavailable (for local development).
    """
    csv_path = Path(
        os.getenv("FEATURES_CSV_PATH", "data/processed/features_mart.csv")
    )

    try:
        log.info("Loading features from PostGIS …")
        # Use the SQLAlchemy engine so pandas doesn't emit the
        # "only supports SQLAlchemy connectable" deprecation warning.
        engine = get_sqlalchemy_engine()
        df = pd.read_sql(
            """
            SELECT
                m.*,
                d.district_name
            FROM features.mart m
            JOIN geo.districts d ON d.district_id = m.district_id
            ORDER BY m.district_id, m.year, m.week
            """,
            engine,
        )
        log.info("Loaded %d rows from PostGIS", len(df))

    except Exception as e:
        log.warning("PostGIS unavailable (%s), trying CSV fallback …", e)
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            log.info("Loaded %d rows from %s", len(df), csv_path)
        else:
            raise RuntimeError(
                f"Cannot load features: DB unavailable and {csv_path} not found. "
                "Run build_features.py --export first."
            ) from e

    return df


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(df: pd.DataFrame):
    """
    Split data into train/validate/test by year (temporal split).
    Handle NULLs in lag features (first few weeks per district).

    Returns:
        dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
                        df_train, df_val, df_test (full DataFrames for analysis)
    """
    log.info("Preparing temporal split …")

    # Fill NULL lags with 0 — these occur at the start of each district's
    # time series (first 2-4 weeks have no prior data to lag from).
    # Filling with 0 is safe: it means "no prior rainfall data" which is
    # the conservative signal for the model.
    for col in FEATURE_COLUMNS:
        null_count = df[col].isna().sum()
        if null_count > 0:
            log.info("  Filling %d NULLs in %s with 0", null_count, col)
            df[col] = df[col].fillna(0)

    # Drop rows where target is NULL (shouldn't happen but be safe)
    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    if len(df) < before:
        log.warning("Dropped %d rows with NULL target", before - len(df))

    # Temporal split
    df_train = df[df["year"].isin(TRAIN_YEARS)].copy()
    df_val   = df[df["year"] == VALIDATE_YEAR].copy()
    df_test  = df[df["year"] == TEST_YEAR].copy()

    log.info("  Train:    %4d rows  (%s)", len(df_train), TRAIN_YEARS)
    log.info("  Validate: %4d rows  (%d)", len(df_val), VALIDATE_YEAR)
    log.info("  Test:     %4d rows  (%d)", len(df_test), TEST_YEAR)

    if len(df_train) == 0:
        raise RuntimeError("Training set is empty — check feature mart data range")
    if len(df_test) == 0:
        raise RuntimeError("Test set is empty — check that 2023 data exists")

    X_train = df_train[FEATURE_COLUMNS].values
    y_train = df_train[TARGET_COL].values
    X_val   = df_val[FEATURE_COLUMNS].values if len(df_val) > 0 else None
    y_val   = df_val[TARGET_COL].values if len(df_val) > 0 else None
    X_test  = df_test[FEATURE_COLUMNS].values
    y_test  = df_test[TARGET_COL].values

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "df_train": df_train, "df_val": df_val, "df_test": df_test,
        "df_full":  df,
    }


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    outbreak_threshold: int,
) -> dict:
    """Compute regression + outbreak classification metrics."""
    # Regression metrics
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    # MAPE — avoid division by zero
    nonzero_mask = y_true > 0
    if nonzero_mask.sum() > 0:
        mape = float(
            np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask])
                           / y_true[nonzero_mask])) * 100
        )
    else:
        mape = float("nan")

    # Outbreak classification (binary: cases > threshold)
    y_true_binary = (y_true >= outbreak_threshold).astype(int)
    y_pred_binary = (y_pred >= outbreak_threshold).astype(int)

    outbreak_f1 = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "outbreak_f1": outbreak_f1,
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Plotting helpers (saved as MLflow artifacts)
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(y_true, y_pred, title, save_path):
    """Scatter plot of actual vs predicted with 45° reference line."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color="#2563eb")
    ax.plot(
        [0, max(y_true.max(), y_pred.max())],
        [0, max(y_true.max(), y_pred.max())],
        "r--", lw=1.5, label="Perfect prediction",
    )
    ax.set_xlabel("Actual Cases")
    ax.set_ylabel("Predicted Cases")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, title, save_path):
    """Plot confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    labels = ["No Outbreak", "Outbreak"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    # Print values in cells
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black",
                    fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(importance, feature_names, title, save_path, top_n=15):
    """Horizontal bar chart of top N feature importances."""
    indices = np.argsort(importance)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        range(len(indices)),
        importance[indices],
        color="#2563eb",
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_shap_summary(model, X, feature_names, title, save_path):
    """Generate SHAP summary plot if shap is available."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        plt.figure(figsize=(10, 7))   # sized so shap.summary_plot uses this canvas
        shap.summary_plot(
            shap_values, X,
            feature_names=feature_names,
            show=False,
            plot_size=(10, 7),
        )
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        return True
    except ImportError:
        log.warning("shap not installed — skipping SHAP plot")
        return False
    except Exception as e:
        log.warning("SHAP plot failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Model 1 — Seasonal Naive Baseline
# ---------------------------------------------------------------------------

def train_seasonal_naive(data: dict, outbreak_threshold: int) -> dict:
    """
    Seasonal naive: predict same week last year's cases.
    This is the absolute floor — any real model MUST beat this.
    """
    log.info("=" * 50)
    log.info("MODEL 1: Seasonal Naive Baseline")
    log.info("=" * 50)

    df_test  = data["df_test"]
    df_train_val = pd.concat([data["df_train"], data["df_val"]])

    # For each (district, week) in test, find same week in most recent prior year
    predictions = []
    for _, row in df_test.iterrows():
        prior = df_train_val[
            (df_train_val["district_id"] == row["district_id"]) &
            (df_train_val["week"] == row["week"])
        ]
        # Use the most recent year's value, or 0 if no prior year exists for that week
        pred = prior.loc[prior["year"].idxmax(), TARGET_COL] if len(prior) > 0 else 0
        predictions.append(pred)

    y_pred = np.array(predictions, dtype=float)
    y_test = df_test[TARGET_COL].values

    metrics = compute_metrics(y_test, y_pred, outbreak_threshold)

    log.info("  RMSE: %.2f  |  MAE: %.2f  |  R²: %.3f  |  Outbreak F1: %.3f",
             metrics["rmse"], metrics["mae"], metrics["r2"], metrics["outbreak_f1"])

    return {"name": "seasonal_naive", "y_pred": y_pred, "metrics": metrics}


# ---------------------------------------------------------------------------
# Model 2 — Ridge Regression
# ---------------------------------------------------------------------------

def train_ridge(data: dict, outbreak_threshold: int) -> dict:
    """
    Ridge regression — linear baseline with L2 regularization.
    Uses StandardScaler since linear models are scale-sensitive.

    Bundles the scaler INSIDE an sklearn Pipeline so MLflow logs a single
    artifact and predict.py can call .predict() on raw feature values.
    Without this, raw features fed to a model trained on scaled inputs
    produce wildly inflated predictions.
    """
    from sklearn.pipeline import Pipeline

    log.info("=" * 50)
    log.info("MODEL 2: Ridge Regression")
    log.info("=" * 50)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0)),
    ])
    model.fit(data["X_train"], data["y_train"])

    y_pred = model.predict(data["X_test"])
    y_pred = np.clip(y_pred, 0, None)  # cases can't be negative

    metrics = compute_metrics(data["y_test"], y_pred, outbreak_threshold)

    log.info("  RMSE: %.2f  |  MAE: %.2f  |  R²: %.3f  |  Outbreak F1: %.3f",
             metrics["rmse"], metrics["mae"], metrics["r2"], metrics["outbreak_f1"])

    return {
        "name": "ridge_regression",
        "model": model,
        "y_pred": y_pred,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Model 3 — XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(data: dict, outbreak_threshold: int) -> dict:
    """
    XGBoost — primary model. Hyperparameters tuned on validation set.
    """
    import xgboost as xgb

    log.info("=" * 50)
    log.info("MODEL 3: XGBoost")
    log.info("=" * 50)

    # Hyperparameter grid (tuned on validation set)
    param_grid = [
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
         "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 500,
         "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3},
        {"max_depth": 5, "learning_rate": 0.1,  "n_estimators": 200,
         "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 5},
    ]

    best_val_rmse = float("inf")
    best_model    = None
    best_params   = None

    for params in param_grid:
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            tree_method="hist",
            **params,
        )

        # Fit with early stopping on validation set
        eval_set = []
        if data["X_val"] is not None and len(data["X_val"]) > 0:
            eval_set = [(data["X_val"], data["y_val"])]

        model.fit(
            data["X_train"], data["y_train"],
            eval_set=eval_set if eval_set else None,
            verbose=False,
        )

        # Evaluate on validation set
        if data["X_val"] is not None and len(data["X_val"]) > 0:
            val_pred = model.predict(data["X_val"])
            val_rmse = float(np.sqrt(mean_squared_error(data["y_val"], val_pred)))
        else:
            # No validation data — use train RMSE as proxy
            val_pred = model.predict(data["X_train"])
            val_rmse = float(np.sqrt(mean_squared_error(data["y_train"], val_pred)))

        log.info("  Params: depth=%d lr=%.2f n=%d → Val RMSE: %.2f",
                 params["max_depth"], params["learning_rate"],
                 params["n_estimators"], val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model    = model
            best_params   = params

    log.info("  Best XGBoost params: %s (Val RMSE: %.2f)", best_params, best_val_rmse)

    # Final evaluation on test set
    y_pred = best_model.predict(data["X_test"])
    y_pred = np.clip(y_pred, 0, None)

    metrics = compute_metrics(data["y_test"], y_pred, outbreak_threshold)
    metrics["best_val_rmse"] = best_val_rmse

    log.info("  TEST — RMSE: %.2f  |  MAE: %.2f  |  R²: %.3f  |  Outbreak F1: %.3f",
             metrics["rmse"], metrics["mae"], metrics["r2"], metrics["outbreak_f1"])

    return {
        "name": "xgboost",
        "model": best_model,
        "params": best_params,
        "y_pred": y_pred,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Model 4 — LightGBM
# ---------------------------------------------------------------------------

def train_lightgbm(data: dict, outbreak_threshold: int) -> dict:
    """
    LightGBM — comparison gradient booster. Often faster than XGBoost
    and sometimes more accurate on small datasets.
    """
    import lightgbm as lgb

    log.info("=" * 50)
    log.info("MODEL 4: LightGBM")
    log.info("=" * 50)

    param_grid = [
        {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 300,
         "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 10},
        {"num_leaves": 50, "learning_rate": 0.05, "n_estimators": 500,
         "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 5},
        {"num_leaves": 40, "learning_rate": 0.1,  "n_estimators": 200,
         "subsample": 0.9, "colsample_bytree": 0.9, "min_child_samples": 10},
    ]

    best_val_rmse = float("inf")
    best_model    = None
    best_params   = None

    for params in param_grid:
        model = lgb.LGBMRegressor(
            objective="regression",
            random_state=42,
            verbose=-1,
            **params,
        )

        model.fit(
            data["X_train"], data["y_train"],
            eval_set=[(data["X_val"], data["y_val"])]
                if data["X_val"] is not None and len(data["X_val"]) > 0
                else None,
        )

        if data["X_val"] is not None and len(data["X_val"]) > 0:
            val_pred = model.predict(data["X_val"])
            val_rmse = float(np.sqrt(mean_squared_error(data["y_val"], val_pred)))
        else:
            val_pred = model.predict(data["X_train"])
            val_rmse = float(np.sqrt(mean_squared_error(data["y_train"], val_pred)))

        log.info("  Params: leaves=%d lr=%.2f n=%d → Val RMSE: %.2f",
                 params["num_leaves"], params["learning_rate"],
                 params["n_estimators"], val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model    = model
            best_params   = params

    log.info("  Best LightGBM params: %s (Val RMSE: %.2f)", best_params, best_val_rmse)

    y_pred = best_model.predict(data["X_test"])
    y_pred = np.clip(y_pred, 0, None)

    metrics = compute_metrics(data["y_test"], y_pred, outbreak_threshold)
    metrics["best_val_rmse"] = best_val_rmse

    log.info("  TEST — RMSE: %.2f  |  MAE: %.2f  |  R²: %.3f  |  Outbreak F1: %.3f",
             metrics["rmse"], metrics["mae"], metrics["r2"], metrics["outbreak_f1"])

    return {
        "name": "lightgbm",
        "model": best_model,
        "params": best_params,
        "y_pred": y_pred,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

def log_model_to_mlflow(
    result: dict,
    data: dict,
    outbreak_threshold: int,
    artifact_dir: Path,
):
    """
    Log one model run to MLflow: params, metrics, plots, and model artifact.
    """
    name = result["name"]
    metrics = result["metrics"]
    y_pred = result["y_pred"]
    y_test = data["y_test"]

    with mlflow.start_run(run_name=name) as run:
        # ---- Tags ----
        mlflow.set_tag("model_type", name)
        mlflow.set_tag("split_strategy", "temporal")
        mlflow.set_tag("train_years", str(TRAIN_YEARS))
        mlflow.set_tag("test_year", str(TEST_YEAR))

        # ---- Parameters ----
        mlflow.log_param("outbreak_threshold", outbreak_threshold)
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))
        mlflow.log_param("train_rows", len(data["y_train"]))
        mlflow.log_param("test_rows", len(data["y_test"]))
        if "params" in result:
            mlflow.log_params(result["params"])

        # ---- Metrics ----
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("mae", metrics["mae"])
        mlflow.log_metric("mape", metrics["mape"])
        mlflow.log_metric("r2", metrics["r2"])
        mlflow.log_metric("outbreak_f1", metrics["outbreak_f1"])
        if "best_val_rmse" in metrics:
            mlflow.log_metric("val_rmse", metrics["best_val_rmse"])

        # ---- Confusion matrix JSON ----
        cm_path = artifact_dir / f"{name}_confusion_matrix.json"
        with open(cm_path, "w") as f:
            json.dump({"confusion_matrix": metrics["confusion_matrix"]}, f, indent=2)
        mlflow.log_artifact(str(cm_path))

        # ---- Plots ----
        # Actual vs predicted
        avp_path = artifact_dir / f"{name}_actual_vs_predicted.png"
        plot_actual_vs_predicted(
            y_test, y_pred,
            f"{name} — Actual vs Predicted (Test {TEST_YEAR})",
            avp_path,
        )
        mlflow.log_artifact(str(avp_path))

        # Confusion matrix plot
        cm_plot_path = artifact_dir / f"{name}_confusion_matrix.png"
        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]),
            f"{name} — Outbreak Detection (threshold={outbreak_threshold})",
            cm_plot_path,
        )
        mlflow.log_artifact(str(cm_plot_path))

        # Feature importance (tree models only)
        if "model" in result and hasattr(result["model"], "feature_importances_"):
            fi_path = artifact_dir / f"{name}_feature_importance.png"
            plot_feature_importance(
                result["model"].feature_importances_,
                FEATURE_COLUMNS,
                f"{name} — Feature Importance",
                fi_path,
            )
            mlflow.log_artifact(str(fi_path))

        # SHAP (tree models only)
        if "model" in result and name in ("xgboost", "lightgbm"):
            shap_path = artifact_dir / f"{name}_shap_summary.png"
            plot_shap_summary(
                result["model"],
                data["X_test"],
                FEATURE_COLUMNS,
                f"{name} — SHAP Feature Impact (Test {TEST_YEAR})",
                shap_path,
            )
            if shap_path.exists():
                mlflow.log_artifact(str(shap_path))

        # ---- Log model artifact ----
        if "model" in result:
            model = result["model"]
            if name == "xgboost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            elif name == "lightgbm" or name == "ridge_regression":
                mlflow.sklearn.log_model(model, artifact_path="model")

        log.info("  MLflow run logged: %s (run_id=%s)", name, run.info.run_id)
        return run.info.run_id


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def register_best_model(
    results: list[dict],
    baseline_rmse: float,
    skip_registry: bool = False,
):
    """
    Register the best model (lowest test RMSE) in MLflow model registry.
    Promote to Production if it beats the seasonal naive baseline by >15%.
    """
    if skip_registry:
        log.info("Skipping model registry (--skip-registry flag)")
        return

    # Find best model (excluding seasonal naive)
    ml_results = [r for r in results if r["name"] != "seasonal_naive"]
    if not ml_results:
        log.warning("No ML models to register")
        return

    best = min(ml_results, key=lambda r: r["metrics"]["rmse"])
    best_rmse = best["metrics"]["rmse"]
    improvement = (baseline_rmse - best_rmse) / baseline_rmse * 100

    log.info("-" * 50)
    log.info("MODEL SELECTION")
    log.info("-" * 50)
    log.info("  Baseline (naive) RMSE:  %.2f", baseline_rmse)
    log.info("  Best model:             %s", best["name"])
    log.info("  Best model RMSE:        %.2f", best_rmse)
    log.info("  Improvement:            %.1f%%", improvement)

    if best_rmse >= baseline_rmse:
        log.warning(
            "Best model does NOT beat baseline — not registering. "
            "Check feature engineering or try different hyperparameters."
        )
        return

    # Register model
    if "mlflow_run_id" not in best:
        log.warning("No MLflow run ID — cannot register")
        return

    model_uri = f"runs:/{best['mlflow_run_id']}/model"

    try:
        result = mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)
        log.info("  Registered as '%s' version %s", MODEL_REGISTRY_NAME, result.version)

        # Promote to Production if improvement > 15%
        client = mlflow.tracking.MlflowClient()
        if improvement > 15:
            client.transition_model_version_stage(
                name=MODEL_REGISTRY_NAME,
                version=result.version,
                stage="Production",
            )
            log.info("  PROMOTED to Production (%.1f%% improvement > 15%% threshold)",
                     improvement)
        else:
            client.transition_model_version_stage(
                name=MODEL_REGISTRY_NAME,
                version=result.version,
                stage="Staging",
            )
            log.info("  Set to Staging (%.1f%% improvement < 15%% threshold)", improvement)

    except Exception as e:
        log.warning("Model registry failed: %s (MLflow server may be down)", e)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    outbreak_threshold: int = DEFAULT_OUTBREAK_THRESHOLD,
    skip_registry: bool = False,
):
    """
    Full training pipeline:
    1. Load features
    2. Prepare temporal split
    3. Train 4 models
    4. Log all to MLflow
    5. Register best model
    """
    log.info("=" * 60)
    log.info("DENGUE MODEL TRAINING PIPELINE — STARTING")
    log.info("=" * 60)
    log.info("  Outbreak threshold: %d cases", outbreak_threshold)
    log.info("  Train years:        %s", TRAIN_YEARS)
    log.info("  Validate year:      %d", VALIDATE_YEAR)
    log.info("  Test year:          %d", TEST_YEAR)

    # 1. Load
    df = load_features()

    # 2. Prepare
    data = prepare_data(df)

    # 3. Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    log.info("  MLflow tracking: %s", MLFLOW_TRACKING_URI)
    log.info("  MLflow experiment: %s", MLFLOW_EXPERIMENT)

    # Artifact directory for plots
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(exist_ok=True)

    # 4. Train all models
    results = []

    # Model 1 — Seasonal Naive (baseline)
    naive_result = train_seasonal_naive(data, outbreak_threshold)
    log_model_to_mlflow(naive_result, data, outbreak_threshold, artifact_dir)
    results.append(naive_result)
    baseline_rmse = naive_result["metrics"]["rmse"]

    # Model 2 — Ridge Regression
    ridge_result = train_ridge(data, outbreak_threshold)
    ridge_result["mlflow_run_id"] = log_model_to_mlflow(
        ridge_result, data, outbreak_threshold, artifact_dir
    )
    results.append(ridge_result)

    # Model 3 — XGBoost
    xgb_result = train_xgboost(data, outbreak_threshold)
    xgb_result["mlflow_run_id"] = log_model_to_mlflow(
        xgb_result, data, outbreak_threshold, artifact_dir
    )
    results.append(xgb_result)

    # Model 4 — LightGBM
    lgbm_result = train_lightgbm(data, outbreak_threshold)
    lgbm_result["mlflow_run_id"] = log_model_to_mlflow(
        lgbm_result, data, outbreak_threshold, artifact_dir
    )
    results.append(lgbm_result)

    # 5. Summary
    log.info("=" * 60)
    log.info("RESULTS SUMMARY (Test Year %d)", TEST_YEAR)
    log.info("=" * 60)
    log.info("  %-20s %8s %8s %8s %10s", "Model", "RMSE", "MAE", "R²", "Outbreak F1")
    log.info("  " + "-" * 58)
    for r in results:
        m = r["metrics"]
        log.info("  %-20s %8.2f %8.2f %8.3f %10.3f",
                 r["name"], m["rmse"], m["mae"], m["r2"], m["outbreak_f1"])

    # 6. Register best model
    register_best_model(results, baseline_rmse, skip_registry)

    log.info("=" * 60)
    log.info("DENGUE MODEL TRAINING PIPELINE — COMPLETE")
    log.info("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dengue prediction models")
    parser.add_argument(
        "--outbreak-threshold", type=int, default=DEFAULT_OUTBREAK_THRESHOLD,
        help=f"Cases per week to classify as outbreak (default: {DEFAULT_OUTBREAK_THRESHOLD})",
    )
    parser.add_argument(
        "--skip-registry", action="store_true",
        help="Skip MLflow model registry (train and log only)",
    )
    args = parser.parse_args()

    run(
        outbreak_threshold=args.outbreak_threshold,
        skip_registry=args.skip_registry,
    )
