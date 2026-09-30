# Model Card — `dengue_risk_model`

*Following the [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) template.*

## Model Details

- **Name:** `dengue_risk_model` (MLflow registry)
- **Current Production version:** v1 (XGBoost), promoted 2026-05-26
- **Type:** Tabular regression (predicting weekly dengue case counts per district)
- **Algorithm:** Gradient-boosted trees (`xgboost.XGBRegressor`, objective `reg:squarederror`)
- **Hyperparameters:** `max_depth=6, learning_rate=0.05, n_estimators=500, subsample=0.8, colsample_bytree=0.8, min_child_weight=3`
- **Training framework:** scikit-learn 1.4.0 + xgboost 2.0.3
- **Tracking:** MLflow 2.10.2
- **Author:** dengue_team (per project metadata)
- **License:** Inherits from the dataset licenses (CC-BY for ERA5-Land, CC-BY-4.0 for WorldPop, Kaggle terms for dengue case CSV)

## Intended Use

### Primary use
- Weekly dengue case forecasting and risk-tier classification (`Low / Moderate / High / Critical`) per Bangladeshi district
- Decision support for public health officers: which districts need surveillance + vector-control resources next week

### Out-of-scope use
- **Not** an individual-risk calculator. Predictions are aggregate counts, not per-person probabilities.
- **Not** a diagnostic tool. Does not replace lab-confirmed case reporting.
- **Not** valid outside Bangladesh — climate and serotype dynamics differ regionally.

## Factors

### Geographic
- Trained only on districts that have historical case reports (the 8 divisional capitals: Dhaka, Chittagong/Chattogram, Khulna, Rajshahi, Barisal/Barishal, Sylhet, Rangpur, Mymensingh)
- Spatial-lag feature uses adjacent-district case counts; districts with no reporting neighbors get `cases_spatial_lag = 0`

### Temporal
- Training data: 2019–2021 (3 years)
- Validation: 2022 (1 year)
- Test: 2023 (1 year, never touched during training/tuning)
- **No random shuffling** — temporal split protects against leakage

## Metrics

Measured on the held-out **2023** test set (280 weekly observations).

| Model | RMSE | MAE | R² | Outbreak F1 (≥20 cases/wk) |
|---|---|---|---|---|
| Seasonal Naive (baseline) | 1349 | 353 | -0.019 | 0.218 |
| Ridge Regression | 1169 | 378 | 0.235 | 0.662 |
| **XGBoost (Production)** | **1129** | 480 | **0.286** | 0.634 |
| LightGBM | 1137 | 544 | 0.276 | 0.626 |

The Production model improves on the seasonal-naive baseline by **16.3%** RMSE. The original automatic-promotion rule (>15% improvement) was met.

### Caveats on these numbers
- R² of 0.286 is modest — the model captures roughly 29% of test-set variance. Real-world dengue is driven by many unobserved factors (mosquito-control campaigns, serotype shifts, household behaviour).
- High MAE relative to RMSE (480 vs 1129) indicates the model handles typical weeks well but mispredicts extreme outbreak peaks.
- Outbreak F1 of 0.63 is acceptable for triage but means roughly 1 in 3 outbreak weeks will be missed or false-alarmed.

## Training Data

| Source | Rows | Coverage | Provenance |
|---|---|---|---|
| `disease.dengue_cases` | 856 weekly | 8 districts × 2019–2023 | Kaggle Bangladesh dengue dataset (daily counts; aggregated weekly) |
| `geo.districts` | 64 polygons | Whole country | [HDX/UN OCHA bgd_admin2 shapefile](https://data.humdata.org/dataset/cod-ab-bgd) |
| `geo.district_population` | 64 | 2020 reference year | [WorldPop 1km Global Mosaic](https://www.worldpop.org/) |
| `weather.era5_district_weekly` | 17,216 weekly | 64 districts × 2019–2026-Q1 | [Copernicus ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) (6-hourly, zonal-aggregated to weekly) |

### Feature set (18 used by the model)
Direct weather: `temp_mean_c`, `temp_max_c`, `rainfall_mm`, `humidity_pct`
Lag (2/4 weeks): `rainfall_lag_2w`, `rainfall_lag_4w`, `temp_lag_2w`, `humidity_lag_2w`
Rolling (4-week): `temp_rolling_4w`, `rainfall_rolling_4w`
Interactions: `humidity_x_temp`, `rainfall_x_density`
Spatial: `cases_spatial_lag`
Cyclical / calendar: `week_sin`, `week_cos`, `month`
Static: `population_density`, `hotspot_rank`

See `DATA_DICTIONARY.md` for definitions.

## Quantitative Analyses

### Per-district performance
The model performs best on Dhaka (largest training signal), worst on Sylhet (smallest case counts). Per-district RMSE was not separately broken out — recommended next step.

### Per-year breakdown
| Year | Total cases | Mean weekly cases |
|---|---|---|
| 2019 | 37,691 | 248 |
| 2020 | 193 | 1.3 |
| 2021 | 450 | 3.0 |
| 2022 | 58,406 | 384 |
| 2023 | 101,312 | 666 |

2020-2021 anomaly likely reflects COVID-era under-reporting + lockdown reducing mosquito-human contact. This creates a non-stationary signal the model cannot fully account for.

## Ethical Considerations

### Risk of misuse
- **Resource allocation bias:** If decision makers route resources solely to districts the model rates "Critical", they may neglect districts where the model has poor coverage (the 56 districts without historical case data).
- **Reporting feedback loop:** Districts that under-report cases will have low historical signal → low predicted risk → less surveillance → less reporting. The hotspot-rank feature can amplify this.
- **Climate-only attribution:** The model treats weather as the primary driver. Public health interventions (e.g. larvicide campaigns) are not in the feature set, so their effects appear as "unexplained variance".

### Privacy
- Disease data is aggregated to district-week. No PII.
- All input data is publicly licensed.

### Recommendations for safe use
1. **Pair model output with reporting-volume context.** A "Low" tier for a district with zero historical cases means "we don't know", not "safe".
2. **Re-evaluate quarterly.** Climate, mosquito vectors, and reporting infrastructure all drift.
3. **Treat outbreak F1 of 0.63 as triage-grade.** Use as a screening signal, not as ground truth for declaring outbreaks.

## Caveats & Recommendations

- The model has only ever seen one severe-outbreak year (2023) in its test set. Performance on a 2024+ outbreak cannot be assured.
- The bootstrap confidence intervals (5% feature noise injection) are an approximation, not a true posterior. Wide intervals on `Critical` districts deserve human review.
- Population data is fixed at 2020. By 2026+ this is a 6-year lag — population growth in fast-urbanizing districts (Dhaka, Gazipur) is not captured.
- Recommend retraining quarterly via `airflow dags trigger build_feature_mart` followed by a manual `train.py` run, then comparing the new RMSE to the current Production model before promotion.

## Maintenance

- **Owner:** dengue_team
- **Retraining trigger:** Day 10 of the project plan adds a `dag_monitor.py` that compares predictions to actuals with a 4-week lag, and `dag_retrain.py` that fires `train.py` if RMSE degrades by >15%.
- **Rollback:** MLflow keeps all historical versions. `client.transition_model_version_stage(name="dengue_risk_model", version=N, stage="Production")` to swap.
