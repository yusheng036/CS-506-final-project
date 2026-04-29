# CS 506 Final Project — MBTA Bus Crowding Analysis

## Video
[YouTube link - to be added]

## How to Build and Run

### Install dependencies
```
make install
```

### Run all models
```
make run
```

### Run tests
```
make test
```

### Dataset
The dataset `Ridership_v1.sqlite` is stored in the `dataset/` folder via Git LFS.

---

## Model Results

| Model | Test MAE | Test RMSE |
|---|---|---|
| Linear Regression | 5.003 | 7.045 |
| KNN (k=3) | 3.077 | 4.994 |
| XGBoost | 3.596 | 5.331 |
| Random Forest | 2.741 | 4.193 |
| Lasso (L1 reg) | 5.348 | 7.452 |
| Ridge (L2 reg) | 5.306 | 7.472 |
| Neural Network | 3.020 | 4.331 |
| Linear SVM | 3.924 | 6.138 |

All models predict `average_load` (average number of passengers on the bus after leaving a stop) using features including route, stop sequence, direction, day type, time period, and season year. Models are evaluated on 2024 data after training on 2016–2022 and validating on 2023. Random Forest achieves the lowest test MAE (2.741) and is used as the primary model for prediction analysis.

Note: Linear Regression and Linear SVM use a random 80/20 train/test split and a smaller feature set, implemented independently. Their results are included for reference but are not directly comparable to the chronological split used by other models.

## Key Findings

- **MIDDAY_SCHOOL and PM_PEAK are consistently the most crowded periods** across the top busiest routes, with predicted loads of 14–18 passengers.
- **VERY_EARLY_MORNING and NIGHT are the least crowded periods**, with predicted loads of 5–8 passengers across all routes.
- **Route 455 is the busiest route overall**, peaking at approximately 18 passengers during MIDDAY_SCHOOL on weekdays.
- **Non-linear models significantly outperform linear baselines** (Random Forest MAE 2.741 vs Linear Regression MAE 5.003), confirming that bus load has strong non-linear interactions between route and time period.
- **Random Forest is the recommended model** for predicting MBTA bus crowding based on lowest MAE and RMSE across all models tested.

---

## Project Description

Our team studies MBTA bus crowding across Boston using historical ridership data. We analyze how passenger load changes depending on route, time of day, and day type, and build predictive models to identify when and where buses are most and least crowded.

## Project Timeline
* Weeks 1–2: Data Collection
* Weeks 3–4: Modeling Crowd Levels
* Weeks 5–6: Quantitative Evaluation + Model Comparison
* Weeks 7–8: Prediction Analysis + Final Report

## Project Goal

The primary goal is to analyze MBTA bus performance data to identify whether bus service quality is consistent across different routes and time periods in Boston.

Specifically, this project aims to:
1. Quantify bus service performance by measuring average passenger load by route, stop, and time of day.
2. Identify temporal disparities in bus performance by comparing service across peak versus off-peak travel periods.
3. Predict bus crowding levels using historical MBTA ridership data based on features such as route, time of day, day of week, and season.
4. Determine the least crowded travel time periods for selected bus routes using regression models trained on historical data.

These goals are evaluated using MAE and RMSE across chronological train/val/test splits.

## Data Collection

1. **MBTA Bus Ridership by Time Period, Season, Route/Line, and Stop**
   The dataset provides ridership activity including average boardings, alightings, and passenger load at each stop across different seasons. Downloaded from the MBTA Open Data Portal.
   https://mbta-massdot.opendata.arcgis.com/datasets/7acd353c1a734eb8a23caf46a0e66b23_0/explore

2. **MBTA Bus Arrival Departure Times (2020-2025)**
   Originally planned for delay analysis, but excluded from final modeling due to data completeness issues. All models use the ridership dataset only.

## Data Modelling

We preprocess the MBTA ridership data by cleaning missing values and aggregating passenger counts by route, stop, and time of day. The prediction target is `average_load` — average passengers on the bus after leaving a stop — used as a proxy for crowding.

We implemented and compared eight models:
- **Linear Regression** — interpretable baseline with One-Hot Encoding
- **KNN Regressor (k=3)** — non-parametric, with StandardScaler; k selected by validation MAE
- **XGBoost** — gradient boosted trees with chronological split
- **Random Forest** — ensemble of 500 trees; best overall performance
- **Lasso (L1 regularization)** — linear model with sparsity penalty
- **Ridge (L2 regularization)** — linear model with shrinkage penalty
- **Neural Network** — 3-layer MLP with dropout, trained with early stopping
- **Linear SVM** — LinearSVR with feature hashing for categorical encoding

All models except Linear Regression and Linear SVM use a chronological train/val/test split: train on 2016–2022, validate on 2023, test on 2024.

## Data Visualization

Visualizations are saved in `diagram/visual/` and `diagram/model/`:
- `seasonal_load.png` — average load by season (2016–2024)
- `weekday_ridership.png` / `weekend_ridership.png` — ridership patterns by time period
- `top20_load.png` — top 20 busiest routes by average load
- `model_comparison.png` — MAE, RMSE, and L∞ error across all 8 models
- `rf_load_by_time.png` — top 5 busiest routes predicted load across all time periods
- `knn_k_selection.png` — KNN validation MAE vs k
- Per-model actual vs predicted scatter plots

## Test Plan

Models are trained on 2016–2022, validated on 2023, and tested on 2024 using MAE and RMSE. All models are also compared against a historical average baseline (group mean by route, time period, and day type). Random Forest outperforms the baseline by 37% on MAE (2.741 vs 4.344).

---

## Dataset Metadata

*ALL rows with null values in at least one field are dropped. Unnecessary columns are dropped to keep the database lightweight without Git LFS for the cleaned version.*

### mode
*dropped — all values are 3 (Bus)*

### season
Reporting period. Only Fall seasons exist in this dataset (Fall 2016 – Fall 2024).

### route_id
*dropped — redundant with route_name for MBTA bus*

### route_name
Human-readable route label (e.g. 1, 47, 66).

### route_variant
*dropped — considered useless*

### stop_sequence
Order of stops along a route. 1 = first stop.

### direction_id
0 = outbound, 1 = inbound

### day_type_name
weekday / saturday / sunday

### time_period_name
- VERY_EARLY_MORNING
- EARLY_AM
- AM_PEAK
- MIDDAY_BASE
- MIDDAY_SCHOOL
- PM_PEAK
- EVENING
- LATE_EVENING
- NIGHT
- OFF_PEAK

### stop_name
Human-readable stop name. Example: MASSACHUSETTS AVE @ BEACON ST.

### stop_id
Numeric identifier for the stop.

### average_ons_per_trip
Average boardings at this stop per trip.

### average_offs_per_trip
Average alightings at this stop per trip.

### average_load_per_trip
Average passengers on the vehicle after leaving this stop. Used as the crowding proxy and prediction target.

### num_trips
Number of trips included in the calculation. Smaller values indicate less reliable averages.

### ons_all_trips
Total boardings across all trips in the dataset.