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

Random Forest achieves the best test MAE (2.741) across all models. All models predict `average_load` (average number of passengers on the bus after leaving a stop) using features including route, stop sequence, direction, day type, time period, and season year. Models are evaluated on 2024 data after training on 2016–2022 and validating on 2023.

---

## Project Description

Our team will study MBTA bus crowding and service reliability across Boston using ridership and arrival/departure data. We want to understand how passenger load and delays change depending on the route, neighborhood, and time of day. Using data from the MBTA Rider Census and other performance datasets, we will measure patterns such as which routes tend to be most crowded, when delays are most common, and whether some areas experience less reliable service than others. The main purpose of this project is to concentrate on quantitative analysis and predictive modeling to identify where and when crowding and reliability problems are most severe. Our results will also allow us to suggest which routes or time periods may need additional service improvements.

## Project Timeline
* Weeks 1–2: Data Collection
* Weeks 3–4: Modeling Crowd Levels + Delay Patterns
* Weeks 5–6: Quantitative Evaluation + Disparity Analysis
* Weeks 7–8: Recommendations + Final Report

## Project Goal

The primary goal of this project is to analyze MBTA bus performance data to identify whether bus service quality is consistent across different geographic areas, routes, and time periods in Boston. In this project, equity is defined as fairness in service reliability rather than economic or demographic factors.

Specifically, this project aims to:
1. Quantify bus service performance across Boston by measuring key indicators such as average delay time, on-time arrival rate, and passenger load by route, stop, and time of day.
2. Identify spatial and temporal disparities in bus performance by comparing service reliability across different neighborhoods, bus routes, and peak versus off-peak travel periods.
3. Predict bus crowding levels and delay patterns using historical MBTA ridership and arrival/departure data, based on features such as route, time of day, day of week, and season.
4. Determine the least crowded and most reliable travel time periods for selected bus routes by applying regression or classification models to historical data.

These goals will be evaluated using quantitative metrics such as prediction error (e.g., MAE or RMSE), differences in average delay across routes and time periods, and consistency of results across train-test splits.

## Data Collection

1. **MBTA Bus Ridership by Time Period, Season, Route/Line, and Stop**
   Description: The dataset provides ridership activity such as average on and off at each stop during different seasons.
   Data Access Method: Downloaded from the MBTA Open Data Portal. https://mbta-massdot.opendata.arcgis.com/datasets/7acd353c1a734eb8a23caf46a0e66b23_0/explore

2. **MBTA Bus Arrival Departure Times (2020-2025)**
   Description: The dataset contains arrival and departure events for MBTA buses and includes both scheduled and actual departure times at key timepoints along each trip.
   Data Access Method: Downloaded from the MBTA Open Data Portal. https://mbta-massdot.opendata.arcgis.com/datasets/924df13d845f4907bb6a6c3ed380d57a/about

## Data Modelling

We preprocess the MBTA ridership data by cleaning missing values and aggregating passenger counts by route, stop, and time of day. Crowd levels are measured using `average_load` (average passengers on the bus after leaving a stop) over fixed time intervals.

We implemented and compared the following models to predict `average_load`:
- Linear Regression (baseline)
- KNN Regressor (k=3, with StandardScaler)
- XGBoost
- Random Forest
- Lasso (L1 regularization)
- Ridge (L2 regularization)
- Neural Network
- Linear SVM

All models use a chronological train/val/test split: train on 2016–2022, validate on 2023, test on 2024.

## Data Visualization

Visualizations are saved in `diagram/visual/` and `diagram/model/`:
- Seasonal average load trends
- Weekday vs weekend ridership patterns
- Top 20 busiest routes by average load
- Per-route load by time period
- Model performance comparison (MAE, RMSE, L∞)
- Actual vs predicted plots for each model

## Test Plan

We use a chronological train/test split to measure real-world generalization. Models are trained on 2016–2022, validated on 2023, and tested on 2024. Performance is evaluated using MAE and RMSE. Models are also compared against a historical average baseline.

---

## Dataset Metadata

*ALL rows with null values in at least one field are dropped, and all unnecessary columns are also dropped.*

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
Number of trips included in the calculation.

### ons_all_trips
Total boardings across all trips in the dataset.