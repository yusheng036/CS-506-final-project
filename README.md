# CS 506 Final Project Proposal

Project Description:

Our team will study MBTA bus crowding and service reliability across Boston using ridership and arrival/departure data. We want to understand how passenger load and delays change depending on the route, neighborhood, and time of day. Using data from the MBTA Rider Census and other performance datasets, we will measure patterns such as which routes tend to be most crowded, when delays are most common, and whether some areas experience less reliable service than others. The main purpose of this project is to concentrate on quantitative analysis and predictive modeling to identify where and when crowding and reliability problems are most severe. Our results will also allow us to suggest which routes or time periods may need additional service improvements.

Project Timeline:
* Weeks 1–2: Data Collection 
* Weeks 3–4: Modeling Crowd Levels + Delay Patterns 
* Weeks 5–6: Quantitative Evaluation + Disparity Analysis 
* Weeks 7–8: Recommendations + Final Report

Project Goal:

The primary goal of this project is to analyze MBTA bus performance data to identify whether bus service quality is consistent across different geographic areas, routes, and time periods in Boston. In this project, equity is defined as fairness in service reliability rather than economic or demographic factors.

Specifically, this project aims to:
1. Quantify bus service performance across Boston by measuring key indicators such as average delay time, on-time arrival rate, and passenger load by route, stop, and time of day.
2. Identify spatial and temporal disparities in bus performance by comparing service reliability across different neighborhoods, bus routes, and peak versus off-peak travel periods.
3. Predict bus crowding levels and delay patterns using historical MBTA ridership and arrival/departure data, based on features such as route, time of day, day of week, and season.
4. Determine the least crowded and most reliable travel time periods for selected bus routes by applying regression or classification models to historical data.
These goals will be evaluated using quantitative metrics such as prediction error (e.g., MAE or RMSE), differences in average delay across routes and time periods, and consistency of results across train-test splits.


Data Collection:

1. MBTA Bus Ridership by Time Period, Season, Route/Line, and Stop
Description: The dataset provides ridership activity such as average on and off at each stop during different seasons.
Data Access Method: We will download this dataset from the MBTA Open Data Portal. https://mbta-massdot.opendata.arcgis.com/datasets/7acd353c1a734eb8a23caf46a0e66b23_0/explore
1. MBTA Bus Arrival Departure Times (2020-2025)
Description: The dataset contains arrival and departure events for MBTA buses and includes both scheduled and actual departure times at key timepoints along each trip. Due to data collection issues, the dataset is not guaranteed to be complete for each stop or date, so our analysis will include quality checks and focus on sufficient coverage.
Data Access Method: We will download this dataset from the MBTA Open Data Portal. https://mbta-massdot.opendata.arcgis.com/datasets/924df13d845f4907bb6a6c3ed380d57a/about


Data Modelling:
	
We will preprocess the MBTA ridership and arrival/departure data by cleaning missing values and aggregating passenger counts by route, stop, and time of day. Crowd levels will be measured using average passenger load over fixed time intervals. To identify the least crowded travel periods, we will use simple predictive models such as XGBoost to estimate passenger load based on features like time of day, day of week, season, and route. The model results will help highlight consistent low-crowding time periods across different bus routes.

Data Visualization
	Through frontend techniques, we fetch the functions and data from backend. These data will then be displayed using matplotlib. The exact plot type will depend on the data type, and we expect it to be a dot plot or linear model. On the other hand, it might be ideal to provide multiple plots, one for the least-crowded stations across different time periods and one for the specific data analysis for each station. In specific, for topic1, it is ideal to visualize the data using a table. For topic2, it is ideal to represent them in the map. For topic 3 and 4, it might be ideal to visualize the data using a bar plot or dot plot. 
	The current plan is : 1) Add multiple location-based heat graphs which display the crowdedness across different time periods. 2) Add table structure for the interface


Test Plan:

To evaluate our system, we will use a chronological train/test split to measure real-world generalization. We will train our models on MBTA bus data from 2023–2024 and test performance on 2025 data to determine whether identified low-crowding periods and reliability patterns remain consistent in a future year. To tune model hyperparameters, we will reserve the final three months of 2024 as a validation set. For crowding prediction, we will evaluate performance using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) if modeling passenger load as a continuous variable. We will also compare our model against baseline options, including a historical average model based on route and time-of-day, to ensure that our predictive model provides meaningful improvement.




Log of Ridership.sqlite change, and explanations on the attributes:

# Meta data for MBTA Ridership

### mode *dropped, in BMTA ridership, all modes are set into 3, since it is about bus (3)*

Indicates the type of transit service. Examples:  
1 = Light Rail (Green Line)
2 = Heavy Rail (Red, Orange, Blue)
3 = Bus
4 = Commuter Rail    
5 = Ferry
6 = The Ride (paratransit)

### season

The reporting period for the ridership data. Examples: 2024 Summer, 2024 Fall.

### route_id *dropped, route_id and route_name is effectively the same for MBTA bus*

Numeric identifier for the route. For buses, this is the bus route number. For rapid transit, typical values include: 1 = Red Line 2 = Orange Line 3 = Blue Line 4 = Green Line

### route_name

Human-readable name or label for the route. Sometimes redundant with route_id.

### route_variant *dropped, considered useless data*

Specific branch or pattern of the route. Examples: 1_0 = Route 1 inbound 1_1 = Route 1 outbound 47_0 = Route 47 inbound

## Trip Structure

### stop_sequence

The order of stops along a route variant. 1 = first stop, 2 = second stop, and so on.

### direction_id

Direction of travel. 0 = outbound 1 = inbound

### day_type_id and day_type_name *day_type_id dropped, since the corresponding relationship is decoded and recorded as below:*

day_type_id (1) = weekday
day_type_id (2) = saturday
day_type_id (3) = sunday

Indicates the type of day represented. Examples: WKD = weekday SAT = Saturday SUN = Sunday

### time_period_id and time_period_name *time_period_id dropped, the relationship between time_period_id and time_period_name is captured as below:*

time_period_01, VERY_EARLY_MORNING
time_period_02, EARLY_AM
time_period_03, AM_PEAK
time_period_04, MIDDAY_BASE
time_period_05, MIDDAY_SCHOOL
time_period_06, PM_PEAK
time_period_07, EVENING
time_period_08, LATE_EVENING
time_period_09, NIGHT
time_period_10, OFF_PEAK
time_period_11, OFF_PEAK


Indicates the time-of-day bucket. Examples: AM Peak, Midday, PM Peak, Evening, Late Night.

## Stop Information

### stop_name

Human-readable name of the stop. Example: MASSACHUSETTS AVE @ BEACON ST.

### stop_id 

Numeric identifier for the stop, unique across MBTA.

## Ridership Metrics

### average_ons_per_trip

Average number of passengers boarding at this stop per trip. Computed as total boardings divided by number of trips.

### average_offs_per_trip

Average number of passengers alighting at this stop per trip.

### average_load_per_trip

Average number of passengers on the vehicle after leaving this stop. Represents crowdedness.

### num_trips

Number of trips included in the calculation. Smaller values indicate less reliable averages.

### ons_all_trips

Total number of boardings at this stop across all trips in the dataset.
