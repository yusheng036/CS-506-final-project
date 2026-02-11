CS 506 Proposal

Project Description:

The project our team intends to develop is a website that helps passengers identify the least crowded bus travel time periods across Boston by analyzing load patterns by route, stop, and time of day. Users will input their intended destination, and the system will provide estimated bus crowding levels based on historical data from the MBTA Rider Census and the MBTA data handling team. The website would provide tables and graphs throughout the day to show the most appropriate time to take the bus to avoid the crowd. 

The primary goal of this project is to provide additional support for mobility-impaired individuals and wheelchair users by helping them identify more accessible and comfortable travel times on public transportation. It will also benefit passengers who experience discomfort in tightly crowded spaces when using MBTA buses.

Project Timeline:
* Weeks 1–2: Collect MBTA Rider Census data, analyze patterns within the datasets, and produce initial tables and graphs summarizing bus crowding trends.
* Weeks 3–4: Implement destination-based matching using an API and route-matching logic, and develop a system that recommends the least crowded travel windows.
* Weeks 5–6: Build the user interface and provide flexibility for users to input destinations and view crowding information.
* Weeks 7–8: Conduct user testing within the team, address bugs and edge cases, and prepare the final project presentation.


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
