import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect("dataset/Ridership.sqlite")

# Plot the results: WHERE (bar chart)
query = """
SELECT stop_name, AVG(average_load) AS avg_load
FROM MBTA_Ridership
WHERE season LIKE "%2024" AND route_name = 1
GROUP BY stop_name
ORDER BY avg_load DESC
"""

df = pd.read_sql_query(query, conn)

plt.figure(figsize=(10, 6))
plt.bar(df["stop_name"], df["avg_load"], color="skyblue")

plt.title("Average Passenger Load per Stop (Route 1, 2024)")
plt.xlabel("Bus Stop")
plt.ylabel("Average Passenger Load")

plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("diagram/load_per_stop.png")
plt.show()


# Plot the results: WHEN (line chart)
query_time = """
SELECT time_period_name, AVG(average_load) AS avg_load
FROM MBTA_Ridership
WHERE season LIKE "%2024" AND route_name = 1
GROUP BY time_period_name
"""

# order time periods
df_time = pd.read_sql_query(query_time, conn)

order = [
    "VERY_EARLY_MORNING",
    "EARLY_AM",
    "AM_PEAK",
    "MIDDAY_BASE",
    "MIDDAY_SCHOOL",
    "OFF_PEAK",
    "PM_PEAK",
    "EVENING",
    "LATE_EVENING",
    "NIGHT"
]

df_time["time_period_name"] = pd.Categorical(
    df_time["time_period_name"],
    categories=order,
    ordered=True
)

df_time = df_time.sort_values("time_period_name")

plt.figure(figsize=(8, 5))
plt.plot(df_time["time_period_name"], df_time["avg_load"], marker='o')

plt.title("Average Passenger Load by Time Period (Route 1, 2024)")
plt.xlabel("Time Period")
plt.ylabel("Average Passenger Load")

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("diagram/load_by_time.png")
plt.show()