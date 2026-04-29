import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

conn = sqlite3.connect("Ridership copy.sqlite")

query = """
SELECT
    time_period_id,
    time_period_name,
    season,
    route_id,
    day_type_name,
    AVG(average_load)   AS avg_load,
    AVG(average_ons)    AS avg_ons,
    AVG(average_offs)   AS avg_offs,
    SUM(ons_all_trips)  AS total_ons
FROM MBTA_Ridership
GROUP BY time_period_id, time_period_name, season, route_id, day_type_name
ORDER BY time_period_id
"""

df = pd.read_sql_query(query, conn)
conn.close()

TIME_ORDER = [
    "VERY_EARLY_MORNING",
    "EARLY_AM",
    "AM_PEAK",
    "MIDDAY_BASE",
    "MIDDAY_SCHOOL",
    "PM_PEAK",
    "EVENING",
    "LATE_EVENING",
    "NIGHT",
    "OFF_PEAK",
]

TIME_LABELS = {
    "VERY_EARLY_MORNING": "Very Early",
    "EARLY_AM":           "Early AM",
    "AM_PEAK":            "AM Peak",
    "MIDDAY_BASE":        "Midday",
    "MIDDAY_SCHOOL":      "Midday School",
    "PM_PEAK":            "PM Peak",
    "EVENING":            "Evening",
    "LATE_EVENING":       "Late Evening",
    "NIGHT":              "Night",
    "OFF_PEAK":           "Off Peak",
}

df["time_period_name"] = pd.Categorical(
    df["time_period_name"], categories=TIME_ORDER, ordered=True
)

METRIC   = "avg_load"
SEASON   = None
DAY_TYPE = "Weekday"
TOP_N    = 5

filtered = df.copy()
if SEASON:
    filtered = filtered[filtered["season"].str.lower() == SEASON.lower()]
if DAY_TYPE:
    filtered = filtered[filtered["day_type_name"].str.lower() == DAY_TYPE.lower()]

print(f"Rows after filter: {len(filtered)}")
print(f"Day types in data: {df['day_type_name'].unique().tolist()}")
print(f"Seasons in data:   {df['season'].unique().tolist()}")

top_routes = (
    filtered.groupby("route_id")[METRIC]
    .mean()
    .nlargest(TOP_N)
    .index.tolist()
)
print(f"Top {TOP_N} routes: {top_routes}")
filtered = filtered[filtered["route_id"].isin(top_routes)]

pivot = (
    filtered.groupby(["time_period_name", "route_id"])[METRIC]
    .mean()
    .unstack("route_id")
    .reindex(TIME_ORDER)
)

x_labels = [TIME_LABELS.get(t, t) for t in TIME_ORDER]

COLORS = ["#185FA5", "#0F6E56", "#993C1D", "#7F77DD", "#BA7517",
          "#1D9E75", "#D85A30", "#534AB7", "#854F0B"]

METRIC_LABELS = {
    "avg_load":  "Avg passenger load per trip",
    "avg_ons":   "Avg boardings per trip",
    "avg_offs":  "Avg alightings per trip",
    "total_ons": "Total boardings (all trips)",
}

fig, ax = plt.subplots(figsize=(12, 5))
for i, col in enumerate(pivot.columns):
    ax.plot(
        x_labels,
        pivot[col].values,
        marker="o",
        linewidth=2,
        color=COLORS[i % len(COLORS)],
        label=f"Route {col}",
    )

ax.set_xlabel("Time period", fontsize=12)
ax.set_ylabel(METRIC_LABELS.get(METRIC, METRIC), fontsize=12)
title_parts = ["MBTA bus ridership"]
if DAY_TYPE:
    title_parts.append(DAY_TYPE.lower())
if SEASON:
    title_parts.append(SEASON.lower())
ax.set_title(" — ".join(title_parts), fontsize=14, fontweight="normal")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(axis="y", color="#e0e0e0", linewidth=0.7)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(title="Route", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("mbta_ridership_line_chart.png", dpi=150)
plt.show()
print("Saved: mbta_ridership_line_chart.png")