import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

conn = sqlite3.connect("Ridership copy.sqlite")

query = """
SELECT
    route_id,
    AVG(average_load)   AS avg_load,
    AVG(average_ons)    AS avg_ons,
    AVG(average_offs)   AS avg_offs,
    SUM(ons_all_trips)  AS total_ons
FROM MBTA_Ridership
GROUP BY route_id
ORDER BY avg_load DESC
"""

df = pd.read_sql_query(query, conn)
conn.close()


METRIC   = "avg_load"
TOP_N    = 20

df = df.nlargest(TOP_N, METRIC)
df["route_label"] = "Route " + df["route_id"].astype(str)

METRIC_LABELS = {
    "avg_load":  "Avg passenger load per trip",
    "avg_ons":   "Avg boardings per trip",
    "avg_offs":  "Avg alightings per trip",
    "total_ons": "Total boardings (all trips)",
}

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(
    df["route_label"],
    df[METRIC],
    color="#185FA5",
    width=0.6,
)

ax.set_xlabel("Route", fontsize=12)
ax.set_ylabel(METRIC_LABELS.get(METRIC, METRIC), fontsize=12)
ax.set_title(f"MBTA bus — {METRIC_LABELS.get(METRIC, METRIC)} by route (top {TOP_N})", fontsize=14, fontweight="normal")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.1f}"))
ax.grid(axis="y", color="#e0e0e0", linewidth=0.7)
ax.spines[["top", "right"]].set_visible(False)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("mbta_avg_load_per_route.png", dpi=150)
plt.show()
print("Saved: mbta_avg_load_per_route.png")