import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

DB_PATH = "../dataset/Ridership.sqlite"
TABLE_NAME = "MBTA_Ridership"

conn = sqlite3.connect(DB_PATH)

query = f"""
SELECT
    season,
    AVG(average_load) AS avg_load
FROM {TABLE_NAME}
WHERE season IS NOT NULL
  AND average_load IS NOT NULL
GROUP BY season
"""

df = pd.read_sql_query(query, conn)
conn.close()

df["season"] = df["season"].astype(str).str.strip()

# try to sort by year first, then season order
season_order_map = {
    "Winter": 1,
    "Spring": 2,
    "Summer": 3,
    "Fall": 4
}

def parse_year(season_str):
    parts = season_str.split()
    for p in parts:
        if p.isdigit():
            return int(p)
    return 9999

def parse_season_name(season_str):
    parts = season_str.split()
    for p in parts:
        if p in season_order_map:
            return p
    return "ZZZ"

df["year"] = df["season"].apply(parse_year)
df["season_name"] = df["season"].apply(parse_season_name)
df["season_rank"] = df["season_name"].map(season_order_map).fillna(99)

df = df.sort_values(["year", "season_rank", "season"])

plt.figure(figsize=(10, 5))
plt.bar(df["season"], df["avg_load"], color="#185FA5")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Season")
plt.ylabel("Average passenger load per trip")
plt.title("MBTA bus ridership — average passenger load by season")
plt.grid(axis="y", color="#e0e0e0", linewidth=0.7)
plt.tight_layout()
plt.savefig("season_visualization.png", dpi=150)
plt.show()

print(df[["season", "avg_load"]])
print("Saved: season_visualization.png")