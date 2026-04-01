import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

DB_PATH = "../dataset/Ridership.sqlite"
TABLE_NAME = "MBTA_Ridership"

conn = sqlite3.connect(DB_PATH)

query = f"""
SELECT
    day_type_name,
    AVG(average_load) AS avg_load
FROM {TABLE_NAME}
WHERE day_type_name IS NOT NULL
  AND average_load IS NOT NULL
GROUP BY day_type_name
"""

df = pd.read_sql_query(query, conn)
conn.close()

# normalize labels
df["day_type_name"] = df["day_type_name"].astype(str).str.strip().str.lower()

plt.figure(figsize=(7, 5))

plt.bar(df["day_type_name"], df["avg_load"], color="#185FA5")

plt.xlabel("Day Type")
plt.ylabel("Average passenger load per trip")
plt.title("MBTA bus ridership by day type")

plt.grid(axis="y", color="#e0e0e0", linewidth=0.7)
plt.tight_layout()

plt.savefig("daytype_visualization.png", dpi=150)
plt.show()

print(df)
print("Saved: daytype_visualization.png")