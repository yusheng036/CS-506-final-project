import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect("Ridership copy.sqlite")

query = """
SELECT stop_name, AVG(average_load) AS avg_load
FROM MBTA_Ridership
WHERE season LIKE "%2024" AND route_id = 1
GROUP BY stop_name
ORDER BY avg_load DESC
"""

df = pd.read_sql_query(query, conn)

plt.figure(figsize=(10, 6))
plt.bar(df["stop_name"], df["avg_load"], color="skyblue")
plt.title("")
plt.xlabel("")
plt.ylabel("")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()