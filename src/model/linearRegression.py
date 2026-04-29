from pathlib import Path
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

project_root = Path(__file__).resolve().parent.parent.parent
DB_PATH = project_root / "dataset" / "Ridership_v1.sqlite"
TABLE_NAME = "MBTA_Ridership"

# =========================
# 1. Load data
# =========================
conn = sqlite3.connect(DB_PATH)

query = f"""
SELECT
    route_name,
    direction_id,
    day_type_name,
    time_period_name,
    average_load
FROM {TABLE_NAME}
WHERE route_name IS NOT NULL
  AND direction_id IS NOT NULL
  AND day_type_name IS NOT NULL
  AND time_period_name IS NOT NULL
  AND average_load IS NOT NULL
"""

df = pd.read_sql_query(query, conn)
conn.close()

# =========================
# 2. Basic cleaning
# =========================
df["route_name"] = df["route_name"].astype(str).str.strip()
df["day_type_name"] = df["day_type_name"].astype(str).str.strip().str.lower()
df["time_period_name"] = df["time_period_name"].astype(str).str.strip()
df["direction_id"] = pd.to_numeric(df["direction_id"], errors="coerce")
df["average_load"] = pd.to_numeric(df["average_load"], errors="coerce")

df = df.dropna()
df = df[df["average_load"] >= 0]

# =========================
# 3. Train model
# =========================
X = df[["route_name", "direction_id", "day_type_name", "time_period_name"]]
y = df["average_load"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            ["route_name", "direction_id", "day_type_name", "time_period_name"]
        )
    ]
)

model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =========================
# 4. Evaluation
# =========================
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print("\n=== Linear Regression Baseline ===")
print(f"MAE: {mae:.4f}")
print(f"R^2: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# =========================
# 5. Choose one route/day to compare all time periods
# =========================
route = "1"
direction = 0
day = "weekday"

time_periods = sorted(df["time_period_name"].dropna().unique())

scenario_df = pd.DataFrame({
    "route_name": [route] * len(time_periods),
    "direction_id": [direction] * len(time_periods),
    "day_type_name": [day] * len(time_periods),
    "time_period_name": time_periods
})

scenario_df["predicted_load"] = model.predict(scenario_df)

# =========================
# 6. Compute actual averages for same route/day
# =========================
actual_df = df[
    (df["route_name"] == route) &
    (df["direction_id"] == direction) &
    (df["day_type_name"] == day)
]

actual_grouped = (
    actual_df.groupby("time_period_name", as_index=False)["average_load"]
    .mean()
    .rename(columns={"average_load": "actual_load"})
)

# =========================
# 7. Merge actual + predicted
# =========================
merged = pd.merge(
    scenario_df,
    actual_grouped,
    on="time_period_name",
    how="left"
)

merged = merged.sort_values("predicted_load").reset_index(drop=True)

print("\n=== Actual vs Predicted by Time Period ===")
print(merged[["time_period_name", "actual_load", "predicted_load"]])

# =========================
# 8. Highlight least crowded time
# =========================
least_idx = merged["predicted_load"].idxmin()
least_time = merged.loc[least_idx, "time_period_name"]
least_value = merged.loc[least_idx, "predicted_load"]

print(f"\nLeast crowded predicted time: {least_time} ({least_value:.2f})")

# =========================
# 9. Double bar chart with highlight
# =========================
x = np.arange(len(merged))
width = 0.38

actual_colors = ["steelblue"] * len(merged)
pred_colors = ["orange"] * len(merged)

# highlight the least crowded predicted bar
pred_colors[least_idx] = "crimson"

plt.figure(figsize=(10, 5))

plt.bar(x - width / 2, merged["actual_load"], width, label="Actual", color=actual_colors)
plt.bar(x + width / 2, merged["predicted_load"], width, label="Predicted", color=pred_colors)

plt.xticks(x, merged["time_period_name"], rotation=30)
plt.ylabel("Average Load")
plt.title(f"Actual vs Predicted by Time Period (Route {route}, {day})")
legend_handles = [
    Patch(facecolor="steelblue", label="Actual"),
    Patch(facecolor="orange", label="Predicted"),
    Patch(facecolor="crimson", label="Least")
]

plt.legend(handles=legend_handles)

# annotate least crowded predicted time
plt.annotate(
    f"Least crowded:\n{least_time}",
    xy=(least_idx + width / 2, least_value),
    xytext=(least_idx + 0.6, least_value + 2),
    arrowprops=dict(arrowstyle="->"),
    fontsize=9
)

plt.tight_layout()
plt.savefig(project_root / "diagram/model/linearRegressionPredict.png", dpi=150)
plt.show()

print("Saved: actual_vs_predicted_highlight.png")