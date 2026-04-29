import sqlite3
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

project_root = Path(__file__).resolve().parent.parent.parent
db_path = project_root / "dataset" / "Ridership_v1.sqlite"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:", cursor.fetchall())

df = pd.read_sql_query("SELECT * FROM MBTA_Ridership", conn)
conn.close()

print(f"Loaded {len(df)} rows")

# Clean
drop_cols = ["mode", "route_id", "route_variant", "day_type_id", "time_period_id", "ObjectId"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

df = df.dropna(subset=["average_load"])
df = df[df["average_offs"] >= 0]
df["day_type_name"] = df["day_type_name"].astype(str).str.strip().str.title()

# Keep season name (Spring/Fall/Summer/Winter) as a feature — more signal than year alone
df["season_name"] = df["season"].str.extract(r"([A-Za-z]+)")[0].str.title()
df["season_year"] = df["season"].str.extract(r"(\d{4})")[0]
df = df.dropna(subset=["season_year", "season_name"])
df["season_year"] = df["season_year"].astype(int)
df = df.drop(columns=["season"])

TIME_ORDER = [
    "VERY_EARLY_MORNING", "EARLY_AM", "AM_PEAK", "MIDDAY_BASE",
    "MIDDAY_SCHOOL", "PM_PEAK", "EVENING", "LATE_EVENING", "NIGHT", "OFF_PEAK"
]
time_map = {t: i for i, t in enumerate(TIME_ORDER)}
df["time_period_encoded"] = df["time_period_name"].map(time_map)
df = df.dropna(subset=["time_period_encoded"])
df["time_period_encoded"] = df["time_period_encoded"].astype(int)

print(f"Rows after cleaning: {len(df)}")

TARGET = "average_load"
FEATURES = [
    "route_name", "stop_id", "stop_sequence", "direction_id",
    "day_type_name", "time_period_encoded",
    "season_name", "season_year", "num_trips",
]

le_route = LabelEncoder()
df["route_name"] = le_route.fit_transform(df["route_name"].astype(str))

le_day = LabelEncoder()
df["day_type_name"] = le_day.fit_transform(df["day_type_name"].astype(str))

le_season = LabelEncoder()
df["season_name"] = le_season.fit_transform(df["season_name"].astype(str))

train = df[df["season_year"] <= 2022]
val   = df[df["season_year"] == 2023]
test  = df[df["season_year"] == 2024]

X_train, y_train = train[FEATURES], train[TARGET]
X_val,   y_val   = val[FEATURES],   val[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# Baseline: mean load per route/time/day_type from training set
baseline_lookup = (
    train.groupby(["route_name", "time_period_encoded", "day_type_name"])[TARGET]
    .mean()
    .reset_index()
    .rename(columns={TARGET: "baseline_pred"})
)
val_b = val[FEATURES + [TARGET]].merge(
    baseline_lookup,
    on=["route_name", "time_period_encoded", "day_type_name"],
    how="left"
)
val_b["baseline_pred"] = val_b["baseline_pred"].fillna(y_train.mean())

b_mae  = mean_absolute_error(y_val, val_b["baseline_pred"])
b_rmse = rmse(y_val, val_b["baseline_pred"])
print(f"\nBaseline  — Val  MAE: {b_mae:.3f}   RMSE: {b_rmse:.3f}")

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_leaf=2,
    max_features=0.5,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

val_preds  = model.predict(X_val)
test_preds = model.predict(X_test)

print(f"\n{'='*50}")
print(f"Baseline      — Val  MAE: {b_mae:.3f}   RMSE: {b_rmse:.3f}")
print(f"Random Forest — Val  MAE: {mean_absolute_error(y_val, val_preds):.3f}   RMSE: {rmse(y_val, val_preds):.3f}")
print(f"Random Forest — Test MAE: {mean_absolute_error(y_test, test_preds):.3f}   RMSE: {rmse(y_test, test_preds):.3f}")
print(f"{'='*50}")

# Feature importance plot
importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
importance.plot(kind="barh", ax=ax, color="#2e7d32")
ax.set_xlabel("Feature importance (mean decrease impurity)", fontsize=12)
ax.set_title("Random Forest — MBTA average load", fontsize=13, fontweight="normal")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#e0e0e0", linewidth=0.7)

plt.tight_layout()
out_path = project_root / "diagram" / "model" / "rf_feature_importance.png"
plt.savefig(out_path, dpi=150)
plt.show()
print(f"Saved: {out_path}")

# Actual vs Predicted scatter plot (test set sample)
sample = min(300, len(y_test))
max_val = max(y_test.values[:sample].max(), test_preds[:sample].max())

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test.values[:sample], test_preds[:sample], alpha=0.3, s=10, color="#2e7d32")
ax.plot([0, max_val], [0, max_val], color="crimson", linewidth=1, label="Perfect fit")
ax.set_xlabel("Actual Average Load", fontsize=12)
ax.set_ylabel("Predicted Average Load", fontsize=12)
ax.set_title("Random Forest: Actual vs Predicted — 2024 Test Set", fontsize=13)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#e0e0e0", linewidth=0.7)

plt.tight_layout()
scatter_path = project_root / "diagram" / "model" / "rf_actual_vs_predicted.png"
plt.savefig(scatter_path, dpi=150)
plt.show()
print(f"Saved: {scatter_path}")

# =========================
# Predict least crowded time period per route
# =========================
# Use the trained model to predict average_load for all route/time/day combinations
# This directly answers project goal 4: identify least crowded travel periods

# Build prediction grid: all routes x all time periods x weekday
routes = sorted(df["route_name"].unique())
time_periods = list(range(len(TIME_ORDER)))
day_encoded = 2
season_encoded = 0
season_year = 2024
num_trips_median = int(df["num_trips"].median())
stop_sequence_median = int(df["stop_sequence"].median())
stop_id_median = int(df["stop_id"].median())
direction = 0  # outbound

rows = []
for route in routes:
    for tp in time_periods:
        rows.append({
            "route_name": route,
            "stop_id": stop_id_median,
            "stop_sequence": stop_sequence_median,
            "direction_id": direction,
            "day_type_name": day_encoded,
            "time_period_encoded": tp,
            "season_name": season_encoded,
            "season_year": season_year,
            "num_trips": num_trips_median,
        })

grid_df = pd.DataFrame(rows)
grid_df["predicted_load"] = model.predict(grid_df[FEATURES])

# For each route, find the least crowded time period
grid_df["time_period_name"] = grid_df["time_period_encoded"].map(
    {i: t for i, t in enumerate(TIME_ORDER)}
)

least_crowded = (
    grid_df.loc[grid_df.groupby("route_name")["predicted_load"].idxmin()]
    [["route_name", "time_period_name", "predicted_load"]]
    .reset_index(drop=True)
)

# Decode route name back to original
least_crowded["route_name"] = le_route.inverse_transform(
    least_crowded["route_name"].astype(int)
)

print("\n=== Least Crowded Time Period per Route (weekday, Random Forest) ===")
print(least_crowded.to_string(index=False))

# Save to CSV
out_csv = project_root / "diagram" / "model" / "least_crowded_per_route.csv"
least_crowded.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

# Plot: top 5 busiest routes, load across all time periods
top5_routes = (
    grid_df.groupby("route_name")["predicted_load"]
    .mean()
    .nlargest(5)
    .index.tolist()
)

top5_decoded = le_route.inverse_transform(top5_routes)

fig, ax = plt.subplots(figsize=(12, 5))
colors = ["#2e7d32", "#185FA5", "#F28C28", "#9C27B0", "#E53935"]

for i, (route_enc, route_dec) in enumerate(zip(top5_routes, top5_decoded)):
    route_data = grid_df[grid_df["route_name"] == route_enc].sort_values("time_period_encoded")
    ax.plot(
        route_data["time_period_name"],
        route_data["predicted_load"],
        marker="o", linewidth=2,
        color=colors[i],
        label=f"Route {route_dec}"
    )

ax.set_xlabel("Time Period", fontsize=12)
ax.set_ylabel("Predicted Average Load", fontsize=12)
ax.set_title(
    "Top 5 Busiest Routes: Predicted Load Across Time Periods\n(weekday, Random Forest)",
    fontsize=13
)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e0e0e0", linewidth=0.7)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

out_plot = project_root / "diagram" / "model" / "rf_load_by_time.png"
plt.savefig(out_plot, dpi=150)
plt.show()
print(f"Saved: {out_plot}")