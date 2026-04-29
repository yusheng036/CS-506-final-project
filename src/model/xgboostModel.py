import sqlite3
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import LabelEncoder

project_root = Path(__file__).resolve().parent.parent.parent
db_path = project_root / "dataset" / "Ridership_v1.sqlite"

print("DB path:", db_path)
print("Exists:", db_path.exists())

conn = sqlite3.connect(db_path)

cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:", cursor.fetchall())

df = pd.read_sql_query("SELECT * FROM MBTA_Ridership", conn)
conn.close()

print(f"Loaded {len(df)} rows")
print(df.head())

# Clean
drop_cols = [
    "mode", "route_id", "route_variant",
    "day_type_id", "time_period_id", "ObjectId"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

df = df.dropna(subset=["average_load"])
df = df[df["average_offs"] >= 0]
df["day_type_name"] = df["day_type_name"].astype(str).str.strip().str.title()

df["season_year"] = df["season"].str.extract(r"(\d{4})")[0]
df = df.dropna(subset=["season_year"])
df["season_year"] = df["season_year"].astype(int)
df = df.drop(columns=["season"])

# time
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
    "route_name", "stop_sequence", "direction_id",
    "day_type_name", "time_period_encoded",
    "season_year", "num_trips",
]

for col in ["route_name", "day_type_name"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

train = df[df["season_year"] <= 2022]
val   = df[df["season_year"] == 2023]
test  = df[df["season_year"] == 2024]

X_train, y_train = train[FEATURES], train[TARGET]
X_val,   y_val   = val[FEATURES],   val[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

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

b_mae = mean_absolute_error(y_val, val_b["baseline_pred"])
b_rmse = root_mean_squared_error(y_val, val_b["baseline_pred"])
print(f"\nBaseline — MAE: {b_mae:.3f}  RMSE: {b_rmse:.3f}")

# Train
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

# Evaluate
val_preds = model.predict(X_val)
test_preds = model.predict(X_test)

print(f"\n{'='*45}")
print(f"Baseline  — Val  MAE: {b_mae:.3f}   RMSE: {b_rmse:.3f}")
print(f"XGBoost   — Val  MAE: {mean_absolute_error(y_val, val_preds):.3f}   RMSE: {root_mean_squared_error(y_val, val_preds):.3f}")
print(f"XGBoost   — Test MAE: {mean_absolute_error(y_test, test_preds):.3f}   RMSE: {root_mean_squared_error(y_test, test_preds):.3f}")
print(f"{'='*45}")

# Plot
importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
importance.plot(kind="barh", ax=ax, color="#185FA5")
ax.set_xlabel("Feature importance (gain)", fontsize=12)
ax.set_title("MBTA average load", fontsize=13, fontweight="normal")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#e0e0e0", linewidth=0.7)

plt.tight_layout()
plt.savefig(project_root / "diagram/model/xgboost_feature_importance.png", dpi=150)
plt.show()
print("Saved: mbta_feature_importance.png")