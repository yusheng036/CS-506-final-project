import sqlite3
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

project_root = Path(__file__).resolve().parent.parent.parent
db_path = project_root / "dataset" / "Ridership_v1.sqlite"

print("DB path:", db_path)
print("Exists:", db_path.exists())

conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT * FROM MBTA_Ridership", conn)
conn.close()

print(f"Loaded {len(df)} rows")

# =========================
# 1. Clean data
# =========================
# Drop columns unused in modeling (mirrors model.py)
drop_cols = ["mode", "route_id", "route_variant", "day_type_id", "time_period_id", "ObjectId"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Remove rows with missing target or negative passenger counts
df = df.dropna(subset=["average_load"])
df = df[df["average_offs"] >= 0]

# Normalize day type to lowercase to match raw DB values (weekday/saturday/sunday)
df["day_type_name"] = df["day_type_name"].astype(str).str.strip().str.lower()

# Extract year from season string e.g. "Fall 2023" -> 2023
# Only Fall seasons exist in this dataset
df["season_year"] = df["season"].str.extract(r"(\d{4})")[0]
df = df.dropna(subset=["season_year"])
df["season_year"] = df["season_year"].astype(int)
df = df.drop(columns=["season"])

# Encode time periods as ordered integers to preserve temporal meaning
# e.g. VERY_EARLY_MORNING=0, EARLY_AM=1, ..., OFF_PEAK=9
TIME_ORDER = [
    "VERY_EARLY_MORNING", "EARLY_AM", "AM_PEAK", "MIDDAY_BASE",
    "MIDDAY_SCHOOL", "PM_PEAK", "EVENING", "LATE_EVENING", "NIGHT", "OFF_PEAK"
]
time_map = {t: i for i, t in enumerate(TIME_ORDER)}
df["time_period_encoded"] = df["time_period_name"].map(time_map)
df = df.dropna(subset=["time_period_encoded"])
df["time_period_encoded"] = df["time_period_encoded"].astype(int)

print(f"Rows after cleaning: {len(df)}")

# =========================
# 2. Feature setup
# =========================
TARGET = "average_load"  # avg passengers on bus after leaving a stop = crowding proxy

# Features chosen to capture route identity, stop position, travel context, and time
# Note: route_name and day_type_name are categorical, label-encoded as integers
# This is a known limitation for KNN (distance metric treats encoded ints as ordinal)
# but avoids high dimensionality from one-hot encoding across many routes
FEATURES = [
    "route_name",         # which bus route
    "stop_sequence",      # position along route
    "direction_id",       # inbound (1) or outbound (0)
    "day_type_name",      # weekday / saturday / sunday
    "time_period_encoded",# time of day bucket
    "season_year",        # year (captures long-term ridership trends)
    "num_trips",          # number of trips averaged over (data reliability indicator)
]

# Label encode categorical columns
for col in ["route_name", "day_type_name"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# =========================
# 3. Chronological train/val/test split
# =========================
# Train on historical data, validate on 2023, test on most recent year (2024)
# This simulates real-world generalization to future seasons
train = df[df["season_year"] <= 2022]
val   = df[df["season_year"] == 2023]
test  = df[df["season_year"] == 2024]

X_train, y_train = train[FEATURES], train[TARGET]
X_val,   y_val   = val[FEATURES],   val[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# =========================
# 4. Tune k on validation set
# =========================
# KNN predicts average_load as the mean of the k nearest neighbors in feature space
# StandardScaler is required: without it, features with large numeric ranges
# (e.g. route_name encoded as 0-200) would dominate the Euclidean distance calculation
k_values = [3, 5, 7, 10, 15, 20]
val_maes = []

for k in k_values:
    pipe = Pipeline([
        ("scaler", StandardScaler()),  # normalize all features to mean=0, std=1
        ("knn", KNeighborsRegressor(n_neighbors=k, n_jobs=-1))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    val_maes.append(mae)
    print(f"  k={k:2d}  Val MAE: {mae:.3f}")

best_k = k_values[val_maes.index(min(val_maes))]
print(f"\nBest k: {best_k}")

# =========================
# 5. Final model with best k
# =========================
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=best_k, n_jobs=-1))
])
final_model.fit(X_train, y_train)

val_preds  = final_model.predict(X_val)
test_preds = final_model.predict(X_test)

val_mae   = mean_absolute_error(y_val, val_preds)
val_rmse  = root_mean_squared_error(y_val, val_preds)
test_mae  = mean_absolute_error(y_test, test_preds)
test_rmse = root_mean_squared_error(y_test, test_preds)

print(f"\n{'='*45}")
print(f"KNN (k={best_k}) — Val  MAE: {val_mae:.3f}   RMSE: {val_rmse:.3f}")
print(f"KNN (k={best_k}) — Test MAE: {test_mae:.3f}   RMSE: {test_rmse:.3f}")
print(f"{'='*45}")

# =========================
# 6. Plot: k vs Validation MAE
# =========================
# Shows how model performance changes with k — justifies our choice of best_k
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(k_values, val_maes, marker="o", color="#185FA5", linewidth=2)
ax.axvline(best_k, color="crimson", linestyle="--", label=f"Best k={best_k}")
ax.set_xlabel("k (number of neighbors)", fontsize=12)
ax.set_ylabel("Validation MAE", fontsize=12)
ax.set_title("KNN: Validation MAE vs k", fontsize=13)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e0e0e0", linewidth=0.7)
plt.tight_layout()
plt.savefig(project_root / "diagram/model/knn_k_selection.png", dpi=150)
plt.show()
print("Saved: knn_k_selection.png")

# =========================
# 7. Plot: Actual vs Predicted (test set sample)
# =========================
# Visualizes model fit on unseen 2024 data — supports claim about prediction quality
sample = min(300, len(y_test))

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test.values[:sample], test_preds[:sample], alpha=0.3, s=10, color="steelblue")
ax.plot([0, y_test.max()], [0, y_test.max()], color="crimson", linewidth=1, label="Perfect fit")
ax.set_xlabel("Actual Average Load", fontsize=12)
ax.set_ylabel("Predicted Average Load", fontsize=12)
ax.set_title(f"KNN (k={best_k}): Actual vs Predicted — 2024 Test Set", fontsize=13)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e0e0e0", linewidth=0.7)
plt.tight_layout()
plt.savefig(project_root / "diagram/model/knn_actual_vs_predicted.png", dpi=150)
plt.show()
print("Saved: knn_actual_vs_predicted.png")