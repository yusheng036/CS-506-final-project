import sqlite3
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

project_root = Path(__file__).resolve().parent.parent.parent
# print(project_root)
db_path = project_root / "dataset" / "Ridership_v1.sqlite"
# print(db_path)

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

# Train Neural Network

categorical_features = [
    "route_name",
    "direction_id",
    "day_type_name",
    "time_period_encoded",
    "season_name",
]

numeric_features = [
    "stop_sequence",
    "season_year",
    "num_trips",
]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ("num", StandardScaler(), numeric_features),
])

X_train_p = preprocess.fit_transform(X_train)
X_val_p = preprocess.transform(X_val)
X_test_p = preprocess.transform(X_test)

model = keras.Sequential([
    keras.layers.Input(shape=(X_train_p.shape[1],)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train_p,
    y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_val_p, y_val),
    callbacks=[early_stop],
    verbose=1
)

val_preds  = model.predict(X_val_p)
test_preds = model.predict(X_test_p)

test_preds = np.asarray(test_preds).ravel()
val_preds = np.asarray(val_preds).ravel()
y_test_arr = y_test.values.ravel()

print(f"Baseline       — Val  MAE: {b_mae:.3f}   RMSE: {b_rmse:.3f}")
print(f"Neural Network — Val  MAE: {mean_absolute_error(y_val, val_preds):.3f}   RMSE: {rmse(y_val, val_preds):.3f}")
print(f"Neural Network — Test MAE: {mean_absolute_error(y_test, test_preds):.3f}   RMSE: {rmse(y_test, test_preds):.3f}")
print("Done")

max_err_nn = np.max(np.abs(y_test_arr - test_preds))
print(f"Neural Network - Max Error: {max_err_nn:.3f}")

print(f"{'='*50}")


# Actual vs Predicted scatter plot (test set sample)
sample = min(300, len(y_test))
max_val = max(y_test.values[:sample].max(), test_preds[:sample].max())

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test.values[:sample], test_preds[:sample], alpha=0.3, s=10, color="#2e7d32")
ax.plot([0, max_val], [0, max_val], color="crimson", linewidth=1, label="Perfect fit")
ax.set_xlabel("Actual Average Load", fontsize=12)
ax.set_ylabel("Predicted Average Load", fontsize=12)
ax.set_title("Neural Network: Actual vs Predicted — 2024 Test Set", fontsize=13)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#e0e0e0", linewidth=0.7)

plt.tight_layout()
scatter_path = project_root / "diagram" / "model" / "nn_actual_vs_predicted.png"
plt.savefig(scatter_path, dpi=150)
plt.show()
print(f"Saved: {scatter_path}")
