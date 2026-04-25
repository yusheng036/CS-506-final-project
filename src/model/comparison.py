import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent

# Test MAE and RMSE for each model (predicting average_load on 2024 test set)
models = [
    "Linear\nRegression",
    "KNN\n(k=3)",
    "XGBoost",
    "Random\nForest",
]
mae  = [5.003, 3.077, 3.596, 2.741]
rmse = [7.045, 4.994, 5.331, 4.193]

# Baseline: mean-lookup (route × time × day_type from training set)
baseline_mae  = 4.344
baseline_rmse = 6.050

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

bars1 = ax.bar(x - width / 2, mae,  width, label="Test MAE",  color="#185FA5")
bars2 = ax.bar(x + width / 2, rmse, width, label="Test RMSE", color="#F28C28")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)

# Baseline reference lines
ax.axhline(baseline_mae,  color="#185FA5", linestyle="--", linewidth=1.2,
           label=f"Baseline MAE ({baseline_mae})")
ax.axhline(baseline_rmse, color="#F28C28", linestyle="--", linewidth=1.2,
           label=f"Baseline RMSE ({baseline_rmse})")

ax.set_ylabel("Error (passengers)", fontsize=12)
ax.set_title("Model Comparison: Test MAE & RMSE\n(predicting average bus load)", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e0e0e0", linewidth=0.7)
ax.set_ylim(0, 8.5)

plt.tight_layout()
out_path = project_root / "diagram" / "model" / "model_comparison.png"
plt.savefig(out_path, dpi=150)
plt.show()
print(f"Saved: {out_path}")
