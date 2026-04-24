import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent

models = ["Linear\nRegression", "XGBoost", "KNN (k=3)"]
mae    = [5.003, 3.596, 3.077]
rmse   = [7.045,  5.331, 4.994]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

bars1 = ax.bar(x - width/2, mae, width, label="Test MAE", color="#185FA5")
bars2 = ax.bar(x + width/2, [0 if v is None else v for v in rmse],
               width, label="Test RMSE", color="#F28C28")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)
for bar, v in zip(bars2, rmse):
    if v is None:
        bar.set_visible(False)
    else:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)

ax.set_ylabel("Error (passengers)", fontsize=12)
ax.set_title("Model Comparison: Test MAE & RMSE\n(predicting average bus load)", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e0e0e0", linewidth=0.7)
ax.set_ylim(0, 6.5)

plt.tight_layout()
plt.savefig(project_root / "diagram/model/model_comparison.png", dpi=150)
plt.show()
print("Saved: model_comparison.png")